import torch
from torch import nn

from model.layers.encoder import Encoder
from model.mention_detection import MentionDetector
from model.coreference_resolution import CoreferenceResolver
from model.entity_typing import EntityClassifier


class DocJEREModel(nn.Module):
    def __init__(
        self,
        model,
        tokenizer,
        n_entity_classes,
        hidden_size=768,
        block_size=64,
        max_height=42,
    ):
        super(DocJEREModel, self).__init__()
        self.hidden_size = hidden_size
        self.tokenizer = tokenizer
        self.encoder = Encoder(model, tokenizer)
        self.word_mlp = nn.Linear(model.config.hidden_size, hidden_size)

        # Mention Detection
        self.mention_detection = MentionDetector(hidden_size=hidden_size)

        self.threshold = torch.nn.Parameter(torch.tensor(0.5))

        # Coreference Resolution
        self.coreference_resolution = CoreferenceResolver(
            hidden_size=hidden_size, block_size=block_size, max_height=max_height
        )

        # Entity Typing
        self.entity_typing = EntityClassifier(
            hidden_size=hidden_size, num_labels=n_entity_classes
        )

        # self.rel_classifier = nn.Linear(self.model.config.hidden_size, rel_num_classes)

    def get_best_spans(self, span_scores, span_indices):
        best_spans = []
        for i in range(span_scores.size(0)):
            # Get the indices of the top spans for each batch item
            top_indices = torch.argsort(span_scores[i], descending=True)
            best_spans.append(span_indices[i, top_indices[0]].tolist())
        return torch.tensor(best_spans, dtype=torch.long)

    def decode_spans(self, input_ids, best_spans):
        decoded_spans = []
        for i, spans in enumerate(best_spans):
            start, end = spans
            # Decode the original text
            decoded_spans.append(self.tokenizer.decode(input_ids[i, start : end + 1]))
        return decoded_spans

    def forward(
        self,
        input_ids,
        attention_mask,
        span_idx,
        span_mask,
        span_labels,
        coreference_labels,
        hts,
        entity_pos,
        entity_types,
        eval_mode=False,
    ):
        embeddings, attention_mask = self.encoder(input_ids, attention_mask)
        B, L, D = embeddings.size()

        # Mention Detection
        (
            logits,
            md_loss,
            md_precision,
            md_recall,
            md_f1,
        ) = self.mention_detection(embeddings, span_idx, span_mask, span_labels)

        # Coreference Resolution
        coref_loss, coref_precision, coref_recall, coref_f1 = (
            self.coreference_resolution(
                embeddings, attention_mask, entity_pos, hts, coreference_labels
            )
        )

        # Entity Typing
        entity_clusters = self.get_coreference_clusters(
            hts, entity_pos, coreference_labels
        )

        entity_embeddings = self.get_entity_embeddings(
            embeddings, attention_mask, entity_clusters
        )
        (
            entity_typing_loss,
            entity_typing_precision,
            entity_typing_recall,
            entity_typing_f1,
        ) = self.entity_typing(entity_embeddings, entity_types)

        # Joint
        if eval_mode:
            predicted_entity_pos = self.get_predicted_entity_pos(
                logits.view(span_idx.size(0), span_idx.size(1), -1), span_idx
            )
            (
                joint_coref_precision,
                joint_coref_recall,
                joint_coref_f1,
            ) = self.coreference_resolution_joint(
                embeddings,
                attention_mask,
                entity_pos,
                hts,
                coreference_labels,
                predicted_entity_pos,
            )

        total_loss = md_loss + coref_loss + entity_typing_loss
        md_outputs = {
            "loss": md_loss,
            "precision": md_precision,
            "recall": md_recall,
            "f1": md_f1,
        }
        coref_outputs = {
            "loss": coref_loss,
            "precision": coref_precision,
            "recall": coref_recall,
            "f1": coref_f1,
        }
        entity_typing_outputs = {
            "loss": entity_typing_loss,
            "precision": entity_typing_precision,
            "recall": entity_typing_recall,
            "f1": entity_typing_f1,
        }

        if eval_mode:
            joint_coref_outputs = {
                "precision": joint_coref_precision,
                "recall": joint_coref_recall,
                "f1": joint_coref_f1,
            }
            return (
                total_loss,
                md_outputs,
                coref_outputs,
                joint_coref_outputs,
                entity_typing_outputs,
            )
        else:
            return (
                total_loss,
                md_outputs,
                coref_outputs,
                entity_typing_outputs,
            )

    def coreference_resolution_joint(
        self,
        x,
        attention_mask,
        entity_pos,
        hts,
        coreference_labels,
        predicted_entity_pos,
    ):
        B, L, D = x.size()
        joint_entity_pos, joint_hts, joint_coreference_labels, proportions = (
            self.preprocess_joint_corefs(
                entity_pos, predicted_entity_pos, hts, coreference_labels
            )
        )
        joint_coref_precision = joint_coref_recall = joint_coref_f1 = 0
        for i in range(B):
            if proportions[i] > 0.1 and len(joint_entity_pos[i]) > 1:
                i_entity_pos = [joint_entity_pos[i]]
                i_hts = [joint_hts[i]]
                i_coreference_labels = [joint_coreference_labels[i]]
                _, coref_precision, coref_recall, coref_f1 = (
                    self.coreference_resolution(
                        x,
                        attention_mask,
                        i_entity_pos,
                        i_hts,
                        i_coreference_labels,
                    )
                )
                joint_coref_precision += coref_precision * proportions[i]
                joint_coref_recall += coref_recall * proportions[i]
                joint_coref_f1 += coref_f1 * proportions[i]
            else:
                joint_coref_precision += 0
                joint_coref_recall += 0
                joint_coref_f1 += 0
        return (
            joint_coref_precision / B,
            joint_coref_recall / B,
            joint_coref_f1 / B,
        )

    def preprocess_joint_corefs(
        self, gt_entity_pos, predicted_entity_pos, hts, coreference_labels
    ):
        B = len(gt_entity_pos)
        joint_ent_proportions = []
        joint_entity_pos, joint_hts, joint_coreference_labels = [], [], []
        for b in range(B):
            b_joint_entity_pos, b_joint_hts, b_joint_coreference_labels = [], [], []
            b_joint_entity_pos = list(
                set(
                    [
                        (entity[0], entity[1])
                        for entity in predicted_entity_pos[b]
                        if (entity[0], entity[1]) in gt_entity_pos[b]
                    ]
                )
            )
            joint_ent_proportions.append(
                len(b_joint_entity_pos) / len(gt_entity_pos[b])
            )
            b_joint_entity_pos.sort()
            for i in range(len(b_joint_entity_pos) - 1):
                for j in range(i + 1, len(b_joint_entity_pos)):
                    index_ent1 = gt_entity_pos[b].index(b_joint_entity_pos[i])
                    index_ent2 = gt_entity_pos[b].index(b_joint_entity_pos[j])
                    b_joint_hts.append((i, j))
                    b_joint_coreference_labels.append(
                        coreference_labels[b][
                            hts[b].index((index_ent1, index_ent2))
                        ].tolist()
                    )
            joint_entity_pos.append(b_joint_entity_pos)
            joint_ent_proportions.append(
                len(b_joint_entity_pos) / len(gt_entity_pos[b])
            )

            joint_hts.append(b_joint_hts)
            joint_coreference_labels.append(b_joint_coreference_labels)
        joint_coreference_labels = [
            torch.tensor(label) for label in joint_coreference_labels
        ]

        print("Proportion of joint entities:", joint_ent_proportions)
        return (
            joint_entity_pos,
            joint_hts,
            joint_coreference_labels,
            joint_ent_proportions,
        )

    def get_predicted_entity_pos(self, logits, span_idx):
        logits = torch.softmax(logits, dim=-1)
        predicted_entity_pos = []
        for b in range(logits.size(0)):
            predicted_labels = torch.argmax(logits[b], dim=-1)
            b_predicted_entity_pos = []
            for predicted_label, span in zip(
                predicted_labels.tolist(), span_idx[b].tolist()
            ):
                if predicted_label == 1:
                    b_predicted_entity_pos.append(span)
            b_predicted_entity_pos.sort(key=lambda x: [x[0], x[1]])
            predicted_entity_pos.append(b_predicted_entity_pos)

        return predicted_entity_pos

    def get_entity_embeddings(self, embeddings, attention, entity_clusters):
        offset = 1  # if self.config.transformer_type in ["bert", "roberta"] else 0
        bs, h, _, c = attention.size()
        B, L, D = embeddings.size()
        entity_embeddings = []
        entity_attention = []
        for i in range(B):
            b_entity_embs, b_entity_atts = [], []
            for entity_num, e in enumerate(entity_clusters[i]):
                entity_embs, entity_atts = [], []
                for m in e:
                    start, end = m
                    e_emb = embeddings[i, start : end + 1]
                    e_att = attention[i, :, start + offset : end + offset + 1]
                    entity_embs.append(torch.logsumexp(e_emb, dim=0))
                    entity_atts.append(e_att)
                entity_embs = torch.stack(entity_embs, dim=0)
                entity_embs = torch.logsumexp(entity_embs, dim=0)
                # entity_attention = torch.stack(entity_attention, dim=0)
                b_entity_embs.append(entity_embs)
            b_entity_embs = torch.stack(b_entity_embs, dim=0)
            entity_embeddings.append(b_entity_embs)
        entity_embeddings = torch.cat(entity_embeddings, dim=0)
        return entity_embeddings

    def get_coreference_clusters(self, hts, entity_pos, coreference_labels):
        coreferences_clusters = []
        for i in range(len(hts)):
            coreferences = []
            for j in range(len(hts[i])):
                if coreference_labels[i][j].tolist() == [0, 1]:
                    ent1_index, ent2_index = hts[i][j]
                    ent1 = entity_pos[i][ent1_index]
                    ent2 = entity_pos[i][ent2_index]
                    # Check if ent1 or ent2 is already in a cluster
                    ent1_cluster = None
                    ent2_cluster = None
                    for cluster in coreferences:
                        if ent1 in cluster:
                            ent1_cluster = cluster
                        if ent2 in cluster:
                            ent2_cluster = cluster
                    # Merge clusters or create a new one
                    if ent1_cluster is not None and ent2_cluster is not None:
                        # Merge clusters
                        if ent1_cluster != ent2_cluster:
                            ent1_cluster.extend(ent2_cluster)
                            coreferences.remove(ent2_cluster)
                    elif ent1_cluster is not None:
                        # Add ent2 to ent1_cluster
                        ent1_cluster.append(ent2)
                    elif ent2_cluster is not None:
                        # Add ent1 to ent2_cluster
                        ent2_cluster.append(ent1)
                    else:
                        # Create a new cluster
                        coreferences.append([ent1, ent2])
            for remaining_entity in entity_pos[i]:
                in_cluster = False
                for cluster in coreferences:
                    if remaining_entity in cluster:
                        in_cluster = True
                        break
                if not in_cluster:
                    coreferences.append([remaining_entity])
            coreferences_clusters.append(coreferences)
        return coreferences_clusters

    def filter_spans(self, span_representations, span_scores):
        filtered_spans = []
        for i in range(span_scores.size(0)):
            # Get the indices of spans with confidence above the threshold
            above_threshold = span_scores[i] > self.threshold
            filtered_spans.append(span_representations[i, above_threshold])
        return filtered_spans
