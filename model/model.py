import torch
from torch import nn

from model.layers.encoder import Encoder
from model.mention_detection import MentionDetector
from model.coreference_resolution import CoreferenceResolver
from model.entity_typing import EntityClassifier
from model.relation_classification import UNet_Relation_Extractor


class DocJEREModel(nn.Module):
    def __init__(
        self,
        model,
        tokenizer,
        n_entity_classes,
        n_relation_classes,
        hidden_size=768,
        block_size=64,
        max_height=42,
    ):
        super(DocJEREModel, self).__init__()
        self.hidden_size = hidden_size
        self.tokenizer = tokenizer
        self.encoder = Encoder(model, tokenizer)
        self.word_mlp = nn.Linear(model.config.hidden_size, hidden_size)
        self.max_height = max_height

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

        # Relation Classification
        self.rel_classifier = UNet_Relation_Extractor(
            hidden_size=hidden_size,
            block_size=block_size,
            num_labels=n_relation_classes,
            max_height=max_height,
        )

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
        entity_centric_hts,
        relation_labels,
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
                embeddings,
                attention_mask,
                entity_pos,
                hts,
                coreference_labels,
            )
        )

        # Entity Typing
        entity_clusters = self.get_coreference_clusters(
            hts, entity_pos, coreference_labels
        )
        print("ENTITY CLUSTERS:", entity_clusters)

        entity_embeddings, entity_attentions = self.get_entity_embeddings(
            embeddings, attention_mask, entity_clusters
        )
        print("ENTITY EMB:", entity_embeddings)
        print("I ENTITY EMB:", entity_embeddings[0])
        print("ENTITY TYPES:", entity_types)
        print("I ENTITY TYPES:", entity_types[0])
        (
            entity_typing_loss,
            entity_typing_precision,
            entity_typing_recall,
            entity_typing_f1,
        ) = self.entity_typing(entity_embeddings, entity_types)

        # TODO: Relation Classification
        re_loss, re_precision, re_recall, re_f1 = self.rel_classifier(
            embeddings,
            entity_embeddings,
            entity_attentions,
            entity_centric_hts,
            relation_labels,
        )

        # Joint
        if eval_mode:
            # Preprocess joint
            predicted_entity_pos = self.get_predicted_entity_pos(
                logits.view(span_idx.size(0), span_idx.size(1), -1), span_idx
            )
            (
                joint_entity_pos,
                joint_hts,
                joint_coreference_labels,
                proportions,
            ) = self.preprocess_joint(
                entity_pos,
                predicted_entity_pos,
                hts,
                coreference_labels,
            )
            # Joint Coreference Resolution

            (
                joint_coref_precision,
                joint_coref_recall,
                joint_coref_f1,
            ) = self.coreference_resolution_joint(
                embeddings,
                attention_mask,
                joint_entity_pos,
                joint_hts,
                joint_coreference_labels,
                proportions,
            )

            # TODO: Joint Entity Typing
            (
                proportions_joint_entity_clusters,
                aligned_mentions_proportions,
                joint_entity_types,
                joint_entity_embeddings,
            ) = (
                [],
                [],
                [],
                [],
            )
            is_joint_entity = False
            for i in range(B):
                if proportions[i] > 0 and len(joint_entity_pos[i]) > 1:
                    i_joint_hts = [joint_hts[i]]
                    i_joint_entity_pos = [joint_entity_pos[i]]
                    i_joint_coreference_labels = [joint_coreference_labels[i]]
                    i_joint_entity_clusters = self.get_coreference_clusters(
                        i_joint_hts,
                        i_joint_entity_pos,
                        i_joint_coreference_labels,
                    )
                    i_joint_entity_clusters = [
                        cluster
                        for cluster in i_joint_entity_clusters[0]
                        if cluster in entity_clusters[i]
                    ]
                    if i_joint_entity_clusters:
                        is_joint_entity = True
                        aligned_mentions_proportions.append(proportions[i])
                        i_joint_entity_types = self.get_joint_entity_types_in_batch(
                            i_joint_entity_clusters, entity_clusters[i], entity_types[i]
                        )
                        proportions_joint_entity_clusters.append(
                            len(i_joint_entity_clusters) / len(entity_clusters[i])
                        )
                        i_joint_entity_embeddings, _ = self.get_batch_entity_embeddings(
                            embeddings[i], attention_mask[i], i_joint_entity_clusters
                        )
                        joint_entity_embeddings.append(i_joint_entity_embeddings)
                        joint_entity_types.append(torch.tensor(i_joint_entity_types))

            if is_joint_entity:
                print("-----------------------------------")
                print(
                    "Proportion of joint entities:", proportions_joint_entity_clusters
                )
                (
                    joint_entity_typing_precision,
                    joint_entity_typing_recall,
                    joint_entity_typing_f1,
                ) = self.entity_typing_joint(
                    joint_entity_embeddings,
                    joint_entity_types,
                    proportions_joint_entity_clusters,
                    aligned_mentions_proportions,
                )

            else:
                joint_entity_typing_precision = joint_entity_typing_recall = (
                    joint_entity_typing_f1
                ) = 0
        print("========================================")
        total_loss = md_loss + coref_loss + entity_typing_loss + re_loss
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
        re_outputs = {
            "loss": re_loss,
            "precision": re_precision,
            "recall": re_recall,
            "f1": re_f1,
        }

        if eval_mode:
            joint_coref_outputs = {
                "precision": joint_coref_precision / B,
                "recall": joint_coref_recall / B,
                "f1": joint_coref_f1 / B,
            }
            joint_entity_typing_outputs = {
                "precision": joint_entity_typing_precision / B,
                "recall": joint_entity_typing_recall / B,
                "f1": joint_entity_typing_f1 / B,
            }
            return (
                total_loss,
                md_outputs,
                coref_outputs,
                joint_coref_outputs,
                entity_typing_outputs,
                joint_entity_typing_outputs,
                re_outputs,
            )
        else:
            return (
                total_loss,
                md_outputs,
                coref_outputs,
                entity_typing_outputs,
                re_outputs,
            )

    def get_predicted_entity_pos(self, logits, span_idx):
        """
        Get the predicted entity positions based on the given logits and span indices.

        Args:
            logits (torch.Tensor): The logits tensor.
            span_idx (torch.Tensor): The tensor containing the span indices.

        Returns:
            List[List[Tuple[int, int]]]: A list of lists of tuples representing the predicted entity positions.
        """
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

    def preprocess_joint(
        self,
        gt_entity_pos,
        predicted_entity_pos,
        hts,
        coreference_labels,
    ):
        """
        Preprocesses the joint entities for a given batch.

        Args:
            gt_entity_pos (list): The ground truth entity positions for each sample in the batch.
            predicted_entity_pos (list): The predicted entity positions for each sample in the batch.
            hts (list): The hypergraph triplets for each sample in the batch.
            coreference_labels (list): The coreference labels for each hypergraph triplet in the batch.

        Returns:
            tuple: A tuple containing the preprocessed joint entity positions, hypergraph triplets,
            coreference labels, and the proportions of joint entities for each sample in the batch.
        """
        B = len(gt_entity_pos)
        joint_ent_proportions = []
        joint_entity_pos, joint_hts, joint_coreference_labels = [], [], []
        for b in range(B):
            b_joint_entity_pos, b_joint_hts, b_joint_coreference_labels = (
                [],
                [],
                [],
            )
            b_joint_entity_pos = list(
                set(
                    [
                        (entity[0], entity[1])
                        for entity in predicted_entity_pos[b]
                        if (entity[0], entity[1]) in gt_entity_pos[b]
                    ]
                )
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

    def coreference_resolution_joint(
        self,
        x,
        attention_mask,
        joint_entity_pos,
        joint_hts,
        joint_coreference_labels,
        proportions,
    ):
        """
        Perform joint coreference resolution.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, D).
            attention_mask (torch.Tensor): Attention mask tensor of shape (B, L).
            joint_entity_pos (List[List[int]]): List of entity positions for each sample in the batch.
            joint_hts (List[List[int]]): List of head tokens for each entity in each sample.
            joint_coreference_labels (List[List[int]]): List of coreference labels for each entity in each sample.
            proportions (List[float]): List of proportions for each sample in the batch.

        Returns:
            Tuple[float, float, float]: Tuple containing the joint coreference precision, recall, and F1 score.
        """
        B, L, D = x.size()
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
        return (joint_coref_precision, joint_coref_recall, joint_coref_f1)

    def get_batch_entity_embeddings(
        self, b_embeddings, b_attention, b_entity_clusters, offset=1
    ):
        """
        Calculate the embeddings and attention weights for a batch of entity clusters.

        Args:
            b_embeddings (torch.Tensor): The embeddings of the batch of sequences.
            b_attention (torch.Tensor): The attention weights of the batch of sequences.
            b_entity_clusters (List[List[Tuple[int, int]]]): The entity clusters for each sequence in the batch.
            offset (int, optional): The offset value. Defaults to 1.

        Returns:
            torch.Tensor: The embeddings of the entity clusters.
            torch.Tensor: The attention weights of the entity clusters.
        """

        h, _, c = b_attention.size()
        offset = 1  # if self.config.transformer_type in ["bert", "roberta"] else 0
        b_entity_embs, b_entity_atts = [], []
        for entity_num, e in enumerate(b_entity_clusters):
            e_emb, e_att = [], []
            for m in e:
                start, end = m
                if start + 1 < c:
                    # In case the entity mention is truncated due to limited max seq length.
                    e_emb.append(torch.logsumexp(b_embeddings[start : end + 1], dim=0))
                    e_att.append(b_attention[:, start + 1])
            if len(e_emb) > 0:
                e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                e_att = torch.stack(e_att, dim=0).mean(0)
            else:
                e_emb = torch.zeros(self.hidden_size).to(b_embeddings)
                e_att = torch.zeros(h, c).to(b_attention)
            b_entity_embs.append(e_emb)
            b_entity_atts.append(e_att)
        b_entity_embs = torch.stack(b_entity_embs, dim=0)
        # entity_embs = torch.logsumexp(entity_embs, dim=0)
        for _ in range(self.max_height - entity_num - 1):
            b_entity_atts.append(e_att)
        b_entity_atts = torch.stack(b_entity_atts, dim=0)

        return b_entity_embs, b_entity_atts

    def get_entity_embeddings(self, embeddings, attention, entity_clusters):
        """
        Get pooled entity embeddings and attention scores for each entity.

        Args:
            embeddings (torch.Tensor): Input embeddings of shape (B, L, D),
                where B is the batch size, L is the sequence length, and D is the embedding dimension.
            attention (torch.Tensor): Attention scores of shape (B, L, L),
                where B is the batch size and L is the sequence length.
            entity_clusters (List[List[int]]): List of entity clusters for each batch,
                where each entity cluster is a list of indices representing the entities in the cluster.

        Returns:
            torch.Tensor: Entity embeddings of shape (N, D),
                where N is the total number of entities across all batches and D is the embedding dimension.
            List[torch.Tensor]: List of attention scores for each entity cluster,
                where each attention score tensor has shape (M, L),
                where M is the number of entities in the cluster and L is the sequence length.
        """
        offset = 1  # if self.config.transformer_type in ["bert", "roberta"] else 0
        B, L, D = embeddings.size()
        entity_embeddings = []
        entity_attention = []
        for i in range(B):
            b_embeddings = embeddings[i]
            b_attention = attention[i]
            b_entity_embs, b_entity_att = self.get_batch_entity_embeddings(
                b_embeddings, b_attention, entity_clusters[i], offset=offset
            )
            entity_embeddings.append(b_entity_embs)
            entity_attention.append(b_entity_att)
        entity_embeddings = torch.cat(entity_embeddings, dim=0)
        return entity_embeddings, entity_attention

    def get_coreference_clusters(self, hts, entity_pos, coreference_labels):
        """
        Get the coreference clusters from the given inputs.

        Args:
            hts (list): List of lists containing the indices of the entities involved in each coreference pair.
            entity_pos (list): List of lists containing the positions of the entities in the input.
            coreference_labels (list): List of lists containing the labels indicating whether a coreference pair exists.

        Returns:
            list: List of coreference clusters, where each cluster is a list of entities.

        """
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

    def get_joint_entity_types_in_batch(
        self, i_joint_entity_clusters, i_entity_clusters, i_entity_types
    ):
        """
        Get the joint entity types in a batch.

        Args:
            i_joint_entity_clusters (list): List of joint entity clusters.
            i_entity_clusters (list): List of entity clusters.
            i_entity_types (list): List of entity types.

        Returns:
            list: List of joint entity types.

        """
        i_joint_entity_types = []
        for cluster in i_joint_entity_clusters:
            if cluster in i_entity_clusters:
                cluster_index = i_entity_clusters.index(cluster)
                i_joint_entity_types.append(i_entity_types[cluster_index].tolist())
        return i_joint_entity_types

    def entity_typing_joint(
        self,
        joint_entity_embeddings,
        joint_entity_types,
        proportions,
        aligned_mentions_proportions,
    ):
        """
        Calculate the joint precision, recall, and F1-score for entity typing.

        Args:
            joint_entity_embeddings (list): List of joint entity embeddings.
            joint_entity_types (list): List of joint entity types.
            proportions (list): List of proportions.
            aligned_mentions_proportions (list): List of aligned mentions proportions.

        Returns:
            tuple: A tuple containing the joint precision, recall, and F1-score.
        """
        B = len(proportions)
        joint_precision = joint_recall = joint_f1 = 0
        print("LEN JOINT ENTITY EMBEDDINGS:", len(joint_entity_embeddings))
        for i in range(B):
            if proportions[i] > 0.1 and len(joint_entity_embeddings[i]) > 1:
                i_labels = joint_entity_types[i]
                _, precision, recall, f1 = self.entity_typing(
                    joint_entity_embeddings[i], [i_labels]
                )
                joint_precision += (
                    precision * proportions[i]  # * aligned_mentions_proportions[i]
                )
                joint_recall += (
                    recall * proportions[i]  # * aligned_mentions_proportions[i]
                )
                joint_f1 += f1 * proportions[i]  # * aligned_mentions_proportions[i]
        return (joint_precision, joint_recall, joint_f1)

    def filter_spans(self, span_representations, span_scores):
        filtered_spans = []
        for i in range(span_scores.size(0)):
            # Get the indices of spans with confidence above the threshold
            above_threshold = span_scores[i] > self.threshold
            filtered_spans.append(span_representations[i, above_threshold])
        return filtered_spans
