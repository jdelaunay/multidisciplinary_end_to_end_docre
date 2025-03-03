import torch
from torch import nn
import numpy as np

from model.layers.encoder import Encoder
from model.mention_detection import MentionDetector
from model.coreference_resolution import CoreferenceResolver
from model.entity_typing import EntityClassifier
from model.relation_classification import UNet_Relation_Extractor

from model.losses import AutomaticWeightedLoss


class DocJEREModel(nn.Module):
    """A neural network model for Document-level Joint Entity and Relation Extraction (DocJERE).
    This model performs multiple tasks including Mention Detection, Coreference Resolution,
    Entity Typing, and Relation Classification. It uses a transformer-based encoder and
    various task-specific modules to achieve these tasks.
    Attributes:
        hidden_size (int): The size of the hidden layers.
        tokenizer (Tokenizer): The tokenizer used for encoding input text.
        encoder (Encoder): The encoder module for generating embeddings.
        word_mlp (nn.Linear): A linear layer for word-level feature transformation.
        max_span_width (int): The maximum width of spans for mention detection.
        max_re_width (int): The maximum height for relation extraction.
        mention_detection (MentionDetector): The module for mention detection.
        threshold (torch.nn.Parameter): A threshold parameter for mention detection.
        coreference_resolution (CoreferenceResolver): The module for coreference resolution.
        entity_typing (EntityClassifier): The module for entity typing.
        rel_classifier (UNet_Relation_Extractor): The module for relation classification.
    Methods:
        get_best_spans(span_scores, span_indices):
            Get the best spans based on span scores.
        decode_spans(input_ids, best_spans):
            Decode the best spans into text.
        forward(input_ids, attention_mask, span_idx, span_mask, span_labels, coreference_labels,
                entity_clusters, hts, entity_pos, entity_types, entity_centric_hts, relation_labels,
                eval_mode=False, error_analysis=False, coeff_md=1, coeff_coref=1, coeff_et=1, coeff_re=1):
            Forward pass of the model.
        preprocess_joint(gt_entity_pos, predicted_entity_pos, hts, coreference_labels):
            Preprocess joint entities for a given batch.
        get_coreference_clusters(hts, entity_pos, coreference_labels):
        coreference_resolution_joint(x, attention_mask, joint_entity_pos, joint_hts, joint_coreference_labels,
                                     joint_entity_clusters, mean_proportions):
        get_batch_entity_embeddings(b_embeddings, b_attention, b_entity_clusters):
        get_entity_embeddings(embeddings, attention, entity_clusters):
    """

    def __init__(
        self,
        model,
        tokenizer,
        n_entity_classes,
        n_relation_classes,
        max_span_width=5,
        hidden_size=768,
        block_size=16,
        max_re_height=42,
        depthwise=True,
    ):
        super(DocJEREModel, self).__init__()
        self.hidden_size = hidden_size
        self.tokenizer = tokenizer
        self.encoder = Encoder(model, tokenizer)
        self.word_mlp = nn.Linear(model.config.hidden_size, hidden_size)
        self.max_span_width = max_span_width
        self.max_re_width = max_re_height
        self.awl = AutomaticWeightedLoss(4)

        # Mention Detection
        self.mention_detection = MentionDetector(
            hidden_size=hidden_size, max_width=max_span_width
        )

        self.threshold = torch.nn.Parameter(torch.tensor(0.5))

        # Coreference Resolution
        self.coreference_resolution = CoreferenceResolver(hidden_size=hidden_size)

        # Entity Typing
        self.entity_typing = EntityClassifier(
            hidden_size=hidden_size, num_labels=n_entity_classes
        )

        # Relation Classification
        self.rel_classifier = UNet_Relation_Extractor(
            hidden_size=hidden_size,
            block_size=block_size,
            num_labels=n_relation_classes,
            max_height=max_re_height,
            depthwise=depthwise,
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
        entity_clusters,
        hts,
        entity_pos,
        entity_types,
        entity_centric_hts,
        relation_labels,
        eval_mode=False,
        error_analysis=False,
        coeff_md=1,
        coeff_coref=1,
        coeff_et=1,
        coeff_re=1,
    ):
        embeddings, attention_mask = self.encoder(input_ids, attention_mask)
        B, L, D = embeddings.size()

        # Mention Detection
        if coeff_md > 0:
            (
                predicted_span_labels,
                predicted_entity_pos,
                md_loss,
                md_precision,
                md_recall,
                md_f1,
            ) = self.mention_detection(
                embeddings, span_idx, span_mask, span_labels, entity_pos
            )
        else:
            predicted_entity_pos = []
            predicted_span_labels = []
            md_loss = torch.tensor(0)
            md_precision = md_recall = md_f1 = 0
        # Coreference Resolution
        if coeff_coref > 0:
            (
                subtask_predicted_entity_clusters,
                coref_loss,
                coref_precision,
                coref_recall,
                coref_f1,
                b3_precision,
                b3_recall,
                b3_f1,
                pair_precision,
                pair_recall,
                pair_f1,
            ) = self.coreference_resolution(
                embeddings,
                attention_mask,
                entity_pos,
                hts,
                coreference_labels,
                entity_clusters,
            )
        else:
            subtask_predicted_entity_clusters = []
            coref_loss = torch.tensor(0)
            coref_precision = coref_recall = coref_f1 = 0
            b3_precision = b3_recall = b3_f1 = 0
            pair_precision = pair_recall = pair_f1 = 0

        # Entity Typing

        entity_embeddings, entity_attentions = self.get_entity_embeddings(
            embeddings, attention_mask, entity_clusters
        )
        if coeff_et > 0:
            (
                subtask_predicted_entity_types,
                entity_typing_loss,
                entity_typing_precision,
                entity_typing_recall,
                entity_typing_f1,
            ) = self.entity_typing(entity_embeddings, entity_types)
        else:
            subtask_predicted_entity_types = []
            entity_typing_loss = torch.tensor(0)
            entity_typing_precision = entity_typing_recall = entity_typing_f1 = 0

        # Relation Classification
        if coeff_re > 0:
            (
                subtask_predicted_relation_labels,
                re_loss,
                re_precision,
                re_recall,
                re_f1,
            ) = self.rel_classifier(
                embeddings,
                entity_embeddings,
                entity_attentions,
                entity_centric_hts,
                relation_labels,
            )
        else:
            subtask_predicted_relation_labels = []
            re_loss = torch.tensor(0)
            re_precision = re_recall = re_f1 = 0

        # Joint
        if eval_mode and coeff_md > 0 and coeff_coref > 0:
            # Preprocess joint
            (
                joint_coref_entity_pos,
                joint_coref_hts,
                joint_coref_labels,
            ) = self.preprocess_joint(
                entity_pos,
                predicted_entity_pos,
                hts,
                coreference_labels,
            )
            perform_coref = False
            for i in range(B):
                if len(joint_coref_entity_pos[i]) > 1:
                    perform_coref = True
                    break

            if perform_coref:
                indexes = [i for i in range(B) if len(joint_coref_entity_pos[i]) > 1]
                joint_coref_embeddings = torch.stack(
                    [embeddings[i] for i in indexes], dim=0
                )
                joint_coref_attention = torch.stack(
                    [attention_mask[i] for i in indexes], dim=0
                )
                joint_gt_entity_type = [entity_types[i] for i in indexes]
                joint_gt_entity_centric_hts = [entity_centric_hts[i] for i in indexes]
                joint_gt_relation_labels = [relation_labels[i] for i in indexes]

                joint_coref_entity_pos = [joint_coref_entity_pos[i] for i in indexes]
                joint_coref_hts = [joint_coref_hts[i] for i in indexes]
                joint_coref_labels = [joint_coref_labels[i] for i in indexes]

                joint_coref_entity_clusters = [entity_clusters[i] for i in indexes]
                joint_coref_mean_proportions = sum(
                    [
                        len(b_joint_coref_entity_clusters)
                        for b_joint_coref_entity_clusters in joint_coref_entity_clusters
                    ]
                ) / sum(
                    [len(b_entity_clusters) for b_entity_clusters in entity_clusters]
                )

                (
                    predicted_entity_clusters,
                    joint_coref_precision,
                    joint_coref_recall,
                    joint_coref_f1,
                    joint_b3_precision,
                    joint_b3_recall,
                    joint_b3_f1,
                    joint_pair_precision,
                    joint_pair_recall,
                    joint_pair_f1,
                ) = self.coreference_resolution_joint(
                    joint_coref_embeddings,
                    joint_coref_attention,
                    joint_coref_entity_pos,
                    joint_coref_hts,
                    joint_coref_labels,
                    joint_coref_entity_clusters,
                    joint_coref_mean_proportions,
                )

                # Prepare Joint Entity Typing & Joint RE
                clusters = []
                for i in range(len(predicted_entity_clusters)):
                    clusters.append([])
                    for cluster in predicted_entity_clusters[i]:
                        if cluster in joint_coref_entity_clusters[i]:
                            clusters[i].append(cluster)
                    clusters[i] = sorted(clusters[i])
                joint_et_entity_clusters = clusters

                perform_joint_et = False
                if coeff_et > 0:
                    for clusters in joint_et_entity_clusters:
                        if clusters:
                            perform_joint_et = True
                            break
                if perform_joint_et:

                    joint_et_indexes = [
                        i
                        for i in range(len(indexes))
                        if len(joint_et_entity_clusters[i]) > 0
                    ]
                    joint_et_entity_clusters = [
                        joint_et_entity_clusters[i] for i in joint_et_indexes
                    ]
                    joint_gt_entity_type = [
                        joint_gt_entity_type[i] for i in joint_et_indexes
                    ]
                    joint_gt_entity_clusters = [
                        joint_coref_entity_clusters[i] for i in joint_et_indexes
                    ]
                    joint_gt_entity_centric_hts = [
                        joint_gt_entity_centric_hts[i] for i in joint_et_indexes
                    ]
                    joint_gt_relation_labels = [
                        joint_gt_relation_labels[i] for i in joint_et_indexes
                    ]

                    joint_et_embeddings = torch.stack(
                        [joint_coref_embeddings[i] for i in joint_et_indexes], dim=0
                    )
                    joint_et_attention = torch.stack(
                        [joint_coref_attention[i] for i in joint_et_indexes], dim=0
                    )

                    joint_et_entity_types = self.get_joint_entity_types(
                        joint_et_entity_clusters,
                        joint_gt_entity_clusters,
                        joint_gt_entity_type,
                    )
                    proportions_et = sum(
                        [
                            len(b_joint_et_entity_types)
                            for b_joint_et_entity_types in joint_et_entity_types
                        ]
                    ) / sum([len(b_entity_types) for b_entity_types in entity_types])

                    joint_et_entity_embeddings, joint_et_entity_attentions = (
                        self.get_entity_embeddings(
                            joint_et_embeddings,
                            joint_et_attention,
                            joint_et_entity_clusters,
                        )
                    )

                    (
                        _,
                        joint_entity_typing_precision,
                        joint_entity_typing_recall,
                        joint_entity_typing_f1,
                    ) = self.entity_typing_joint(
                        joint_et_entity_embeddings,
                        joint_et_entity_types,
                        proportions_et,
                    )

                    joint_entity_centric_hts, joint_relation_labels = (
                        self.get_joint_relation_labels(
                            joint_et_entity_clusters,
                            joint_gt_entity_clusters,
                            joint_gt_entity_centric_hts,
                            joint_gt_relation_labels,
                        )
                    )

                    perform_joint_re = False
                    if coeff_re > 0:
                        for i in range(len(joint_entity_centric_hts)):
                            if joint_entity_centric_hts[i] and (
                                len(joint_et_entity_clusters[i]) > 1
                            ):
                                perform_joint_re = True
                                break

                    if perform_joint_re:
                        re_indexes = [
                            i
                            for i in range(len(joint_et_indexes))
                            if joint_entity_centric_hts[i]
                        ]

                        joint_re_embeddings = torch.stack(
                            [joint_et_embeddings[i] for i in re_indexes], dim=0
                        )
                        joint_re_entity_embeddings = [
                            joint_et_entity_embeddings[i] for i in re_indexes
                        ]
                        joint_re_entity_attentions = [
                            joint_et_entity_attentions[i] for i in re_indexes
                        ]
                        joint_entity_centric_hts = [
                            joint_entity_centric_hts[i] for i in re_indexes
                        ]
                        joint_re_relation_labels = [
                            joint_relation_labels[i] for i in re_indexes
                        ]
                        joint_gt_entity_centric_hts = [
                            joint_gt_entity_centric_hts[i] for i in re_indexes
                        ]

                        proportions_re = sum(
                            [
                                len(b_joint_entity_centric_hts)
                                for b_joint_entity_centric_hts in joint_entity_centric_hts
                            ]
                        ) / sum(
                            [
                                len(b_entity_centric_hts)
                                for b_entity_centric_hts in entity_centric_hts
                            ]
                        )

                        (
                            joint_re_precision,
                            joint_re_recall,
                            joint_re_f1,
                        ) = self.relation_extraction_joint(
                            joint_re_embeddings,
                            joint_re_entity_embeddings,
                            joint_re_entity_attentions,
                            joint_entity_centric_hts,
                            joint_re_relation_labels,
                            proportions_re,
                        )
                    else:
                        joint_re_precision = joint_re_recall = joint_re_f1 = 0

                else:
                    joint_entity_typing_precision = joint_entity_typing_recall = (
                        joint_entity_typing_f1
                    ) = 0
                    joint_re_precision = joint_re_recall = joint_re_f1 = 0
            else:
                joint_coref_precision = joint_coref_recall = joint_coref_f1 = 0
                joint_b3_precision = joint_b3_recall = joint_b3_f1 = 0
                joint_pair_precision = joint_pair_recall = joint_pair_f1 = 0
                joint_entity_typing_precision = joint_entity_typing_recall = (
                    joint_entity_typing_f1
                ) = 0
                joint_re_precision = joint_re_recall = joint_re_f1 = 0
        else:
            joint_coref_precision = joint_coref_recall = joint_coref_f1 = 0
            joint_b3_precision = joint_b3_recall = joint_b3_f1 = 0
            joint_pair_precision = joint_pair_recall = joint_pair_f1 = 0
            joint_entity_typing_precision = joint_entity_typing_recall = (
                joint_entity_typing_f1
            ) = 0
            joint_re_precision = joint_re_recall = joint_re_f1 = 0

        epsilon = 1e-6
        # total_loss = (
        #     coeff_md * md_loss
        #     + coeff_coref * coref_loss
        #     + coeff_et * entity_typing_loss
        #     + coeff_re * re_loss
        # )
        total_loss = self.awl(
            coeff_md * md_loss,
            coeff_coref * coref_loss,
            coeff_et * entity_typing_loss,
            coeff_re * re_loss,
        )

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
            "b3_precision": b3_precision,
            "b3_recall": b3_recall,
            "b3_f1": b3_f1,
            "pair_precision": pair_precision,
            "pair_recall": pair_recall,
            "pair_f1": pair_f1,
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
                "precision": joint_coref_precision,
                "recall": joint_coref_recall,
                "f1": joint_coref_f1,
                "b3_precision": joint_b3_precision,
                "b3_recall": joint_b3_recall,
                "b3_f1": joint_b3_f1,
                "pair_precision": joint_pair_precision,
                "pair_recall": joint_pair_recall,
                "pair_f1": joint_pair_f1,
            }

            joint_entity_typing_outputs = {
                "precision": joint_entity_typing_precision,
                "recall": joint_entity_typing_recall,
                "f1": joint_entity_typing_f1,
            }
            joint_re_outputs = {
                "precision": joint_re_precision,
                "recall": joint_re_recall,
                "f1": joint_re_f1,
            }
            if error_analysis:
                md_outputs["predictions"] = predicted_span_labels
                coref_outputs["predictions"] = subtask_predicted_entity_clusters
                entity_typing_outputs["predictions"] = subtask_predicted_entity_types
                re_outputs["predictions"] = subtask_predicted_relation_labels
            return (
                total_loss,
                md_outputs,
                coref_outputs,
                joint_coref_outputs,
                entity_typing_outputs,
                joint_entity_typing_outputs,
                re_outputs,
                joint_re_outputs,
            )
        else:
            return (
                total_loss,
                md_outputs,
                coref_outputs,
                entity_typing_outputs,
                re_outputs,
            )

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
        joint_entity_pos, joint_hts, joint_coreference_labels = [], [], []

        for b in range(B):
            predicted_entity_pos[b] = sorted(
                [(entity[0], entity[1]) for entity in predicted_entity_pos[b]]
            )

            b_joint_entity_pos, b_joint_hts, b_joint_coreference_labels = (
                [],
                [],
                [],
            )

            b_joint_entity_pos = [
                entity
                for entity in predicted_entity_pos[b]
                if entity in gt_entity_pos[b]
            ]
            for i in range(len(b_joint_entity_pos) - 1):
                for j in range(i + 1, len(b_joint_entity_pos)):
                    h = b_joint_entity_pos[i]
                    t = b_joint_entity_pos[j]
                    index_h = gt_entity_pos[b].index(h)
                    index_t = gt_entity_pos[b].index(t)
                    hts_index = hts[b].index((index_h, index_t))
                    b_joint_hts.append((i, j))
                    b_joint_coreference_labels.append(
                        coreference_labels[b][hts_index].tolist()
                    )

            joint_entity_pos.append(b_joint_entity_pos)

            joint_hts.append(b_joint_hts)
            joint_coreference_labels.append(b_joint_coreference_labels)
        joint_coreference_labels = [
            torch.tensor(label) for label in joint_coreference_labels
        ]
        return (
            joint_entity_pos,
            joint_hts,
            joint_coreference_labels,
        )

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
            coreferences_clusters.append(sorted(coreferences))
        return coreferences_clusters

    def coreference_resolution_joint(
        self,
        x,
        attention_mask,
        joint_entity_pos,
        joint_hts,
        joint_coreference_labels,
        joint_entity_clusters,
        mean_proportions,
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
        (
            predicted_entity_clusters,
            _,
            coref_precision,
            coref_recall,
            coref_f1,
            b3_precision,
            b3_recall,
            b3_f1,
            pair_precision,
            pair_recall,
            pair_f1,
        ) = self.coreference_resolution(
            x,
            attention_mask,
            joint_entity_pos,
            joint_hts,
            joint_coreference_labels,
            joint_entity_clusters,
        )
        joint_coref_precision = coref_precision * mean_proportions
        joint_coref_recall = coref_recall * mean_proportions
        joint_coref_f1 = coref_f1 * mean_proportions
        joint_b3_precision = b3_precision * mean_proportions
        joint_b3_recall = b3_recall * mean_proportions
        joint_b3_f1 = b3_f1 * mean_proportions
        joint_pair_precision = pair_precision * mean_proportions
        joint_pair_recall = pair_recall * mean_proportions
        joint_pair_f1 = pair_f1 * mean_proportions
        return (
            predicted_entity_clusters,
            joint_coref_precision,
            joint_coref_recall,
            joint_coref_f1,
            joint_b3_precision,
            joint_b3_recall,
            joint_b3_f1,
            joint_pair_precision,
            joint_pair_recall,
            joint_pair_f1,
        )

    def get_batch_entity_embeddings(self, b_embeddings, b_attention, b_entity_clusters):
        """
        Calculate the embeddings and attention weights for a batch of entity clusters.

        Args:
            b_embeddings (torch.Tensor): The embeddings of the batch of sequences.
            b_attention (torch.Tensor): The attention weights of the batch of sequences.
            b_entity_clusters (List[List[Tuple[int, int]]]): The entity clusters for each sequence in the batch.

        Returns:
            torch.Tensor: The embeddings of the entity clusters.
            torch.Tensor: The attention weights of the entity clusters.
        """
        h, _, c = b_attention.size()
        b_entity_embs, b_entity_atts = [], []
        for entity_num, e in enumerate(b_entity_clusters):
            e_emb, e_att = [], []
            for m in e:
                (start, end) = m
                if (start < c) and (end < c):
                    # In case the entity mention is truncated due to limited max seq length.
                    m_emb = b_embeddings[start:end].mean(0)
                    m_att = b_attention[:, start]
                else:
                    m_emb = torch.zeros(self.hidden_size).to(b_embeddings)
                    m_att = torch.zeros(h, c).to(b_attention)
                e_emb.append(m_emb)
                e_att.append(m_att)
            if len(e_emb) > 0:
                e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                e_att = torch.stack(e_att, dim=0).mean(0)

            b_entity_embs.append(e_emb)
            b_entity_atts.append(e_att)
        b_entity_embs = torch.stack(b_entity_embs, dim=0)
        for _ in range(self.max_re_width - entity_num - 1):
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
        B, L, D = embeddings.size()
        entity_embeddings = []
        entity_attention = []
        for i in range(B):
            b_entity_embs, b_entity_att = self.get_batch_entity_embeddings(
                embeddings[i], attention[i], entity_clusters[i]
            )
            entity_embeddings.append(b_entity_embs)
            entity_attention.append(b_entity_att)
        return entity_embeddings, entity_attention

    def get_joint_entity_types(
        self, joint_entity_clusters, entity_clusters, entity_types
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
        joint_entity_types = [[] for _ in range(len(joint_entity_clusters))]
        for i in range(len(joint_entity_clusters)):
            for cluster in joint_entity_clusters[i]:
                cluster_index = entity_clusters[i].index(cluster)
                joint_entity_types[i].append(entity_types[i][cluster_index].tolist())
        return [torch.tensor(types) for types in joint_entity_types]

    def get_joint_relation_labels(
        self,
        joint_entity_clusters,
        entity_clusters,
        entity_centric_hts,
        relation_labels,
    ):
        joint_entity_centric_hts, joint_relation_labels = [
            [] for _ in range(len(joint_entity_clusters))
        ], [[] for _ in range(len(joint_entity_clusters))]

        for i in range(len(joint_entity_clusters)):
            for a, h in enumerate(joint_entity_clusters[i]):
                for b, t in enumerate(joint_entity_clusters[i]):
                    if (a != b) and (h != t):
                        joint_entity_centric_hts[i].append((a, b))
                        h_index = entity_clusters[i].index(h)
                        t_index = entity_clusters[i].index(t)
                        joint_relation_labels[i].append(
                            relation_labels[i][
                                entity_centric_hts[i].index((h_index, t_index))
                            ]
                        )

        return joint_entity_centric_hts, joint_relation_labels

    def entity_typing_joint(
        self,
        joint_entity_embeddings,
        joint_entity_types,
        proportion,
    ):
        """
        Calculate the joint precision, recall, and F1-score for entity typing.

        Args:
            joint_entity_embeddings (list): List of joint entity embeddings.
            joint_entity_types (list): List of joint entity types.
            proportions (list): List of proportions.

        Returns:
            tuple: A tuple containing the joint precision, recall, and F1-score.
        """
        predicted_types, _, precision, recall, f1 = self.entity_typing(
            joint_entity_embeddings, joint_entity_types
        )
        joint_precision = precision * proportion
        joint_recall = recall * proportion
        joint_f1 = f1 * proportion
        return (predicted_types, joint_precision, joint_recall, joint_f1)

    def relation_extraction_joint(
        self,
        joint_embeddings,
        joint_entity_embeddings,
        joint_entity_attentions,
        joint_entity_centric_hts,
        joint_relation_labels,
        proportion,
    ):
        _, _, precision, recall, f1 = self.rel_classifier(
            joint_embeddings,
            joint_entity_embeddings,
            joint_entity_attentions,
            joint_entity_centric_hts,
            joint_relation_labels,
        )
        joint_precision = precision * proportion
        joint_recall = recall * proportion
        joint_f1 = f1 * proportion
        return (joint_precision, joint_recall, joint_f1)

    def filter_spans(self, span_representations, span_scores):
        filtered_spans = []
        for i in range(span_scores.size(0)):
            # Get the indices of spans with confidence above the threshold
            above_threshold = span_scores[i] > self.threshold
            filtered_spans.append(span_representations[i, above_threshold])
        return filtered_spans
