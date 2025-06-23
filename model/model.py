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
                eval_mode=False, error_analysis=False, coeff_md=1, coeff_cr=1, coeff_et=1, coeff_re=1):
            Forward pass of the model.
        preprocess_joint(gt_entity_pos, predicted_entity_pos, hts, coreference_labels):
            Preprocess joint entities for a given batch.
        get_coreference_clusters(hts, entity_pos, coreference_labels):
        coreference_resolution_joint(x, attention_mask, e2e_entity_pos, e2e_hts, e2e_coreference_labels,
                                     e2e_entity_clusters, mean_proportions):
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
        titles=None,
        eval_mode=False,
        error_analysis=False,
        coeff_md=1,
        coeff_cr=1,
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
            ) = self.mention_detection(
                embeddings, span_idx, span_mask, span_labels, entity_pos
            )
            predicted_span_labels = torch.flatten(predicted_span_labels)
        else:
            predicted_entity_pos = []
            predicted_span_labels = []
            md_loss = torch.tensor(0)

        # Coreference Resolution
        if coeff_cr > 0:
            (
                isolated_predicted_entity_clusters,
                coref_loss,
            ) = self.coreference_resolution(
                embeddings,
                attention_mask,
                entity_pos,
                hts,
                coreference_labels,
            )
        else:
            isolated_predicted_entity_clusters = []
            coref_loss = torch.tensor(0)

        # Entity Typing

        entity_embeddings, entity_attentions = self.get_entity_embeddings(
            embeddings, attention_mask, entity_clusters
        )
        if coeff_et > 0:
            (
                isolated_predicted_entity_types,
                entity_typing_loss,
            ) = self.entity_typing(entity_embeddings, entity_types)
        else:
            isolated_predicted_entity_types = torch.empty(0, dtype=torch.long)
            entity_typing_loss = torch.tensor(0)

        # Relation Classification
        if coeff_re > 0:
            (
                isolated_re_preds,
                isolated_re_scores,
                isolated_re_topks,
                re_loss,
            ) = self.rel_classifier(
                embeddings,
                entity_embeddings,
                entity_attentions,
                entity_centric_hts,
                relation_labels,
            )
        else:
            isolated_re_preds = torch.empty(0, dtype=torch.long)
            isolated_re_scores = isolated_re_topks = torch.empty(0, dtype=torch.long)
            re_loss = torch.tensor(0)

        # Joint
        if eval_mode and coeff_md > 0 and coeff_cr > 0:
            # Preprocess joint
            (
                e2e_entity_pos,
                e2e_coref_hts,
                e2e_coref_labels,
            ) = self.preprocess_e2e(
                entity_pos,
                predicted_entity_pos,
                hts,
                coreference_labels,
            )
            perform_coref = False
            for i in range(B):
                if len(e2e_entity_pos[i]) > 1:
                    perform_coref = True
                    break

            if perform_coref:
                indexes = [i for i in range(B) if len(e2e_entity_pos[i]) > 1]
                e2e_coref_embeddings = torch.stack(
                    [embeddings[i] for i in indexes], dim=0
                )
                e2e_coref_attention = torch.stack(
                    [attention_mask[i] for i in indexes], dim=0
                )
                e2e_gt_entity_type = [entity_types[i] for i in indexes]
                e2e_gt_entity_centric_hts = [entity_centric_hts[i] for i in indexes]
                e2e_gt_relation_labels = [relation_labels[i] for i in indexes]

                e2e_entity_pos = [e2e_entity_pos[i] for i in indexes]
                e2e_coref_hts = [e2e_coref_hts[i] for i in indexes]
                e2e_coref_labels = [e2e_coref_labels[i] for i in indexes]

                e2e_coref_entity_clusters = [entity_clusters[i] for i in indexes]

                e2e_predicted_entity_clusters, _ = self.coreference_resolution(
                    e2e_coref_embeddings,
                    e2e_coref_attention,
                    e2e_entity_pos,
                    e2e_coref_hts,
                    e2e_coref_labels,
                )

                # Prepare Joint Entity Typing & Joint RE
                clusters = []
                for i in range(len(e2e_predicted_entity_clusters)):
                    clusters.append([])
                    for cluster in e2e_predicted_entity_clusters[i]:
                        if cluster in e2e_coref_entity_clusters[i]:
                            clusters[i].append(cluster)
                    clusters[i] = sorted(clusters[i])
                e2e_et_entity_clusters = clusters

                perform_e2e_et = False
                if coeff_et > 0:
                    for clusters in e2e_et_entity_clusters:
                        if clusters:
                            perform_e2e_et = True
                            break
                if perform_e2e_et:
                    e2e_et_indexes = [
                        i
                        for i in range(len(indexes))
                        if len(e2e_et_entity_clusters[i]) > 0
                    ]
                    e2e_et_indexes_no_mentions = [
                        indexes[i]
                        for i in range(len(indexes))
                        if len(e2e_et_entity_clusters[i]) == 0
                    ]
                    e2e_cr_indexes_no_mentions = [
                        i for i in range(B) if i not in indexes
                    ]
                    e2e_et_indexes_no_mentions += e2e_cr_indexes_no_mentions
                    e2e_et_indexes_no_mentions = sorted(
                        list(set(e2e_et_indexes_no_mentions))
                    )

                    e2e_et_indexes_from_origin, e2e_re_indexes_from_origin = [], []
                    for i in indexes:
                        if len(e2e_et_entity_clusters[indexes.index(i)]) > 0:
                            e2e_et_indexes_from_origin.append(i)
                        if len(e2e_et_entity_clusters[indexes.index(i)]) > 1:
                            e2e_re_indexes_from_origin.append(i)

                    e2e_et_entity_clusters = [
                        e2e_et_entity_clusters[i] for i in e2e_et_indexes
                    ]
                    e2e_gt_entity_type = [e2e_gt_entity_type[i] for i in e2e_et_indexes]
                    e2e_gt_entity_clusters = [
                        e2e_coref_entity_clusters[i] for i in e2e_et_indexes
                    ]
                    e2e_gt_entity_centric_hts = [
                        e2e_gt_entity_centric_hts[i] for i in e2e_et_indexes
                    ]
                    e2e_gt_relation_labels = [
                        e2e_gt_relation_labels[i] for i in e2e_et_indexes
                    ]

                    e2e_et_embeddings = torch.stack(
                        [e2e_coref_embeddings[i] for i in e2e_et_indexes], dim=0
                    )
                    e2e_et_attention = torch.stack(
                        [e2e_coref_attention[i] for i in e2e_et_indexes], dim=0
                    )

                    e2e_et_entity_types = self.get_e2e_entity_types(
                        e2e_et_entity_clusters,
                        e2e_gt_entity_clusters,
                        e2e_gt_entity_type,
                    )

                    e2e_et_entity_embeddings, e2e_et_entity_attentions = (
                        self.get_entity_embeddings(
                            e2e_et_embeddings,
                            e2e_et_attention,
                            e2e_et_entity_clusters,
                        )
                    )

                    e2e_predicted_entity_types, _ = self.entity_typing(
                        e2e_et_entity_embeddings,
                        e2e_et_entity_types,
                    )

                    e2e_re_entity_clusters = e2e_et_entity_clusters.copy()

                    # if len(e2e_et_entity_clusters) != len(entity_clusters):
                    #     for i in e2e_et_indexes_no_mentions:
                    #         e2e_et_entity_clusters.insert(i, [])

                    (
                        e2e_entity_centric_hts,
                        e2e_relation_labels,
                        e2e_relative_entity_centric_hts,
                    ) = self.get_e2e_relation_labels(
                        e2e_re_entity_clusters,
                        e2e_gt_entity_clusters,
                        e2e_gt_entity_centric_hts,
                        e2e_gt_relation_labels,
                    )

                    perform_e2e_re = False
                    if coeff_re > 0:
                        for i in range(len(e2e_entity_centric_hts)):
                            if e2e_entity_centric_hts[i] and (
                                len(e2e_relative_entity_centric_hts[i]) > 1
                            ):
                                perform_e2e_re = True
                                break

                    if perform_e2e_re:
                        e2e_re_indexes = [
                            i
                            for i in range(len(e2e_et_indexes))
                            if e2e_entity_centric_hts[i]
                        ]
                        e2e_re_indexes_no_mentions = [
                            indexes[i]
                            for i in range(len(e2e_et_indexes))
                            if len(e2e_entity_centric_hts[i]) == 0
                        ]
                        e2e_re_indexes_no_mentions += e2e_et_indexes_no_mentions
                        e2e_re_indexes_no_mentions = sorted(
                            list(set(e2e_re_indexes_no_mentions))
                        )

                        e2e_re_embeddings = torch.stack(
                            [e2e_et_embeddings[i] for i in e2e_re_indexes], dim=0
                        )
                        e2e_re_entity_embeddings = [
                            e2e_et_entity_embeddings[i] for i in e2e_re_indexes
                        ]
                        e2e_re_entity_attentions = [
                            e2e_et_entity_attentions[i] for i in e2e_re_indexes
                        ]
                        e2e_entity_centric_hts = [
                            e2e_entity_centric_hts[i] for i in e2e_re_indexes
                        ]
                        e2e_relative_entity_centric_hts = [
                            e2e_relative_entity_centric_hts[i] for i in e2e_re_indexes
                        ]
                        e2e_relation_labels = [
                            e2e_relation_labels[i] for i in e2e_re_indexes
                        ]
                        e2e_gt_entity_centric_hts = [
                            e2e_gt_entity_centric_hts[i] for i in e2e_re_indexes
                        ]

                        (
                            e2e_re_preds,
                            e2e_re_scores,
                            e2e_re_topks,
                            _,
                        ) = self.rel_classifier(
                            e2e_re_embeddings,
                            e2e_re_entity_embeddings,
                            e2e_re_entity_attentions,
                            e2e_relative_entity_centric_hts,
                            e2e_relation_labels,
                        )
                        if len(e2e_entity_centric_hts) != len(entity_centric_hts):
                            for i in e2e_re_indexes_no_mentions:
                                e2e_entity_centric_hts.insert(i, [])
                    else:
                        e2e_re_preds = torch.empty(0, dtype=torch.long)
                        e2e_re_scores = e2e_re_topks = torch.empty(0, dtype=torch.long)

                else:
                    indexes, e2e_et_indexes_from_origin, e2e_re_indexes_from_origin = (
                        [],
                        [],
                        [],
                    )
                    e2e_et_entity_clusters = []
                    e2e_re_preds = torch.empty(0, dtype=torch.long)
                    e2e_re_scores = e2e_re_topks = torch.empty(0, dtype=torch.long)
                    e2e_predicted_entity_types = torch.empty(0, dtype=torch.long)
                    e2e_entity_centric_hts = [[] for _ in range(len(embeddings))]
            else:
                indexes, e2e_et_indexes_from_origin, e2e_re_indexes_from_origin = (
                    [],
                    [],
                    [],
                )
                e2e_predicted_entity_clusters, e2e_et_entity_clusters = [], []
                e2e_re_preds = torch.empty(0, dtype=torch.long)
                e2e_re_scores = e2e_re_topks = torch.empty(0, dtype=torch.long)
                e2e_predicted_entity_types = torch.empty(0, dtype=torch.long)
                e2e_entity_centric_hts = [[] for _ in range(len(embeddings))]
        else:
            indexes, e2e_et_indexes_from_origin, e2e_re_indexes_from_origin = [], [], []
            e2e_predicted_entity_clusters, e2e_et_entity_clusters = [], []
            e2e_re_preds = torch.empty(0, dtype=torch.long)
            e2e_re_scores = e2e_re_topks = torch.empty(0, dtype=torch.long)
            e2e_predicted_entity_types = torch.empty(0, dtype=torch.long)
            e2e_entity_centric_hts = [[] for _ in range(len(embeddings))]

        if sum([coeff_md, coeff_cr, coeff_et, coeff_re]) == max(
            [coeff_md, coeff_cr, coeff_et, coeff_re]
        ):
            total_loss = max(
                [
                    coeff_md * md_loss,
                    coeff_cr * coref_loss,
                    coeff_et * entity_typing_loss,
                    coeff_re * re_loss,
                ]
            )
        else:
            total_loss = self.awl(
                coeff_md * md_loss,
                coeff_cr * coref_loss,
                coeff_et * entity_typing_loss,
                coeff_re * re_loss,
            )

        md_outputs = {"loss": md_loss, "predictions": predicted_span_labels}
        coref_outputs = {
            "loss": coref_loss,
            "predictions": isolated_predicted_entity_clusters,
            "titles": titles,
        }

        entity_typing_outputs = {
            "loss": entity_typing_loss,
            "predictions": isolated_predicted_entity_types,
            "titles": titles,
        }
        re_outputs = {
            "loss": re_loss,
            "predictions": isolated_re_preds,
            "scores": isolated_re_scores,
            "topks": isolated_re_topks,
            "titles": titles,
        }

        if eval_mode:
            e2e_coref_outputs = {
                "predictions": e2e_predicted_entity_clusters,
                "titles": [titles[i] for i in indexes],
            }
            e2e_entity_typing_outputs = {
                "predictions": e2e_predicted_entity_types,
                "entity_clusters": e2e_et_entity_clusters,
                "titles": [titles[i] for i in e2e_et_indexes_from_origin],
            }
            e2e_re_outputs = {
                "predictions": e2e_re_preds,
                "titles": [titles[i] for i in e2e_re_indexes_from_origin],
                "scores": e2e_re_scores,
                "topks": e2e_re_topks,
                "e2e_entity_centric_hts": e2e_entity_centric_hts,
            }
            return (
                total_loss,
                md_outputs,
                coref_outputs,
                e2e_coref_outputs,
                entity_typing_outputs,
                e2e_entity_typing_outputs,
                re_outputs,
                e2e_re_outputs,
            )
        else:
            return (
                total_loss,
                md_outputs,
                coref_outputs,
                entity_typing_outputs,
                re_outputs,
            )

    def preprocess_e2e(
        self,
        gt_entity_pos,
        predicted_entity_pos,
        hts,
        coreference_labels,
    ):
        """
        Preprocesses the end-to-end (e2e) entities for a given batch.

        This function filters the predicted entity positions to retain only those that match
        the ground truth entity positions. It then constructs hypergraph triplets (hts) and
        coreference labels for the filtered entities. Additionally, it calculates the number
        of false positive predictions (extras) for each sample in the batch.
        Args:
            gt_entity_pos (list of list of tuples): The ground truth entity positions for each
                sample in the batch. Each entity is represented as a tuple (start, end).
            predicted_entity_pos (list of list of tuples): The predicted entity positions for
                each sample in the batch. Each entity is represented as a tuple (start, end).
            hts (list of list of tuples): The hypergraph triplets for each sample in the batch.
                Each triplet is represented as a tuple (head_index, tail_index).
            coreference_labels (list of list of lists): The coreference labels for each hypergraph
                triplet in the batch. Each label is a list of values corresponding to the triplet.
        Returns:
            tuple: A tuple containing:
                - e2e_entity_pos (list of list of tuples): The filtered e2e entity positions for
                  each sample in the batch.
                - e2e_hts (list of list of tuples): The hypergraph triplets for the filtered e2e
                  entities in each sample.
                - e2e_coreference_labels (list of torch.Tensor): The coreference labels for the
                  filtered e2e hypergraph triplets in each sample.
        """
        B = len(gt_entity_pos)
        e2e_entity_pos, e2e_hts, e2e_coreference_labels = [], [], []

        for b in range(B):
            b_gt_entity_pos = [(entity[0], entity[1]) for entity in gt_entity_pos[b]]
            predicted_entity_pos[b] = [entity for entity in predicted_entity_pos[b]]

            b_e2e_entity_pos, b_e2e_hts, b_e2e_coreference_labels = (
                [],
                [],
                [],
            )

            b_e2e_entity_pos = [
                entity
                for entity in predicted_entity_pos[b]
                if (entity[0], entity[1]) in b_gt_entity_pos
            ]
            for i in range(len(b_e2e_entity_pos) - 1):
                for j in range(i + 1, len(b_e2e_entity_pos)):
                    h = (b_e2e_entity_pos[i][0], b_e2e_entity_pos[i][1])
                    t = (b_e2e_entity_pos[j][0], b_e2e_entity_pos[j][1])
                    index_h = b_gt_entity_pos.index(h)
                    index_t = b_gt_entity_pos.index(t)
                    hts_index = hts[b].index((index_h, index_t))
                    b_e2e_hts.append((i, j))
                    b_e2e_coreference_labels.append(
                        coreference_labels[b][hts_index].tolist()
                    )
            e2e_entity_pos.append(b_e2e_entity_pos)
            e2e_hts.append(b_e2e_hts)
            e2e_coreference_labels.append(b_e2e_coreference_labels)
        e2e_coreference_labels = [
            torch.tensor(label) for label in e2e_coreference_labels
        ]
        return (
            e2e_entity_pos,
            e2e_hts,
            e2e_coreference_labels,
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
            b_entity_atts.append(torch.zeros(e_att.size()).to(b_attention))
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

    def get_e2e_entity_types(self, e2e_entity_clusters, entity_clusters, entity_types):
        """
        Get the joint entity types in a batch.

        Args:
            i_e2e_entity_clusters (list): List of joint entity clusters.
            i_entity_clusters (list): List of entity clusters.
            i_entity_types (list): List of entity types.

        Returns:
            list: List of joint entity types.

        """
        e2e_entity_types = [[] for _ in range(len(e2e_entity_clusters))]
        for i in range(len(e2e_entity_clusters)):
            for cluster in e2e_entity_clusters[i]:
                cluster_index = entity_clusters[i].index(cluster)
                e2e_entity_types[i].append(entity_types[i][cluster_index].tolist())
        return [torch.tensor(types) for types in e2e_entity_types]

    def get_e2e_relation_labels(
        self,
        e2e_entity_clusters,
        entity_clusters,
        entity_centric_hts,
        relation_labels,
    ):
        """
        Extracts end-to-end (e2e) relation labels and associated mappings for entity clusters.
        This function processes the provided entity clusters and relation labels to generate
        mappings and labels for end-to-end entity-centric relations.
        Args:
            e2e_entity_clusters (list of list): A list of entity clusters for end-to-end processing,
                where each cluster is a list of entity identifiers.
            entity_clusters (list of list): A list of original entity clusters, where each cluster
                is a list of entity identifiers.
            entity_centric_hts (list of list of tuple): A list of head-tail pairs (tuples) for each
                entity cluster, representing relationships between entities.
            relation_labels (list of list): A list of relation labels for each entity cluster,
                where each label corresponds to a head-tail pair.
        Returns:
            tuple: A tuple containing three lists:
                - e2e_entity_centric_hts (list of list of tuple): Head-tail pairs (indices) for
                  e2e entity clusters.
                - e2e_relation_labels (list of list): Relation labels for the e2e head-tail pairs.
                - e2e_relative_entity_centric_hts (list of list of tuple): Relative head-tail
                  pairs (indices) within the e2e entity clusters.
        """

        e2e_entity_centric_hts, e2e_relation_labels, e2e_relative_entity_centric_hts = (
            [[] for _ in range(len(e2e_entity_clusters))],
            [[] for _ in range(len(e2e_entity_clusters))],
            [[] for _ in range(len(e2e_entity_clusters))],
        )

        for i in range(len(relation_labels)):
            for a, h in enumerate(e2e_entity_clusters[i]):
                for b, t in enumerate(e2e_entity_clusters[i]):
                    if (a != b) and (h != t):
                        h_index = entity_clusters[i].index(h)
                        t_index = entity_clusters[i].index(t)
                        e2e_entity_centric_hts[i].append((h_index, t_index))
                        e2e_relative_entity_centric_hts[i].append((a, b))

                        e2e_relation_labels[i].append(
                            relation_labels[i][
                                entity_centric_hts[i].index((h_index, t_index))
                            ]
                        )

        return (
            e2e_entity_centric_hts,
            e2e_relation_labels,
            e2e_relative_entity_centric_hts,
        )

    def filter_spans(self, span_representations, span_scores):
        filtered_spans = []
        for i in range(span_scores.size(0)):
            # Get the indices of spans with confidence above the threshold
            above_threshold = span_scores[i] > self.threshold
            filtered_spans.append(span_representations[i, above_threshold])
        return filtered_spans
