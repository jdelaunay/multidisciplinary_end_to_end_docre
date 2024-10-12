import torch
from torch import nn
from opt_einsum import contract

from model.at_loss import ATLoss
from model.layers.depthwise_separable_convolution import DepthwiseSeparableConv
from model.layers.attn_unet import AttentionUNet
from model.metrics import compute_metrics_multi_class

import torch.nn.functional as F


class CoreferenceResolver(nn.Module):
    """
    A module for resolving coreferences in a given text.

    Args:
        hidden_size (int): The size of the hidden state.

    Attributes:
        hidden_size (int): The size of the hidden state.
        threshold (nn.Parameter): The learnable threshold for coreference resolution.
        coref_decoder (nn.Linear): The linear layer for decoding.

    Methods:
        get_hrt: Get the head, tail, and entity embeddings.
        forward: Perform forward pass through the module.

    """

    def __init__(self, hidden_size=768):
        super(CoreferenceResolver, self).__init__()
        self.hidden_size = hidden_size
        threshold = torch.tensor(0.9)
        self.threshold = nn.Parameter(threshold, requires_grad=True)
        print("Init threshold", self.threshold)
        self.classifier = nn.Sequential(
            nn.Linear(1, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, 2),
        )
        

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        """
        Get the head, tail, and entity embeddings.

        Args:
            sequence_output (torch.Tensor): The sequence output.
            attention (torch.Tensor): The attention tensor.
            entity_pos (list): The list of entity positions.
            hts (list): The list of head-tail pairs.

        Returns:
            tuple: A tuple containing the head embeddings, tail embeddings,
                   entity embeddings, entity attentions, and the number of entities.

        """
        #offset = 1  # if self.config.transformer_type in ["bert", "roberta"] else 0
        _, h, _, c = attention.size()
        bs = len(entity_pos)
        ne = max([len(b_ents) for b_ents in entity_pos])

        hss, tss = [], []
        entity_es = []
        entity_as = []
        for i in range(bs):
            entity_embs, entity_atts = [], []
            if entity_pos[i]:
                for entity_num, e in enumerate(entity_pos[i]):
                    (start, end) = e
                    if (start < c) and (end < c):
                        e_emb = sequence_output[i, start : end]
                        e_emb = e_emb.mean(0)
                        e_att = attention[i, :, start]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                    entity_embs.append(e_emb)
                    entity_atts.append(e_att)
                for _ in range(ne - entity_num - 1):
                    entity_atts.append(e_att)

                entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
                entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

            entity_es.append(entity_embs)
            entity_as.append(entity_atts)
            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            if len(ht_i) > 0:
                hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
                ts = torch.index_select(entity_embs, 0, ht_i[:, 1])
            else:
                hs = torch.empty(0, self.hidden_size).to(sequence_output.device)
                ts = torch.empty(0, self.hidden_size).to(sequence_output.device)
            hss.append(hs)
            tss.append(ts)
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        return hss, tss, entity_es, entity_as, ne

    

    def forward(
        self, x, attention_mask, entity_pos, hts, coreference_labels, entity_clusters
    ):
        """
        Perform a forward pass through the module.

        Args:
            x (torch.Tensor): The input tensor.
            attention_mask (torch.Tensor): The attention mask.
            entity_pos (list): The list of entity positions.
            hts (list): The list of head-tail pairs.
            coreference_labels (list): The list of coreference labels.

        Returns:
            tuple: A tuple containing the loss, precision, recall, and F1 score.

        """
        B, L, D = x.size()

        # get hs, ts and entity_embs >> entity_rs
        hs, ts, entity_embs, _, ne = self.get_hrt(
            x, attention_mask, entity_pos, hts
        )

        similarities = []
        for _b in range(len(entity_embs)):
            entity_emb = entity_embs[_b]
            cosine_sim = F.cosine_similarity(entity_emb.unsqueeze(1), entity_emb.unsqueeze(0), dim=-1)
            cosine_sim = (cosine_sim - self.threshold) / (cosine_sim.std() + 1e-5)
            cosine_sim = torch.triu(cosine_sim, diagonal=1).unsqueeze(-1)
            for i in range(cosine_sim.size(1) - 1):
                for j in range(i+1, cosine_sim.size(1)):
                    similarities.append(cosine_sim[i, j])
            
        logits = self.classifier(torch.stack(similarities).to(x.device))

        loss_fnt = ATLoss()
        labels = torch.cat(coreference_labels).to(logits.device)

        loss = loss_fnt(logits.float(), labels.float())

        pair_precision, pair_recall, pair_f1 = compute_metrics_multi_class(
            logits, labels
        )

        predicted_entity_clusters = self.get_coreference_clusters(
            hts, entity_pos, logits
        )

        precision, recall, f1 = self.compute_metrics(
            predicted_entity_clusters, entity_clusters
        )
        b3_precision, b3_recall, b3_f1 = self.b3_score(
            predicted_entity_clusters, entity_clusters
        )

        return (
            predicted_entity_clusters,
            loss,
            precision,
            recall,
            f1,
            b3_precision,
            b3_recall,
            b3_f1,
            pair_precision,
            pair_recall,
            pair_f1,
        )

    def compute_metrics(self, predicted_clusters, ground_truth_clusters):
        """
        Compute recall, precision, and F1 score between predicted entity clusters and ground truth entity clusters.

        Args:
            predicted_clusters (list): List of predicted entity clusters.
            ground_truth_clusters (list): List of ground truth entity clusters.

        Returns:
            float: Recall score.
            float: Precision score.
            float: F1 score.

        """
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for i in range(len(predicted_clusters)):
            for pred_cluster in predicted_clusters[i]:
                if pred_cluster in ground_truth_clusters[i]:
                    true_positives += 1
                else:
                    false_positives += 1

            for true_cluster in ground_truth_clusters[i]:
                if true_cluster not in predicted_clusters[i]:
                    false_negatives += 1

        if true_positives == 0:
            return 0, 0, 0
        else:
            recall = true_positives / (true_positives + false_negatives)
            precision = true_positives / (true_positives + false_positives)
            f1 = 2 * (precision * recall) / (precision + recall)

        return precision, recall, f1

    def get_coreference_clusters(self, hts, entity_pos, logits):
        """
        Get the coreference clusters from the given inputs.

        Args:
            hts (list): List of lists containing the indices of the entities involved in each coreference pair.
            entity_pos (list): List of lists containing the positions of the entities in the input.
            coreference_labels (list): List of lists containing the labels indicating whether a coreference pair exists.

        Returns:
            list: List of coreference clusters, where each cluster is a list of entities.

        """
        logits = F.softmax(logits, dim=1)
        coreferences_clusters = []
        offset = 0
        for i in range(len(hts)):
            coreferences = []
            for j in range(len(hts[i])):
                if logits[offset + j][1] >= 0.5:
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
                    if ent1_cluster != ent2_cluster:
                        if ent1_cluster:
                            # Add ent2 to ent1_cluster
                            ent1_cluster.append(ent2)
                        elif ent2_cluster:
                            # Add ent1 to ent2_cluster
                            ent2_cluster.append(ent1)
                    elif not ent1_cluster and not ent2_cluster:
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
            coreferences = [list(set(cluster)) for cluster in coreferences]
            for index, c1 in enumerate(coreferences[:-1]):
                for c2 in coreferences[index + 1 :]:
                    if (c2 == c1) or (c2 in c1):
                        coreferences.remove(c2)
                        if len(coreferences) == 1:
                            break
            coreferences = [sorted(cluster) for cluster in coreferences]
            coreferences_clusters.append(sorted(coreferences))
            offset += len(hts[i])

        return coreferences_clusters

    def b3_score(self, predicted_clusters, true_clusters):
        """
        Calculate the B^3 (Bagga and Baldwin) precision, recall, and F1 score.

        Parameters:
            predicted_clusters (list of list of int): List of predicted clusters with mention indices.
            true_clusters (list of list of int): List of true clusters with mention indices.

        Returns:
            precision (float): B^3 precision score.
            recall (float): B^3 recall score.
            f1_score (float): B^3 F1 score.
        """

        def get_mention_to_cluster_map(clusters):
            mention_to_cluster = {}
            for cluster_id, cluster in enumerate(clusters):
                for mention in cluster:
                    mention_to_cluster[mention] = cluster_id
            return mention_to_cluster

        precision_sum = 0.0
        recall_sum = 0.0
        total_mentions = 0
        for i in range(len(predicted_clusters)):
            pred_mention_to_cluster = get_mention_to_cluster_map(predicted_clusters[i])
            true_mention_to_cluster = get_mention_to_cluster_map(true_clusters[i])

            all_mentions = set(pred_mention_to_cluster.keys()).union(
                set(true_mention_to_cluster.keys())
            )

            for mention in all_mentions:
                if (
                    mention in pred_mention_to_cluster
                    and mention in true_mention_to_cluster
                ):
                    pred_cluster = predicted_clusters[i][
                        pred_mention_to_cluster[mention]
                    ]
                    true_cluster = true_clusters[i][true_mention_to_cluster[mention]]

                    intersection = len(
                        set(pred_cluster).intersection(set(true_cluster))
                    )
                    precision = intersection / len(pred_cluster)
                    recall = intersection / len(true_cluster)

                    precision_sum += precision
                    recall_sum += recall
                elif mention in pred_mention_to_cluster:
                    precision_sum += 0
                    recall_sum += 0
                elif mention in true_mention_to_cluster:
                    precision_sum += 0
                    recall_sum += 0

                total_mentions += 1

        if total_mentions == 0:
            return 0.0, 0.0, 0.0
        else:
            precision = precision_sum / total_mentions
            recall = recall_sum / total_mentions
            f1_score = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            return precision, recall, f1_score