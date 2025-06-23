import torch
from model.layers.depthwise_separable_convolution import DepthwiseSeparableConv


import torch.nn as nn


class CoreferenceResolver(nn.Module):
    def __init__(self):
        super(CoreferenceResolver, self).__init__()
        self.convolution = DepthwiseSeparableConv(
            in_channels=1, out_channels=256, kernel_size=3, padding=1
        )
        self.classifier = nn.Linear(256, 1)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(
        self, x, attention_mask, entity_pos, hts, coreference_labels, entity_clusters
    ):
        B, L, D = x.size()
        # Extract mention embeddings
        hs, ts, mention_embs, mention_as, ne = self.get_hrt(
            x, attention_mask, entity_pos, hts
        )

        mention_encode = x.new_zeros(B, ne, D)
        for _b in range(len(mention_embs)):
            mention_emb = mention_embs[_b]
            if type(mention_emb) == list:
                mention_num = 0
                mention_emb = torch.empty(0, self.hidden_size)
            else:
                mention_num = mention_emb.size(0)
            mention_encode[_b, :mention_num, :] = mention_emb

        # Compute pairwise similarity matrix
        similarity_matrix = torch.matmul(mention_encode, mention_encode.transpose(1, 2))
        similarity_matrix = torch.tril(similarity_matrix, diagonal=-1)

        x = (
            self.convolution(
                similarity_matrix.unsqueeze(-1).permute(0, 3, 1, 2).contiguous()
            )
            .permute(0, 2, 3, 1)
            .contiguous()
        )
        logits = self.classifier(x)

        # Compute cross entropy loss
        labels = torch.cat(coreference_labels).to(logits.device)
        # logits = torch.cat(similarity_scores).to(similarity_scores.device)
        loss = self.criterion(logits.view(-1), labels.view(-1))

        # Threshold probabilities to get binary labels
        pairwise_labels = torch.where(
            similarity_matrix > 0.5,
            torch.ones_like(similarity_matrix),
            torch.zeros_like(similarity_matrix),
        )
        # Convert binary labels to coreference clusters
        coreference_clusters = []
        for b in range(B):
            cluster = []
            for m1 in range(ne):
                for m2 in range(m1 + 1, ne):
                    if pairwise_labels[b, m1, m2] == 1:
                        cluster.append((m1, m2))
            coreference_clusters.append(sorted(list(set(cluster))))

        # Compute metrics
        precision, recall, f1 = self.compute_metrics(
            coreference_clusters, entity_clusters
        )
        b3_precision, b3_recall, b3_f1 = self.b3_score(
            coreference_clusters, entity_clusters
        )

        return (
            coreference_clusters,
            loss,
            precision,
            recall,
            f1,
            b3_precision,
            b3_recall,
            b3_f1,
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
        offset = 1  # if self.config.transformer_type in ["bert", "roberta"] else 0
        _, h, _, c = attention.size()
        bs = len(entity_pos)
        ne = max([len(x) for x in entity_pos])

        hss, tss = [], []
        entity_es = []
        entity_as = []
        for i in range(bs):
            entity_embs, entity_atts = [], []
            if entity_pos[i]:
                for entity_num, e in enumerate(entity_pos[i]):
                    (start, end) = e
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
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

        precision = precision_sum / total_mentions
        recall = recall_sum / total_mentions
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return precision, recall, f1_score
