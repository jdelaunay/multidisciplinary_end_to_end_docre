import torch
from torch import nn
from allennlp_light.modules.matrix_attention import (
    CosineMatrixAttention,
)
from model.layers.attn_unet import AttentionUNet
from model.at_loss import ATLoss
from model.metrics import compute_metrics_multi_class

import torch.nn.functional as F


class CoreferenceResolver(nn.Module):
    """
    A module for resolving coreferences in a given text.

    Args:
        hidden_size (int): The size of the hidden state.
        block_size (int): The size of the block.
        max_height (int): The maximum height.

    Attributes:
        hidden_size (int): The size of the hidden state.
        block_size (int): The size of the block.
        max_height (int): The maximum height.
        cosine_attn (CosineMatrixAttention): The cosine matrix attention module.
        coref_unet (AttentionUNet): The attention U-Net module.
        coref_head_extractor (nn.Linear): The linear layer for extracting the head.
        coref_tail_extractor (nn.Linear): The linear layer for extracting the tail.
        coref_decoder (nn.Linear): The linear layer for decoding.

    Methods:
        get_hrt: Get the head, tail, and entity embeddings.
        get_ht: Get the head and tail embeddings.
        forward: Perform forward pass through the module.

    """

    def __init__(self, hidden_size=768, block_size=64, max_height=4):
        super(CoreferenceResolver, self).__init__()
        self.hidden_size = hidden_size
        self.block_size = block_size
        self.max_height = max_height
        self.cosine_attn = CosineMatrixAttention()
        input_channels = 1
        out_channels = 256
        self.coref_unet = AttentionUNet(
            in_channels=input_channels, out_channels=out_channels
        )  # , down_channel=256)

        self.coref_head_extractor = nn.Linear(
            1 * hidden_size + out_channels, hidden_size
        )
        self.coref_tail_extractor = nn.Linear(
            1 * hidden_size + out_channels, hidden_size
        )
        self.coref_decoder = nn.Linear(hidden_size * block_size, 2)

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

    def get_ht(self, rel_enco, hts):
        """
        Get the head and tail embeddings.

        Args:
            rel_enco (torch.Tensor): The relation encoding tensor.
            hts (list): The list of head-tail pairs.

        Returns:
            torch.Tensor: The head-tail embeddings.

        """
        htss = []
        for i in range(len(hts)):
            ht_index = hts[i]
            for h_index, t_index in ht_index:
                htss.append(rel_enco[i, h_index, t_index])
        htss = torch.stack(htss, dim=0)
        return htss

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
        hs, ts, entity_embs, entity_as, ne = self.get_hrt(
            x, attention_mask, entity_pos, hts
        )

        ent_encode = x.new_zeros(B, ne, D)
        for _b in range(len(entity_embs)):
            entity_emb = entity_embs[_b]
            if type(entity_emb) == list:
                entity_num = 0
                entity_emb = torch.empty(0, self.hidden_size)
            else:
                entity_num = entity_emb.size(0)
            ent_encode[_b, :entity_num, :] = entity_emb

        cosine_sim = self.cosine_attn(ent_encode, ent_encode).unsqueeze(-1)
        attn_input = cosine_sim.permute(0, 3, 1, 2).contiguous()

        attn_map = self.coref_unet(attn_input)
        h_t = self.get_ht(attn_map, hts)

        hs = torch.tanh(self.coref_head_extractor(torch.cat([hs, h_t], dim=1)))
        ts = torch.tanh(self.coref_tail_extractor(torch.cat([ts, h_t], dim=1)))

        b1 = hs.view(-1, self.hidden_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.hidden_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(
            -1, self.hidden_size * self.block_size
        )
        logits = self.coref_decoder(bl)
        predicted_entity_clusters = get_coreference_clusters(
            hts, entity_pos, coreference_labels
        )
        # predicted_entity_clusters = torch.cat(predicted_entity_clusters).to(
        #     logits.device
        # )
        # entity_clusters = torch.cat(entity_clusters).to(logits.device)
        # loss = F.cross_entropy(
        #     predicted_entity_clusters.float(),
        #     entity_clusters.float(),
        #     reduction="mean",
        # )
        precision, recall, f1 = self.compute_metrics(
            predicted_entity_clusters, entity_clusters
        )

        loss_fnt = ATLoss()
        coreference_labels = torch.cat(coreference_labels).to(logits.device)

        loss = loss_fnt(logits.float(), coreference_labels.float())
        # precision, recall, f1 = compute_metrics_multi_class(logits, coreference_labels)
        return predicted_entity_clusters, loss, precision, recall, f1

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


def get_coreference_clusters(hts, entity_pos, coreference_labels):
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
