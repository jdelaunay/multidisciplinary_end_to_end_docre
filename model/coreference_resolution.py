import torch
from torch import nn
from opt_einsum import contract

from model.losses import ATLoss, FocalLoss
from model.layers.depthwise_separable_convolution import DepthwiseSeparableConv

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
        forward: Perform a forward pass through the module.
        compute_metrics: Compute precision, recall, and F1 score.
        get_coreference_clusters: Get the coreference clusters from the given inputs.
        b3_score: Calculate the B^3 precision, recall, and F1 score.

    """

    def __init__(self, hidden_size=768):
        super(CoreferenceResolver, self).__init__()
        self.hidden_size = hidden_size
        threshold = torch.tensor(0.9)
        self.threshold = nn.Parameter(threshold, requires_grad=True)
        print("Init threshold", self.threshold)
        self.classifier = nn.Sequential(
            nn.Linear(2 * self.hidden_size + 1, self.hidden_size),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 512),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.4),
            nn.Linear(2 * self.hidden_size + 1, self.hidden_size),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 512),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Linear(256, 2),
        )

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        """
        Get the head, tail, and entity embeddings.

        Args:
            sequence_output (torch.Tensor): The sequence output tensor of shape (batch_size, seq_len, hidden_size).
            attention (torch.Tensor): The attention tensor of shape (batch_size, num_heads, seq_len, seq_len).
            entity_pos (list): A list of lists containing the start and end positions of entities for each batch.
            hts (list): A list of lists containing the head-tail pairs for each batch.

        Returns:
            tuple: A tuple containing:
            - hss (torch.Tensor): The head embeddings of shape (num_pairs, hidden_size).
            - tss (torch.Tensor): The tail embeddings of shape (num_pairs, hidden_size).
            - entity_es (list): A list of tensors containing entity embeddings for each batch.
            - entity_as (list): A list of tensors containing entity attentions for each batch.
            - ne (int): The maximum number of entities in any batch.
        """
        # offset = 1  # if self.config.transformer_type in ["bert", "roberta"] else 0
        # offset = 1  # if self.config.transformer_type in ["bert", "roberta"] else 0
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
                        e_emb = sequence_output[i, start:end]
                        e_emb, _ = torch.max(e_emb, dim=0)
                        e_emb = sequence_output[i, start:end]
                        e_emb, _ = torch.max(e_emb, dim=0)
                        e_att = attention[i, :, start]
                    else:
                        e_emb = torch.zeros(self.hidden_size).to(sequence_output)
                        e_emb = torch.zeros(self.hidden_size).to(sequence_output)
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

    def forward(self, x, attention_mask, entity_pos, hts, coreference_labels):
        """
        Perform a forward pass through the module.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, hidden_size).
            attention_mask (torch.Tensor): The attention mask tensor of shape (batch_size, num_heads, seq_len, seq_len).
            entity_pos (list): A list of lists containing the start and end positions of entities for each batch.
            hts (list): A list of lists containing the head-tail pairs for each batch.
            coreference_labels (list): A list of tensors containing the coreference labels for each batch.
            entity_clusters (list): A list of lists containing the ground truth entity clusters for each batch.

        Returns:
            tuple: A tuple containing:
            - predicted_entity_clusters (list): The predicted coreference clusters.
            - loss (torch.Tensor): The computed loss.
            - precision (float): The precision score.
            - recall (float): The recall score.
            - f1 (float): The F1 score.
            - b3_precision (float): The B^3 precision score.
            - b3_recall (float): The B^3 recall score.
            - b3_f1 (float): The B^3 F1 score.
            - pair_precision (float): The pairwise precision score.
            - pair_recall (float): The pairwise recall score.
            - pair_f1 (float): The pairwise F1 score.
        """
        B, L, D = x.size()

        # get hs, ts and entity_embs >> entity_rs
        hs, ts, entity_embs, _, ne = self.get_hrt(x, attention_mask, entity_pos, hts)
        hs, ts, entity_embs, _, ne = self.get_hrt(x, attention_mask, entity_pos, hts)

        similarities, heads_embs, tails_embs = [], [], []
        similarities, heads_embs, tails_embs = [], [], []
        for _b in range(len(entity_embs)):
            entity_emb = entity_embs[_b]
            cosine_sim = F.cosine_similarity(
                entity_emb.unsqueeze(1), entity_emb.unsqueeze(0), dim=-1
            )
            cosine_sim = F.cosine_similarity(
                entity_emb.unsqueeze(1), entity_emb.unsqueeze(0), dim=-1
            )
            cosine_sim = (cosine_sim - self.threshold) / (cosine_sim.std() + 1e-5)
            cosine_sim = torch.triu(cosine_sim, diagonal=1).unsqueeze(-1)
            for i in range(cosine_sim.size(1) - 1):
                for j in range(i + 1, cosine_sim.size(1)):
                    similarities.append(cosine_sim[i, j]),
                    heads_embs.append(entity_emb[i])
                    tails_embs.append(entity_emb[j])

        similarities = torch.stack(similarities).to(x.device)
        heads_embs = torch.stack(heads_embs).to(x.device).view(similarities.size(0), -1)
        tails_embs = torch.stack(tails_embs).to(x.device).view(similarities.size(0), -1)
        inputs = torch.cat([similarities, heads_embs, tails_embs], dim=1).view(
            -1, 2 * self.hidden_size + 1
        )
        logits = self.classifier(inputs)
        # logits = self.classifier(torch.stack(similarities).to(x.device))

        loss_fnt = FocalLoss()
        labels = torch.cat(coreference_labels).to(logits.device)
        logits = logits.view(labels.size())
        logits = logits.view(labels.size())

        loss = loss_fnt(logits.float(), labels.float())

        predicted_entity_clusters = self.get_coreference_clusters(
            hts, entity_pos, logits
        )

        return (
            predicted_entity_clusters,
            loss,
        )

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
                remaining_entity = remaining_entity
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
