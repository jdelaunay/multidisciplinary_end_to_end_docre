import torch
import torch.nn as nn
from opt_einsum import contract

from model.losses import ATLoss

from model.layers.attn_unet import AttentionUNet


class UNet_Relation_Extractor(nn.Module):
    """
    Implements a UNet-based relation extraction model.

    Args:
        hidden_size (int): The size of the hidden layer.
        block_size (int): The size of the block.
        num_labels (int): The number of labels.
        max_height (int): The maximum height.

    Attributes:
        hidden_size (int): The size of the hidden layer.
        max_height (int): The maximum height.
        block_size (int): The size of the block.
        at_loss (ATLoss): The ATLoss object.
        liner (nn.Linear): The linear layer.
        relation_unet (AttentionUNet): The AttentionUNet object.
        head_extractor (nn.Linear): The linear layer for head extraction.
        tail_extractor (nn.Linear): The linear layer for tail extraction.
        relation_classifier (nn.Linear): The linear layer for relation classification.

    Methods:
        get_ht(self, rel_enco, hts): Returns the ht values.
        get_htss(self, entity_embeddings, entity_attentions, entity_centric_hts): Returns the hss and tss values.
        get_channel_map(self, sequence_output, entity_as): Returns the channel map.
        forward(self, x, entity_embeddings, entity_attentions, entity_centric_hts, labels): Performs forward pass of the model.
    """

    def __init__(
        self,
        hidden_size=768,
        block_size=64,
        num_labels=-1,
        max_height=5,
        depthwise=True,
        loss_type="",
    ):
        super(UNet_Relation_Extractor, self).__init__()
        self.hidden_size = hidden_size
        self.max_height = max_height
        self.block_size = block_size
        self.num_labels = num_labels
        self.loss_fnt = ATLoss()
        self.loss_type = loss_type
        input_channels = max_height
        out_channels = 256
        self.liner = nn.Linear(hidden_size, input_channels)
        self.relation_unet = AttentionUNet(
            in_channels=input_channels, out_channels=out_channels, depthwise=depthwise
        )
        self.head_extractor = nn.Linear(1 * hidden_size + out_channels, hidden_size)
        self.tail_extractor = nn.Linear(1 * hidden_size + out_channels, hidden_size)

        self.dropout = nn.Dropout(0.6)

        self.relation_classifier = nn.Linear(hidden_size * block_size, num_labels)

    def get_ht(self, rel_enco, hts):
        """
        Returns the ht values.

        Args:
            rel_enco (Tensor): The relation encoding tensor.
            hts (List[List[Tuple[int, int]]]): The list of ht indices.

        Returns:
            Tensor: The ht values.
        """
        htss = []
        for i in range(len(hts)):
            ht_index = hts[i]
            for h_index, t_index in ht_index:
                htss.append(rel_enco[i, h_index, t_index])
        htss = torch.stack(htss, dim=0)
        return htss

    def get_htss(self, sequence_outputs, entity_embeddings, entity_centric_hts):
        """
        Returns the hss and tss values.

        Args:
            entity_embeddings (Tensor): The entity embeddings tensor.
            entity_attentions (List[Tensor]): The list of entity attentions tensors.
            entity_centric_hts (List[List[Tuple[int, int]]]): The list of entity-centric ht indices.

        Returns:
            Tuple[Tensor, Tensor]: The hss and tss values.
        """
        hss, tss = [], []
        for i in range(len(entity_centric_hts)):
            ht_i = torch.LongTensor(entity_centric_hts[i]).to(sequence_outputs.device)
            hs = torch.index_select(entity_embeddings[i], 0, ht_i[:, 0])
            ts = torch.index_select(entity_embeddings[i], 0, ht_i[:, 1])
            hss.append(hs)
            tss.append(ts)
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        return hss, tss

    def get_channel_map(self, sequence_output, entity_as):
        """
        Returns the channel map.

        Args:
            sequence_output (Tensor): The sequence output tensor.
            entity_as (List[Tensor]): The list of entity attention tensors.

        Returns:
            Tensor: The channel map.
        """
        bs, _, d = sequence_output.size()
        ne = self.max_height

        index_pair = []
        for i in range(ne):
            tmp = torch.cat(
                (torch.ones((ne, 1), dtype=int) * i, torch.arange(0, ne).unsqueeze(1)),
                dim=-1,
            )
            index_pair.append(tmp)
        index_pair = (
            torch.stack(index_pair, dim=0).reshape(-1, 2).to(sequence_output.device)
        )
        map_rss = []
        for b in range(bs):
            entity_atts = entity_as[b]
            h_att = torch.index_select(entity_atts, 0, index_pair[:, 0])
            t_att = torch.index_select(entity_atts, 0, index_pair[:, 1])
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = contract("ld,rl->rd", sequence_output[b], ht_att)
            map_rss.append(rs)
        map_rss = torch.cat(map_rss, dim=0).reshape(bs, ne, ne, d)
        return map_rss

    def forward(
        self, x, entity_embeddings, entity_attentions, entity_centric_hts, labels
    ):
        """
        Performs forward pass of the module.

        Args:
            x (Tensor): The input tensor.
            entity_embeddings (Tensor): The entity embeddings tensor.
            entity_attentions (List[Tensor]): The list of entity attentions tensors.
            entity_centric_hts (List[List[Tuple[int, int]]]): The list of entity-centric ht indices.
            labels (List[int]): The list of labels.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: The loss, precision, recall, and f1 score.
        """
        hs, ts = self.get_htss(x, entity_embeddings, entity_centric_hts)

        feature_map = self.get_channel_map(x, entity_attentions)
        attn_input = self.liner(feature_map).permute(0, 3, 1, 2).contiguous()
        attn_map = self.relation_unet(attn_input)

        h_t = self.get_ht(attn_map, entity_centric_hts)
        hs = torch.tanh(self.head_extractor(torch.cat([hs, h_t], dim=1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, h_t], dim=1)))
        b1 = hs.view(-1, self.hidden_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.hidden_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(
            -1, self.hidden_size * self.block_size
        )
        bl = self.dropout(bl)

        logits = self.relation_classifier(bl)
        labels = [torch.tensor(label) for label in labels]
        labels = torch.cat(labels, dim=0).to(logits)
        loss = self.loss_fnt(logits.float(), labels.float())
        predicted_labels = self.loss_fnt.get_label(logits, num_labels=self.num_labels)

        scores_topk = self.loss_fnt.get_score(logits, self.num_labels)
        scores = scores_topk[0]
        topks = scores_topk[1]

        return predicted_labels, scores, topks, loss
