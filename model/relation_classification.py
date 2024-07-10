import torch
import torch.nn as nn
from opt_einsum import contract

from model.at_loss import ATLoss

from model.layers.attn_unet import AttentionUNet
from model.metrics import compute_metrics_rels


class UNet_Relation_Extractor(nn.Module):
    def __init__(self, hidden_size=768, block_size=64, num_labels=-1, max_height=4):
        super(UNet_Relation_Extractor, self).__init__()
        self.hidden_size = hidden_size
        self.max_height = max_height
        self.block_size = block_size
        self.at_loss = ATLoss()
        input_channels = max_height
        out_channels = 256
        self.liner = nn.Linear(hidden_size, input_channels)
        self.relation_unet = AttentionUNet(
            in_channels=input_channels, out_channels=out_channels
        )
        self.head_extractor = nn.Linear(1 * hidden_size + out_channels, hidden_size)
        self.tail_extractor = nn.Linear(1 * hidden_size + out_channels, hidden_size)

        self.relation_classifier = nn.Linear(hidden_size * block_size, num_labels)

    def get_ht(self, rel_enco, hts):
        htss = []
        for i in range(len(hts)):
            ht_index = hts[i]
            for h_index, t_index in ht_index:
                htss.append(rel_enco[i, h_index, t_index])
        htss = torch.stack(htss, dim=0)
        return htss

    def get_htss(self, entity_embeddings, entity_attentions, entity_centric_hts):
        hss, tss = [], []
        for i in range(len(entity_centric_hts)):
            ht_i = torch.LongTensor(entity_centric_hts[i]).to(entity_embeddings.device)
            hs = torch.index_select(entity_embeddings, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embeddings, 0, ht_i[:, 1])
            hss.append(hs)
            tss.append(ts)
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        return hss, tss

    def get_channel_map(self, sequence_output, entity_as):
        # sequence_output = sequence_output.to('cpu')
        # attention = attention.to('cpu')
        bs, _, d = sequence_output.size()
        # ne = max([len(x) for x in entity_as])  # 本次bs中的最大实体数
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
        hs, ts = self.get_htss(entity_embeddings, entity_attentions, entity_centric_hts)

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

        logits = self.relation_classifier(bl)
        labels = [torch.tensor(label) for label in labels]
        labels = torch.cat(labels, dim=0).to(logits)
        loss = self.at_loss(logits.float(), labels.float())
        precision, recall, f1 = compute_metrics_rels(logits, labels)
        return loss, precision, recall, f1
