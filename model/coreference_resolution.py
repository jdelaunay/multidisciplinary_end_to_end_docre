import torch
from torch import nn
from allennlp_light.modules.matrix_attention import (
    CosineMatrixAttention,
)
from model.layers.attn_unet import AttentionUNet
from model.at_loss import ATLoss
from model.metrics import compute_metrics_rels


class CoreferenceResolver(nn.Module):
    def __init__(self, hidden_size=768, block_size=64, max_height=4):
        super(CoreferenceResolver, self).__init__()
        self.hidden_size = hidden_size
        self.block_size = block_size
        self.max_height = max_height
        self.cosine_attn = CosineMatrixAttention()
        input_channels = out_channels = 1
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
                print(hs.size(), ts.size())
            else:
                hs = torch.empty(0, self.hidden_size).to(sequence_output.device)
                ts = torch.empty(0, self.hidden_size).to(sequence_output.device)
            hss.append(hs)
            tss.append(ts)
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        return hss, tss, entity_es, entity_as, ne

    def get_ht(self, rel_enco, hts):
        htss = []
        for i in range(len(hts)):
            ht_index = hts[i]
            for h_index, t_index in ht_index:
                htss.append(rel_enco[i, h_index, t_index])
        htss = torch.stack(htss, dim=0)
        return htss

    def forward(self, x, attention_mask, entity_pos, hts, coreference_labels):
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
        loss_fnt = ATLoss()
        coreference_labels = torch.cat(coreference_labels).to(logits.device)

        loss = loss_fnt(logits.float(), coreference_labels.float())
        precision, recall, f1 = compute_metrics_rels(logits, coreference_labels)
        return loss, precision, recall, f1
