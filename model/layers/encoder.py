import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):
    def __init__(self, model, tokenizer, transformer_type="roberta"):
        super(Encoder, self).__init__()
        self.model = model
        self.transformer_type = transformer_type
        self.tokenizer = tokenizer
        self.max_position_embeddings = self.model.config.max_position_embeddings
        if transformer_type == "bert":
            self.start_tokens = [self.tokenizer.cls_token_id]
            self.end_tokens = [self.tokenizer.sep_token_id]
        elif transformer_type == "roberta":
            self.start_tokens = [self.tokenizer.cls_token_id]
            self.end_tokens = [self.tokenizer.sep_token_id, self.tokenizer.sep_token_id]
            self.max_position_embeddings = self.model.config.max_position_embeddings - 2
        else:
            raise ValueError(f"Unknown transformer type: {transformer_type}")

    def forward(self, input_ids, attention_mask):
        sequence_output, attention = self.process_long_input(
            input_ids,
            attention_mask,
            start_tokens=self.start_tokens,
            end_tokens=self.start_tokens,
        )
        return sequence_output, attention

    def process_long_input(self, input_ids, attention_mask, start_tokens, end_tokens):
        # Split the input to 2 overlapping chunks. Now BERT can encode inputs of which the length are up to 1024.
        n, c = input_ids.size()
        start_tokens = torch.tensor(start_tokens).to(input_ids)
        end_tokens = torch.tensor(end_tokens).to(input_ids)
        len_start = start_tokens.size(0)
        len_end = end_tokens.size(0)
        if c <= self.max_position_embeddings:
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
            sequence_output = output[0]
            attention = output[-1][-1]
        else:
            new_input_ids, new_attention_mask, num_seg = [], [], []
            seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()
            for i, l_i in enumerate(seq_len):
                if l_i <= self.max_position_embeddings:
                    new_input_ids.append(input_ids[i, : self.max_position_embeddings])
                    new_attention_mask.append(
                        attention_mask[i, : self.max_position_embeddings]
                    )
                    num_seg.append(1)
                else:
                    input_ids1 = torch.cat(
                        [
                            input_ids[i, : self.max_position_embeddings - len_end],
                            end_tokens,
                        ],
                        dim=-1,
                    )
                    input_ids2 = torch.cat(
                        [
                            start_tokens,
                            input_ids[
                                i,
                                (l_i - self.max_position_embeddings + len_start) : l_i,
                            ],
                        ],
                        dim=-1,
                    )
                    attention_mask1 = attention_mask[i, : self.max_position_embeddings]
                    attention_mask2 = attention_mask[
                        i, (l_i - self.max_position_embeddings) : l_i
                    ]
                    new_input_ids.extend([input_ids1, input_ids2])
                    new_attention_mask.extend([attention_mask1, attention_mask2])
                    num_seg.append(2)
            input_ids = torch.stack(new_input_ids, dim=0)
            attention_mask = torch.stack(new_attention_mask, dim=0)
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
            sequence_output = output[0]
            attention = output[-1][-1]
            i = 0
            new_output, new_attention = [], []
            for n_s, l_i in zip(num_seg, seq_len):
                if n_s == 1:
                    output = F.pad(
                        sequence_output[i], (0, 0, 0, c - self.max_position_embeddings)
                    )
                    att = F.pad(
                        attention[i],
                        (
                            0,
                            c - self.max_position_embeddings,
                            0,
                            c - self.max_position_embeddings,
                        ),
                    )
                    new_output.append(output)
                    new_attention.append(att)
                elif n_s == 2:
                    output1 = sequence_output[i][
                        : self.max_position_embeddings - len_end
                    ]
                    mask1 = attention_mask[i][: self.max_position_embeddings - len_end]
                    att1 = attention[i][
                        :,
                        : self.max_position_embeddings - len_end,
                        : self.max_position_embeddings - len_end,
                    ]
                    output1 = F.pad(
                        output1, (0, 0, 0, c - self.max_position_embeddings + len_end)
                    )
                    mask1 = F.pad(
                        mask1, (0, c - self.max_position_embeddings + len_end)
                    )
                    att1 = F.pad(
                        att1,
                        (
                            0,
                            c - self.max_position_embeddings + len_end,
                            0,
                            c - self.max_position_embeddings + len_end,
                        ),
                    )

                    output2 = sequence_output[i + 1][len_start:]
                    mask2 = attention_mask[i + 1][len_start:]
                    att2 = attention[i + 1][:, len_start:, len_start:]
                    output2 = F.pad(
                        output2,
                        (0, 0, l_i - self.max_position_embeddings + len_start, c - l_i),
                    )
                    mask2 = F.pad(
                        mask2, (l_i - self.max_position_embeddings + len_start, c - l_i)
                    )
                    att2 = F.pad(
                        att2,
                        [
                            l_i - self.max_position_embeddings + len_start,
                            c - l_i,
                            l_i - self.max_position_embeddings + len_start,
                            c - l_i,
                        ],
                    )
                    mask = mask1 + mask2 + 1e-10
                    output = (output1 + output2) / mask.unsqueeze(-1)
                    att = att1 + att2
                    att = att / (att.sum(-1, keepdim=True) + 1e-10)
                    new_output.append(output)
                    new_attention.append(att)
                i += n_s
            sequence_output = torch.stack(new_output, dim=0)
            attention = torch.stack(new_attention, dim=0)
        return sequence_output, attention
