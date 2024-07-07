import torch
from torch import nn
from allennlp_light.modules.matrix_attention import (
    CosineMatrixAttention,
    DotProductMatrixAttention,
    BilinearMatrixAttention,
)


class CoreferenceResolver(nn.Module):
    def __init__(self, hidden_size=768):
        super(CoreferenceResolver, self).__init__()
        self.dot_product_attn = DotProductMatrixAttention()
        self.cosine_attn = CosineMatrixAttention()
        self.bilinear_attn = BilinearMatrixAttention(hidden_size, hidden_size)

    def forward(self, x):
        dot_product_sim = self.dot_product_attn(x, x).unsqueeze(-1)
        cosine_sim = self.cosine_attn(x, x).unsqueeze(-1)
        bilinear_sim = self.bilinear_attn(x, x).unsqueeze(-1)
        # attn_input = torch.cat([dot_product_sim, cosine_sim, bilinear_sim], dim=-1).permute(0, 3, 1, 2).contiguous()
        attn_input = cosine_sim.permute(0, 3, 1, 2).contiguous()
        return attn_input
