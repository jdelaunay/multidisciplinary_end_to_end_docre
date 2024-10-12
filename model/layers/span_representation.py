import torch
from torch import nn
import torch.nn.functional as F
from allennlp_light.modules.span_extractors import EndpointSpanExtractor, SelfAttentiveSpanExtractor

class SpanConvBlock(nn.Module):
    def __init__(self, hidden_size, kernel_size, span_mode='conv_normal'):
        super().__init__()

        if span_mode == 'conv_conv':
            self.conv = nn.Conv1d(hidden_size, hidden_size,
                                  kernel_size=kernel_size)

            # initialize the weights
            nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='relu')

        elif span_mode == 'conv_max':
            self.conv = nn.MaxPool1d(kernel_size=kernel_size, stride=1)
        elif span_mode == 'conv_mean' or span_mode == 'conv_sum':
            self.conv = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

        self.span_mode = span_mode

        self.pad = kernel_size - 1

    def forward(self, x):

        x = torch.einsum('bld->bdl', x)

        if self.pad > 0:
            x = F.pad(x, (0, self.pad), "constant", 0)

        x = self.conv(x)

        if self.span_mode == "conv_sum":
            x = x * (self.pad + 1)

        return torch.einsum('bdl->bld', x)


class SpanConv(nn.Module):
    def __init__(self, hidden_size, max_width=4, span_mode='conv_max'):
        super().__init__()

        kernels = [i + 2 for i in range(max_width - 1)]

        self.convs = nn.ModuleList()

        for kernel in kernels:
            self.convs.append(SpanConvBlock(hidden_size, kernel, span_mode))

        self.project = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x, *args):

        span_reps = [x]

        for conv in self.convs:
            h = conv(x)
            span_reps.append(h)

        span_reps = torch.stack(span_reps, dim=-2)

        return self.project(span_reps)
    
class SpanAttention(nn.Module):

    def __init__(self, hidden_size, max_width, width_embedding=128):
        super().__init__()

        self.span_extractor = SelfAttentiveSpanExtractor(hidden_size,
                                                         num_width_embeddings=max_width,
                                                         span_width_embedding_dim=width_embedding,
                                                        )
        self.downproject = nn.Sequential(
            nn.Linear(hidden_size + width_embedding, hidden_size),
            nn.ReLU()
        )

    def forward(self, h, span_idx):
        # h of shape [B, L, D]
        # query_seg of shape [D, max_width]

        B, L, D = h.size()

        span_rep = self.span_extractor(h, span_idx)

        return self.downproject(span_rep).view(B, L, -1, D)
    
class SpanEndpointsBlock(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()

        self.kernel_size = kernel_size

    def forward(self, x):
        B, L, D = x.size()

        span_idx = torch.LongTensor(
            [[i, i + self.kernel_size - 1] for i in range(L)]).to(x.device)

        x = F.pad(x, (0, 0, 0, self.kernel_size - 1), "constant", 0)

        # endrep
        start_end_rep = torch.index_select(x, dim=1, index=span_idx.view(-1))

        start_end_rep = start_end_rep.view(B, L, 2, D)

        return start_end_rep, span_idx


class SpanEndpointsV2(nn.Module):
    def __init__(self, max_width, span_mode='endpoints_logsumexp'):
        super().__init__()

        assert span_mode in ['endpoints_mean', 'endpoints_logsumexp',
                             'endpoints_max', 'endpoints_cat']

        self.K = max_width

        kernels = [i + 1 for i in range(max_width)]

        self.convs = nn.ModuleList()

        for kernel in kernels:
            self.convs.append(SpanEndpointsBlock(kernel))

        self.span_mode = span_mode

    def forward(self, x, *args):
        B, L, D = x.size()

        span_reps = []
        span_idx = []

        for conv in self.convs:
            outputs = conv(x)
            span_reps.append(outputs[0])
            span_idx.append(outputs[1])

        span_reps = torch.stack(span_reps, dim=-3)
        span_idx = torch.stack(span_idx, dim=-2)

        if self.span_mode == 'endpoints_mean':
            span_reps = torch.mean(span_reps, dim=-2)
        elif self.span_mode == 'endpoints_logsumexp':
            span_reps = torch.logsumexp(span_reps, dim=-2)
        elif self.span_mode == 'endpoints_max':
            span_reps = torch.max(span_reps, dim=-2).values
        elif self.span_mode == 'endpoints_cat':
            span_reps = span_reps.view(B, L, self.K, -1)

        return span_reps, span_idx


    
