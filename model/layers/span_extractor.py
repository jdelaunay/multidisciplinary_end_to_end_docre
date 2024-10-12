import torch

import torch.nn as nn

class SpanClassifier(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(SpanClassifier, self).__init__()
                
        self.linear_start = nn.Linear(hidden_size, hidden_size)  # Linear layer for start position
        self.linear_end = nn.Linear(hidden_size, hidden_size)  # Linear layer for end position
        self.linear_span = nn.Linear(2 * hidden_size, num_labels)  # Linear layer for span classification
        self.softmax = nn.Softmax(dim=-1)  # Softmax layer for probability scores

    
    def forward(self, x):
        # Linear projections for start and end positions
        start_logits = self.linear_start(x)
        end_logits = self.linear_end(x)
        
        # Computing span scores
        span_logits = []
        for i in range(x.size(1)):
            max_width = x.size(1) - i
            for j in range(i, i + max_width):
                span_vector = torch.cat([start_logits[:, i, :], end_logits[:, j, :]], dim=-1)
                span_logit = self.linear_span(span_vector)
                span_logits.append(span_logit)
                
        span_logits = torch.stack(span_logits, dim=1)
        span_probs = self.softmax(span_logits)  # Apply softmax to obtain probabilities
        
        return span_probs
