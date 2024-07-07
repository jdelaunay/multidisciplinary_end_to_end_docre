from transformers import RobertaModel, RobertaTokenizer
import torch
from torch import nn
from allennlp_light.modules.span_extractors import EndpointSpanExtractor

# 1. Load the RoBERTa model and tokenizer
model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
roberta_model = RobertaModel.from_pretrained(model_name)

# 2. Prepare input data
text = ["Barack Obama was born in Hawaii.", "Apple released the iPhone 13."]

# Tokenize the input text and get input IDs and attention masks
encoding = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=20)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']

# Get RoBERTa embeddings (outputs is a tuple, we need the first element)
with torch.no_grad():
    outputs = roberta_model(input_ids, attention_mask=attention_mask)
    sentence_embeddings = outputs[0]  # shape: (batch_size, seq_length, hidden_size)

# 3. Define spans (for simplicity, we assume spans are provided)
# Span indices in the format [start, end], e.g., [start, end] for each span
# Example: [(0, 1), (4, 4), (6, 7)] -> first two tokens, fifth token, seventh and eighth token
span_indices = torch.tensor([[[1, 2], [5, 5], [6, 8]],
                             [[0, 1], [2, 3], [4, 4]]])  # (batch_size, num_spans, 2)

# 4. Initialize the EndpointSpanExtractor
input_dim = roberta_model.config.hidden_size
span_extractor = EndpointSpanExtractor(input_dim=input_dim,
                                       num_width_embeddings=10,
                                       span_width_embedding_dim=5,
                                       combination="x,y")

# 5. Extract span representations
span_representations = span_extractor(sentence_embeddings, span_indices)

print("Span Representations Shape:", span_representations.shape)
print("Span Representations:", span_representations)

# Step 2: Define a simple scoring function
class SpanScorer(nn.Module):
    def __init__(self, input_dim):
        super(SpanScorer, self).__init__()
        self.scorer = nn.Linear(input_dim, 1)
    
    def forward(self, span_representations):
        return self.scorer(span_representations).squeeze(-1)  # (batch_size, num_spans)

# Initialize the span scorer
span_scorer = SpanScorer(input_dim=span_representations.size(-1))

# Compute span scores
span_scores = span_scorer(span_representations)  # (batch_size, num_spans)
print("Span Scores:", span_scores)

# Step 3: Identify the best spans
best_spans = []
for i in range(span_scores.size(0)):
    # Get the indices of the top spans for each batch item
    top_indices = torch.argsort(span_scores[i], descending=True)
    best_spans.append(span_indices[i, top_indices[0]].tolist())

print("Best Spans:", best_spans)

# Step 4: Decode spans to get the boundaries and corresponding text
for i, spans in enumerate(best_spans):
    start, end = spans
    # Decode the original text
    decoded_span = tokenizer.decode(input_ids[i, start:end+1])
    print(f"Sentence {i+1}, Best Span: ({start}, {end}) -> Text: '{decoded_span}'")
