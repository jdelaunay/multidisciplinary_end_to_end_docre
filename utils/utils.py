import json
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


def load_json(file_path):
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file '{file_path}'.")
        return None


def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [
        [1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"]))
        for f in batch
    ]

    span_idx = pad_sequence(
        [b["span_idx"] for b in batch], batch_first=True, padding_value=0
    )
    span_labels = pad_sequence(
        [b["span_labels"] for b in batch], batch_first=True, padding_value=-1
    )
    span_mask = span_labels != -1
    span_mask = torch.tensor(span_mask, dtype=torch.long)
    coreference_labels = [f["coreference_labels"] for f in batch]
    coreference_labels = [torch.tensor(label) for label in coreference_labels]
    entity_types = [f["entity_types"] for f in batch]
    entity_types = [torch.tensor(label) for label in entity_types]
    hts = [f["hts"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)

    output = {
        "input_ids": input_ids,
        "attention_mask": input_mask,
        "span_idx": span_idx,
        "span_mask": span_mask,
        "span_labels": span_labels,
        "coreference_labels": coreference_labels,
        "hts": hts,
        "entity_pos": [f["entity_pos"] for f in batch],
        "entity_types": entity_types,
    }
    return output
