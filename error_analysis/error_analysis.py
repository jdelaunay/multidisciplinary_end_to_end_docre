import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from transformers import AutoTokenizer
import sentencepiece
import json
import os
import sys


def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def plot_confusion_matrix(
    y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    # else:
    # print("Confusion matrix, without normalization")

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    return ax


def get_specific_type_labels(data, types, tokenizer, max_span_width=15):
    type_specific_span_labels = {}
    total_relation_labels = []
    for type in types:
        type_specific_span_labels[type] = []
    for sample in data:
        vertex_set = sample["vertexSet"]
        for i, entity in enumerate(vertex_set):
            for mention in entity:
                mention["entity_id"] = i
        mentions = [mention for entity in vertex_set for mention in entity]

        sents = []
        rels = sample["labels"]

        # Preprocess sents
        sent_map = []
        tokens = [token for sent in sample["sents"] for token in sent]
        sent_map = {}
        for i_s, token in enumerate(tokens):
            tokens_wordpiece = tokenizer.tokenize(token)
            sent_map[i_s] = len(sents)
            sents.extend(tokens_wordpiece)
        sent_map[i_s + 1] = len(sents)

        # Preprocess entities
        entity_pos = []

        for mention in mentions:
            sent_id = mention["sent_id"]
            pos = mention["pos"]
            if sent_id == 0:
                offset = 0
            else:
                offset = sum([len(sample["sents"][i]) for i in range(sent_id)])
            start = sent_map[pos[0] + offset]
            end = sent_map[pos[1] + offset]
            entity_pos.append(
                (
                    start,
                    end,
                    mention["entity_id"],
                    mention["type"],
                    mention["name"],
                    mention["sent_id"],
                )
            )

        entity_pos = list(set(entity_pos))
        entity_pos.sort()
        # Check for duplicates in entity_pos
        cleaned_entity_pos = []
        type_specific_entity_pos = {}
        for type in types:
            type_specific_entity_pos[type] = []
        for ent in entity_pos:
            cleaned_entity_pos_only_pos = [x[:2] for x in cleaned_entity_pos]
            if ent[:2] not in cleaned_entity_pos_only_pos:
                cleaned_entity_pos.append(ent)
                type_specific_entity_pos[ent[3]].append(ent[:2])
        entity_pos = sorted(cleaned_entity_pos)

        # Preprocess coreferences
        corefs = {}
        for i in range(len(entity_pos) - 1):
            for j in range(i + 1, len(entity_pos)):
                ent1 = i
                ent2 = j
                if (ent1, ent2) not in corefs.keys():
                    if entity_pos[i][2] == entity_pos[j][2]:
                        corefs[(ent1, ent2)] = [0, 1]
                    else:
                        corefs[(ent1, ent2)] = [1, 0]
        hts, coreference_labels = [], []
        for key, value in corefs.items():
            hts.append(key)
            coreference_labels.append(value)
        assert len(hts) == len(coreference_labels)
        # Preprocess entity types and relations

        entity_pos_grouped = [[] for _ in range(len(vertex_set))]
        for pos in entity_pos:
            entity_pos_grouped[pos[2]].append(pos)

        ## Preprocess entity types
        ent_map = {}
        effective_ent_count = 0
        for i, entity in enumerate(entity_pos_grouped):
            ent_map[i] = effective_ent_count
            if entity:
                effective_ent_count += 1

        for i in range(effective_ent_count):
            assert i in ent_map.values()

        ## Preprocess relations

        ### Sample positive relations
        entity_centric_hts = []
        relation_labels = []
        for label in rels:
            if entity_pos_grouped[label["h"]] and entity_pos_grouped[label["t"]]:
                entity_centric_hts.append((ent_map[label["h"]], ent_map[label["t"]]))
                relation_labels.append(label["r"])

        ### Sample negative relations
        for i in range(effective_ent_count):
            for j in range(effective_ent_count):
                if i != j:
                    if (i, j) not in entity_centric_hts:
                        entity_centric_hts.append((i, j))
                        relation_labels.append("Na")

        sorted_entity_centric_hts = sorted(entity_centric_hts)
        sorted_relation_labels = [
            relation_labels[entity_centric_hts.index(ht)]
            for ht in sorted_entity_centric_hts
        ]
        entity_centric_hts = sorted_entity_centric_hts
        relation_labels = sorted_relation_labels
        if entity_centric_hts:
            # Tokenize input
            # sents = sents[:max_seq_length - 2]
            input_ids = tokenizer.convert_tokens_to_ids(sents)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

            # List all possible spans
            span_idx = []
            for i in range(len(input_ids)):
                span_idx.extend([(i, i + j) for j in range(max_span_width)])

            type_span_labels = {}
            for type in types:
                type_span_labels[type] = []
            for key, type_entity_pos in type_specific_entity_pos.items():
                type_span_labels[key] = []
                for span in span_idx:
                    if span in type_entity_pos:
                        type_span_labels[key].append(1)
                    else:
                        type_span_labels[key].append(0)
            for key in type_span_labels.keys():
                type_specific_span_labels[key].append(type_span_labels[key])
        total_relation_labels.extend(relation_labels)
    return type_specific_span_labels, total_relation_labels


def pad_labels(labels):
    max_length = max(len(span_labels) for span_labels in labels)
    padded_labels = []
    for span_labels in labels:
        padded_labels.append(span_labels + [0] * (max_length - len(span_labels)))
    return padded_labels


def compute_type_specific_span_accuracy(y_true, y_pred):
    correct = 0
    total = 0
    for i in range(len(y_true)):
        if y_true[i] == 1:
            total += 1
            if y_pred[i] == 1:
                correct += 1
    return correct / total


if __name__ == "__main__":
    # Load the data
    classes2id = ["ent2id.json", "rel2id.json"]
    tasks = ["entity_typing", "relation_extraction"]
    for c2id, task in zip(classes2id, tasks):
        labels = load_json(os.path.join(sys.argv[1], f"{task}.json"))
        y_true = labels["gt"]
        y_pred = labels["predictions"]
        c2id = load_json(os.path.join(sys.argv[2], c2id))
        id2classes = {v: k for k, v in c2id.items()}
        ax = plot_confusion_matrix(
            y_true,
            y_pred,
            classes=list(id2classes.values()),
            normalize=True,
        )
        os.makedirs("confusion_matrix", exist_ok=True)

        plt.savefig(os.path.join("confusion_matrix", f"{"".join(sys.argv[1].split('/')[1:]).split(".")[0]}_confusion_matrix.png"))
        # plt.show()
        plt.close()

    # Recognition of INTERACTION spans
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    test_data = load_json(os.path.join(sys.argv[2], "test.json"))
    types = load_json(os.path.join(sys.argv[2], "ent2id.json")).keys()
    labels = load_json(os.path.join(sys.argv[1], "mention_detection.json"))
    y_pred = labels["predictions"]
    type_specific_gt_span_labels_dict, total_relation_labels = get_specific_type_labels(
        test_data, types, tokenizer
    )
    for type, type_specific_gt_span_labels_batches in type_specific_gt_span_labels_dict.items():
        type_specific_gt_span_labels_batches = [
            type_specific_gt_span_labels_batches[i:i + 4] for i in range(0, len(type_specific_gt_span_labels_batches), 4)
        ]
        if len(type_specific_gt_span_labels_batches[-1]) < 4:
            type_specific_gt_span_labels_batches[-1].extend([[]] * (4 - len(type_specific_gt_span_labels_batches[-1])))
        type_specific_gt_span_labels_batches = [pad_labels(labels) for labels in type_specific_gt_span_labels_batches]
        type_specific_gt_span_labels = [label for batches in type_specific_gt_span_labels_batches for batch in batches for label in batch]
        print(len(type_specific_gt_span_labels), len(y_pred))
        assert len(type_specific_gt_span_labels) == len(y_pred)
        type_span_acc = compute_type_specific_span_accuracy(type_specific_gt_span_labels, y_pred)
        print(f"Type: {type}, Span Accuracy: {type_span_acc}")
    count_relation_labels = {
        label: total_relation_labels.count(label) / len(total_relation_labels) * 100
        for label in set(total_relation_labels)
    }
    print(count_relation_labels)
