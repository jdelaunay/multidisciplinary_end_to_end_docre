import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

import sys
import os
from nltk.stem import WordNetLemmatizer


def load_json(file_path):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    return data


def get_generic_stats(docred_data):
    total_mentions, total_relations, types, relation_types = [], [], [], []
    total_mentions_wo_interactions = []
    num_data = len(docred_data)
    max_length = 0
    nb_sents_without_mentions = 0
    # Iterate through each data point
    for data in docred_data:
        mentions = [mention for entity in data["vertexSet"] for mention in entity]
        types += [mention["type"] for mention in mentions]
        types = list(set(types))

        nb_sents_without_mentions += len(data["sents"]) - len(
            set([mention["sent_id"] for mention in mentions])
        )

        # Mentions
        total_mentions.append(len(mentions))
        mentions_wo_interactions = [
            mention
            for mention in mentions
            if (mention["type"] != "INTERACTION") and (mention["type"] != "COMPARISON")
        ]
        total_mentions_wo_interactions.append(len(mentions_wo_interactions))

        # Relations
        total_relations.append(len(data["labels"]))
        relation_types += [label["r"] for label in data["labels"]]
        relation_types = list(set(relation_types))

        # Lengths
        local_max_length = max(
            [mention["pos"][1] - mention["pos"][0] for mention in mentions]
        )
        if local_max_length > max_length:
            max_length = local_max_length

    n_entities = [len(data["vertexSet"]) for data in docred_data]
    total_mentions.sort()
    tokens = [tok for data in docred_data for sent in data["sents"] for tok in sent]
    sents = [sent for data in docred_data for sent in data["sents"]]
    vocab = set(tokens)
    vocab_size = len(vocab)
    type_token_ratio = vocab_size / len(tokens)
    print("========== Generic Stats ==========")
    print("Number of data", num_data)
    print("Number of tokens:", len(tokens))
    print("Number of sentences:", len(sents))
    print("Number of sentences without mentions:", nb_sents_without_mentions)
    print("Average number of sentences per data:", len(sents) / num_data)
    print("Vocabulary size:", vocab_size)
    print("Type to Token Ratio:", type_token_ratio)
    print("Number of mentions:", sum(total_mentions))
    print(
        "Number of mentions without interactions:", sum(total_mentions_wo_interactions)
    )
    print("Number of entities:", sum(n_entities))
    print("Number of entity types:", len(types))
    print("Number of relations:", sum(total_relations))
    print("Number of relation types:", len(relation_types))
    print("-----------------------------------")


def get_types_proportions(docred_data):
    types_count = {}
    n_entities = [len(data["vertexSet"]) for data in docred_data]
    for data in docred_data:
        for entity in data["vertexSet"]:
            types = [mention["type"] for mention in entity]
            most_common_type = max(set(types), key=types.count)
            if most_common_type in types_count:
                types_count[most_common_type] += 1
            else:
                types_count[most_common_type] = 1
    print("========== Entity Types Proportions ==========")
    for key in types_count:
        print(f"Proportion of {key} entities:", types_count[key] / sum(n_entities))
    print("-----------------------------------")


def get_relation_labels_proportions(docred_data):
    relation_labels = {}
    total_relations = [len(data["labels"]) for data in docred_data]
    for data in docred_data:
        for label in data["labels"]:
            if label["r"] in relation_labels:
                relation_labels[label["r"]] += 1
            else:
                relation_labels[label["r"]] = 1
    print("========== Relation Labels Proportions ==========")
    for key in relation_labels:
        print(
            f"Proportion of {key} relations:",
            relation_labels[key] / sum(total_relations),
        )
    print("-----------------------------------")


def get_nested_mentions_count(docred_data):
    nested_mentions = 0
    total_mentions = [len(data["vertexSet"]) for data in docred_data]
    for data in docred_data:
        data_mentions = [mention for entity in data["vertexSet"] for mention in entity]
        for mention in data_mentions:
            same_sent_mentions = [
                m for m in data_mentions if m["sent_id"] == mention["sent_id"]
            ]
            same_sent_mentions.sort(
                key=lambda x: x["pos"][1] - x["pos"][0], reverse=True
            )
            for m in same_sent_mentions:
                if (
                    mention["pos"][0] <= m["pos"][0] and mention["pos"][1] > m["pos"][1]
                ) or (
                    mention["pos"][0] < m["pos"][0] and mention["pos"][1] >= m["pos"][1]
                ):
                    nested_mentions += 1
                    break
    print("========== Nested Mentions ==========")
    print("Number of nested mentions:", nested_mentions)
    print("Proportion of nested mentions:", nested_mentions / sum(total_mentions))
    print("-----------------------------------")


def get_unique_terms_count(docred_data):
    unique_terms = set()
    all_terms = []
    for data in docred_data:
        for entity in data["vertexSet"]:
            for mention in entity:
                if mention["type"] not in ["INTERACTION", "COMPARISON"]:
                    unique_terms.add(mention["name"])
                    all_terms.append(mention["name"])
    print("Number of unique terms:", len(unique_terms))
    lemmatizer = WordNetLemmatizer()
    lemmatized_unique_terms = set(
        [
            ".".join([lemmatizer.lemmatize(word) for word in term.split(" ")])
            for term in unique_terms
        ]
    )
    terms_to_remove = set()
    for term in lemmatized_unique_terms:
        if term.isupper() and term + "s" in lemmatized_unique_terms:
            terms_to_remove.add(term + "s")
    lemmatized_unique_terms -= terms_to_remove
    print("Number of lemmatized unique terms:", len(lemmatized_unique_terms))
    print("Number of all terms:", len(all_terms))


def get_same_name_in_entity(docred_data):
    same_cluster_mentions = 0
    total_mentions = 0
    for data in docred_data:
        for entity in data["vertexSet"]:
            mention_names = [mention["name"] for mention in entity]
            total_mentions += len(mention_names)
            same_cluster_mentions += sum(
                mention_names.count(name) > 1 for name in set(mention_names)
            )

    proportion_same_cluster = same_cluster_mentions / total_mentions
    print(
        "Proportion of entity mentions in the same cluster and equal to another mention:",
        proportion_same_cluster,
    )


if __name__ == "__main__":
    path = sys.argv[1]
    train_data = load_json(os.path.join(path, "train_annotated.json"))
    dev_data = load_json(os.path.join(path, "dev.json"))
    test_data = load_json(os.path.join(path, "test.json"))
    docred_data = train_data + dev_data + test_data
    get_generic_stats(docred_data)
    get_types_proportions(docred_data)
    get_relation_labels_proportions(docred_data)
    get_nested_mentions_count(docred_data)
    get_same_name_in_entity(docred_data)
    get_unique_terms_count(docred_data)
