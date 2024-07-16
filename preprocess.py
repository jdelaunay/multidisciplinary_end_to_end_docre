from tqdm import tqdm
import numpy as np
import torch


def read_dataset(
    data, tokenizer, classes_to_id, rel_to_id, max_seq_length=1024, max_width=4
):
    features = []
    print("Loading dataset...")
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
            end = sent_map[pos[1] + offset] - 1
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
        entity_pos.sort(key=lambda x: (x[0], x[1]))
        # Check for duplicates in entity_pos
        posi = [x[:2] for x in entity_pos]
        duplicates = [pos for pos in posi if posi.count(pos) > 1]
        if duplicates:
            print("Duplicates found in entity_pos:", duplicates)
            duplicate_indices = [
                i for i, x in enumerate(entity_pos) if x[:2] in duplicates
            ]
            ent_dupli = [entity_pos[i] for i in duplicate_indices]
            entity_pos = [x for x in entity_pos if x not in ent_dupli[1:]]

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
        entity_types = []

        entity_pos_grouped = [[] for _ in range(len(vertex_set))]
        for pos in entity_pos:
            entity_pos_grouped[pos[2]].append(pos)

        ## Preprocess entity types
        ent_map = {}
        effective_ent_count = 0
        for i, entity in enumerate(entity_pos_grouped):
            ent_map[i] = effective_ent_count
            if entity:
                types = [mention[3] for mention in entity]
                ent_type = max(set(types), key=types.count)
                entity_types.append(get_ent_class(ent_type, classes_to_id))
                effective_ent_count += 1

        for i in range(effective_ent_count):
            assert i in ent_map.values()
        # print("ENT MAP", ent_map)

        ## Preprocess relations

        ### Sample positive relations
        entity_centric_hts = []
        relation_labels = []
        for label in rels:
            if entity_pos_grouped[label["h"]] and entity_pos_grouped[label["t"]]:
                entity_centric_hts.append((ent_map[label["h"]], ent_map[label["t"]]))
                relation_labels.append(get_ent_class(label["r"], rel_to_id))

        ### Sample negative relations
        for i in range(effective_ent_count):
            for j in range(effective_ent_count):
                if i != j:
                    if (i, j) not in entity_centric_hts:
                        entity_centric_hts.append((i, j))
                        relation_labels.append([1] + [0] * (len(rel_to_id) - 1))

        for i in range(effective_ent_count):
            for j in range(effective_ent_count):
                if i != j:
                    assert (i, j) in entity_centric_hts

        # Clean entity_pos
        entity_pos = [(x[0], x[1]) for x in entity_pos]

        # Clean entity clusters
        entity_clusters = [
            [(x[0], x[1]) for x in entity] for entity in entity_pos_grouped if entity
        ]
        assert len(entity_clusters) == effective_ent_count
        assert len(entity_types) == effective_ent_count

        # Tokenize input
        # sents = sents[:max_seq_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        # List all possible spans
        span_idx = []
        for i in range(len(input_ids)):
            # width = min(max_width, len(input_ids) - i)
            span_idx.extend([(i, i + j) for j in range(max_width)])

        span_labels = []
        for span in span_idx:
            if span in entity_pos:
                span_labels.append(1)
            else:
                span_labels.append(0)
        span_idx = torch.tensor(span_idx, dtype=torch.long)
        span_labels = torch.tensor(span_labels, dtype=torch.long)

        feature = {
            "input_ids": input_ids,
            "tokens": tokens,
            "span_idx": span_idx,
            "span_labels": span_labels,
            "hts": hts,
            "coreference_labels": coreference_labels,
            "entity_clusters": entity_clusters,
            "entity_pos": entity_pos,
            "entity_types": entity_types,
            "entity_centric_hts": entity_centric_hts,
            "relation_labels": relation_labels,
        }
        features.append(feature)
    print("Dataset loaded.")
    print("Number of samples:", len(features))
    return features


def get_ent_class(type, classes_to_id):
    id = classes_to_id[type]
    one_hot_id = [0] * len(classes_to_id)
    one_hot_id[id] = 1
    return one_hot_id
