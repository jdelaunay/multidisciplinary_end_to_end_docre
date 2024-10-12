import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm
import os
import sys
from time import time
from torchinfo import summary

from utils.utils import load_json, collate_fn
from torch.utils.data import DataLoader

from model.model import DocJEREModel
from preprocess import read_dataset

import json


def update_batch_metrics(
    precision, recall, f1, outputs, loss=None, b3=False, pairwise=False
):
    if pairwise:
        precision += outputs["pair_precision"]
        recall += outputs["pair_recall"]
        f1 += outputs["pair_f1"]
    elif b3:
        precision += outputs["b3_precision"]
        recall += outputs["b3_recall"]
        f1 += outputs["b3_f1"]
    else:
        precision += outputs["precision"]
        recall += outputs["recall"]
        f1 += outputs["f1"]
    if loss:
        loss += outputs["loss"].item()
        metrics = precision, recall, f1, loss
    else:
        metrics = precision, recall, f1
    return metrics


def test(model, test_loader, device, save_path, dataset_name="docred", coeff_md=1, coeff_coref=1, coeff_et=1, coeff_re=1):

    # Test loop
    model.load_state_dict(torch.load(save_path))
    print(f"Model loaded from: {save_path}")
    print("COREF THRESHOLD", model.coreference_resolution.threshold)
    model.eval()
    md_loss, md_precision, md_recall, md_f1 = 1e-30, 0, 0, 0
    coref_loss, coref_precision, coref_recall, coref_f1 = 1e-30, 0, 0, 0
    b3_precision, b3_recall, b3_f1 = 0, 0, 0
    pair_precision, pair_recall, pair_f1 = 0, 0, 0
    et_loss, et_precision, et_recall, et_f1 = 1e-30, 0, 0, 0
    re_loss, re_precision, re_recall, re_f1 = 1e-30, 0, 0, 0
    joint_coref_precision, joint_coref_recall, joint_coref_f1 = 0, 0, 0
    joint_b3_precision, joint_b3_recall, joint_b3_f1 = 0, 0, 0
    joint_pair_precision, joint_pair_recall, joint_pair_f1 = 0, 0, 0
    joint_et_precision, joint_et_recall, joint_et_f1 = 0, 0, 0
    joint_re_precision, joint_re_recall, joint_re_f1 = 0, 0, 0
    test_loss = 0
    inference_time = 0
    n_examples = len(test_loader.dataset)

    md_predictions, coref_predictions, et_predictions, re_predictions = [], [], [], []
    md_gt, coref_gt, et_gt, re_gt = [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            t0 = time()
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "span_idx": batch["span_idx"].to(device),
                "span_mask": batch["span_mask"].to(device),
                "span_labels": batch["span_labels"].to(device),
                "coreference_labels": batch["coreference_labels"],
                "entity_clusters": batch["entity_clusters"],
                "hts": batch["hts"],
                "entity_pos": batch["entity_pos"],
                "entity_types": batch["entity_types"],
                "entity_centric_hts": batch["entity_centric_hts"],
                "relation_labels": batch["relation_labels"],
                "coeff_md": coeff_md,
                "coeff_coref": coeff_coref,
                "coeff_et": coeff_et,
                "coeff_re": coeff_re,
            }
            md_gt.extend([label.item() for x in batch["span_labels"] for label in x])
            coref_gt.extend(batch["entity_clusters"])
            et_gt.extend([torch.argmax(label).item() for x in batch["entity_types"] for label in x])
            re_gt.extend([np.array(label).argmax() for x in batch["relation_labels"] for label in x])
            (
                loss,
                md_outputs,
                coref_outputs,
                joint_coref_outputs,
                entity_typing_outputs,
                joint_entity_typing_outputs,
                re_outputs,
                joint_re_outputs,
            ) = model(**inputs, eval_mode=True, error_analysis=True)
            inference_time += time() - t0
            test_loss += loss.item()
            md_precision, md_recall, md_f1, md_loss = update_batch_metrics(
                md_precision, md_recall, md_f1, md_outputs, loss=md_loss
            )
            coref_precision, coref_recall, coref_f1, coref_loss = (
                update_batch_metrics(
                    coref_precision,
                    coref_recall,
                    coref_f1,
                    coref_outputs,
                    loss=coref_loss,
                )
            )
            b3_precision, b3_recall, b3_f1 = update_batch_metrics(
                b3_precision, b3_recall, b3_f1, coref_outputs, b3=True
            )
            pair_precision, pair_recall, pair_f1 = update_batch_metrics(
                pair_precision, pair_recall, pair_f1, coref_outputs, pairwise=True
            )
            et_precision, et_recall, et_f1, et_loss = update_batch_metrics(
                et_precision, et_recall, et_f1, entity_typing_outputs, loss=et_loss
            )
            re_precision, re_recall, re_f1, re_loss = update_batch_metrics(
                re_precision, re_recall, re_f1, re_outputs, loss=re_loss
            )
            joint_coref_precision, joint_coref_recall, joint_coref_f1 = (
                update_batch_metrics(
                    joint_coref_precision,
                    joint_coref_recall,
                    joint_coref_f1,
                    joint_coref_outputs,
                )
            )
            joint_b3_precision, joint_b3_recall, joint_b3_f1 = update_batch_metrics(
                joint_b3_precision,
                joint_b3_recall,
                joint_b3_f1,
                joint_coref_outputs,
                b3=True,
                pairwise=False,
            )

            joint_pair_precision, joint_pair_recall, joint_pair_f1 = (
                update_batch_metrics(
                    joint_pair_precision,
                    joint_pair_recall,
                    joint_pair_f1,
                    joint_coref_outputs,
                    pairwise=True,
                )
            )
            joint_et_precision, joint_et_recall, joint_et_f1 = update_batch_metrics(
                joint_et_precision,
                joint_et_recall,
                joint_et_f1,
                joint_entity_typing_outputs,
            )
            joint_re_precision, joint_re_recall, joint_re_f1 = update_batch_metrics(
                joint_re_precision, joint_re_recall, joint_re_f1, joint_re_outputs
            )
            if type(md_outputs["predictions"]) == list:
                md_predictions.extend(md_outputs["predictions"])
            else:
                md_predictions.extend(list(md_outputs["predictions"].cpu().numpy()))
            coref_predictions.extend(coref_outputs["predictions"])
            et_predictions.extend([label.item() for label in entity_typing_outputs["predictions"]])
            re_predictions.extend([label.item() for label in re_outputs["predictions"]])
    test_loss /= len(test_loader)

    test_md_metrics = [md_loss, md_precision, md_recall, md_f1]
    test_md_metrics = [item / len(test_loader) for item in test_md_metrics]
    test_coref_metrics = [coref_loss, coref_precision, coref_recall, coref_f1]
    test_coref_metrics = [item / len(test_loader) for item in test_coref_metrics]
    test_b3_metrics = [b3_precision, b3_recall, b3_f1]
    test_b3_metrics = [item / len(test_loader) for item in test_b3_metrics]
    test_pair_metrics = [pair_precision, pair_recall, pair_f1]
    test_pair_metrics = [item / len(test_loader) for item in test_pair_metrics]
    test_et_metrics = [et_loss, et_precision, et_recall, et_f1]
    test_et_metrics = [item / len(test_loader) for item in test_et_metrics]
    test_re_metrics = [re_loss, re_precision, re_recall, re_f1]
    test_re_metrics = [item / len(test_loader) for item in test_re_metrics]

    # Joint tasks metrics
    test_joint_coref_metrics = [
        joint_coref_precision,
        joint_coref_recall,
        joint_coref_f1,
    ]
    test_joint_coref_metrics = [
        item / len(test_loader) for item in test_joint_coref_metrics
    ]
    test_joint_b3_metrics = [joint_b3_precision, joint_b3_recall, joint_b3_f1]
    test_joint_b3_metrics = [
        item / len(test_loader) for item in test_joint_b3_metrics
    ]
    test_joint_pair_metrics = [
        joint_pair_precision,
        joint_pair_recall,
        joint_pair_f1,
    ]
    test_joint_pair_metrics = [
        item / len(test_loader) for item in test_joint_pair_metrics
    ]
    test_joint_et_metrics = [joint_et_precision, joint_et_recall, joint_et_f1]
    test_joint_et_metrics = [
        item / len(test_loader) for item in test_joint_et_metrics
    ]
    test_joint_re_metrics = [joint_re_precision, joint_re_recall, joint_re_f1]
    test_joint_re_metrics = [
        item / len(test_loader) for item in test_joint_re_metrics
    ]

    inference_time_per_example = inference_time / n_examples
    description = f"Test Loss: {test_loss:.4f}\n\
                    Total Inference Time: {inference_time:.4f}, Avg Inference Time per Example: {inference_time_per_example:.4f}\n\
                    ---------------------\n\
                    Test MD Loss: {test_md_metrics[0]:.4f}, Test Precision: {test_md_metrics[1]:.4f}, Test Recall: {test_md_metrics[2]:.4f}, Test F1: {test_md_metrics[3]:.4f}\n\
                    Test Coref Loss: {test_coref_metrics[0]:.4f}, Test Coref Precision: {test_coref_metrics[1]:.4f}, Test Coref Recall: {test_coref_metrics[2]:.4f}, Test Coref F1: {test_coref_metrics[3]:.4f}\n\
                    Test B3 Precision: {test_b3_metrics[0]:.4f}, Test B3 Recall: {test_b3_metrics[1]:.4f}, Test B3 F1: {test_b3_metrics[2]:.4f}\n\
                    Test Pair Precision: {test_pair_metrics[0]:.4f}, Test Pair Recall: {test_pair_metrics[1]:.4f}, Test Pair F1: {test_pair_metrics[2]:.4f}\n\
                    Test ET Loss: {test_et_metrics[0]:.4f}, Test ET Precision: {test_et_metrics[1]:.4f}, Test ET Recall: {test_et_metrics[2]:.4f}, Test ET F1: {test_et_metrics[3]:.4f}\n\
                    Test RE Loss: {test_re_metrics[0]:.4f}, Test RE Precision: {test_re_metrics[1]:.4f}, Test RE Recall: {test_re_metrics[2]:.4f}, Test RE F1: {test_re_metrics[3]:.4f}\n\
                    ======================\n\
                    Test Joint Coref Precision: {test_joint_coref_metrics[0]:.4f}, Test Joint Coref Recall: {test_joint_coref_metrics[1]:.4f}, Test Joint Coref F1: {test_joint_coref_metrics[2]:.4f}\n\
                    Test Joint B3 Precision: {test_joint_b3_metrics[0]:.4f}, Test Joint B3 Recall: {test_joint_b3_metrics[1]:.4f}, Test Joint B3 F1: {test_joint_b3_metrics[2]:.4f}\n\
                    Test Joint Pair Precision: {test_joint_pair_metrics[0]:.4f}, Test Joint Pair Recall: {test_joint_pair_metrics[1]:.4f}, Test Joint Pair F1: {test_joint_pair_metrics[2]:.4f}\n\
                    Test Joint ET Precision: {test_joint_et_metrics[0]:.4f}, Test Joint ET Recall: {test_joint_et_metrics[1]:.4f}, Test Joint ET F1: {test_joint_et_metrics[2]:.4f}\n\
                    Test Joint RE Precision: {test_joint_re_metrics[0]:.4f}, Test Joint RE Recall: {test_joint_re_metrics[1]:.4f}, Test Joint RE F1: {test_joint_re_metrics[2]:.4f}\n"
    print(description)
    mention_detection_results = {
        "predictions": [int(pred) for pred in md_predictions],
        "gt": [int(gt) for gt in md_gt]
    }
    coreference_resolution_results = {
        "predictions": coref_predictions,
        "gt": coref_gt
    }
    entity_typing_results = {
        "predictions": et_predictions,
        "gt": et_gt
    }
    relation_extraction_results = {
        "predictions": [int(pred) for pred in re_predictions],
        "gt": [int(gt) for gt in re_gt]
    }
    dir = "error_analysis"
    os.makedirs(os.path.join(dir, dataset_name), exist_ok=True)
    with open(os.path.join(dir, dataset_name, "mention_detection.json"), "w") as f:
        json.dump(mention_detection_results, f)
    with open(os.path.join(dir, dataset_name, "coreference_resolution.json"), "w") as f:
        json.dump(coreference_resolution_results, f)
    with open(os.path.join(dir, dataset_name, "entity_typing.json"), "w") as f:
        json.dump(entity_typing_results, f)
    with open(os.path.join(dir, dataset_name, "relation_extraction.json"), "w") as f:
        json.dump(relation_extraction_results, f)
    

if __name__ == "__main__":
    seed = 12
    torch.manual_seed(seed)
    np.random.seed(seed)

    config_path = sys.argv[1]
    config = load_json(config_path)

    # Add datetime to log dir
    save_path = os.path.join(
        config["log_dir"],
        f"{config['run_name']}.pt",
    )
    os.makedirs(config["log_dir"], exist_ok=True)

    # Load ent2id
    ent2id = load_json(config["ent2id_path"])
    ent_num_classes = len(ent2id.keys())

    # Load rel2id
    rel2id = load_json(config["rel2id_path"])
    rel_num_classes = len(rel2id.keys())

    # Load encoder and tokenizer
    model_config = AutoConfig.from_pretrained(config["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModel.from_pretrained(config["model_name"], config=model_config)
    print("Model loading...")
    model = DocJEREModel(model,
                         tokenizer,
                         ent_num_classes,
                         rel_num_classes,
                         max_span_width=config["max_span_width"],
                         max_re_height=config["max_re_height"],
                         depthwise=config["depthwise"]
                        )
    print("Model loaded")
    model.cuda()
    summary(model, depth=7)

    # Load datasets
    if "arpi" in config["train_path"]:
        print("Using ARPI dataset")
        dataset_name = "coastred"
    elif "dwie" in config["train_path"]:
        print("Using DWIE dataset")
        dataset_name = "dwie"
    else:
        print("Using DocRED dataset")
        dataset_name = "docred"

    test_dataset = read_dataset(load_json(config["test_path"]), tokenizer, ent2id, rel2id, max_span_width=config["max_span_width"])

    # Create data loaders
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn
    )
    device = torch.device(config["device"])
    test(model, test_loader, device, save_path, dataset_name=dataset_name, coeff_md=config["coeff_md"], coeff_coref=config["coeff_cr"], coeff_et=config["coeff_et"], coeff_re=config["coeff_re"])
