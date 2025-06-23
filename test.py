import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm
import os
import sys
from time import time
from torchinfo import summary

from torch.utils.data import DataLoader

from model.model import DocJEREModel
from utils.utils import (
    load_json,
    collate_fn,
    get_batch_inputs,
)
from preprocess import read_dataset
from evaluate import evaluate

import json


def test(
    model,
    test_loader,
    test_ds,
    id2ent,
    id2rel,
    test_path,
    device,
    pretrained_weights_path,
    run_name="test",
    coefficients=[1, 1, 1, 1],
):

    if sum(coefficients) == max(coefficients):
        e2e_mode = False
    else:
        e2e_mode = True

    # Test loop
    checkpoint = torch.load(pretrained_weights_path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model loaded from: {pretrained_weights_path}")
    model.eval()
    md_loss, cr_loss, et_loss, re_loss = 1e-30, 1e-30, 1e-30, 1e-30
    (
        md_preds,
        cr_preds,
        et_preds,
        re_preds,
    ) = (
        [],
        [],
        [],
        [],
    )
    test_loss = 0
    inference_time = 0
    e2e_cr_preds, e2e_et_preds, e2e_re_preds = [], [], []
    e2e_titles_cr_preds, e2e_titles_et_preds, e2e_titles_re_preds = [], [], []
    re_scores, re_topks = [], []
    e2e_re_scores, e2e_re_topks = [], []
    e2e_et_clusters = []
    e2e_entity_centric_hts = []
    md_predictions, cr_predictions, et_predictions, re_predictions = [], [], [], []
    md_gt, coref_gt, et_gt, re_gt = [], [], [], []

    with torch.no_grad():
        for batch in test_loader:
            t0 = time()
            inputs = get_batch_inputs(batch, coefficients, device)
            md_gt.extend(
                [
                    torch.argmax(label).item()
                    for x in batch["span_labels"]
                    for label in x
                ]
            )
            coref_gt.extend(batch["entity_clusters"])
            re_gt.extend(
                [
                    np.array(label).argmax()
                    for x in batch["relation_labels"]
                    for label in x
                ]
            )
            (
                loss,
                md_outputs,
                coref_outputs,
                e2e_cr_outputs,
                entity_typing_outputs,
                e2e_entity_typing_outputs,
                re_outputs,
                e2e_re_outputs,
            ) = model(**inputs, eval_mode=True, error_analysis=True)
            inference_time += time() - t0
            test_loss += loss.item()
            md_loss += md_outputs["loss"].item()
            md_preds.append(md_outputs["predictions"])
            cr_loss += coref_outputs["loss"].item()
            cr_preds.append(coref_outputs["predictions"])
            et_loss += entity_typing_outputs["loss"].item()
            et_preds.append(entity_typing_outputs["predictions"].cpu().numpy())
            re_loss += re_outputs["loss"].item()
            re_pred = re_outputs["predictions"].cpu().numpy()
            re_pred[np.isnan(re_pred)] = 0
            re_preds.append(re_pred)
            re_scores.append(re_outputs["scores"].cpu())
            re_topks.append(re_outputs["topks"].cpu())

            e2e_cr_preds.append(e2e_cr_outputs["predictions"])
            e2e_titles_cr_preds.append(e2e_cr_outputs["titles"])
            e2e_et_preds.append(e2e_entity_typing_outputs["predictions"].cpu().numpy())
            e2e_et_clusters.extend(e2e_entity_typing_outputs["entity_clusters"])
            e2e_titles_et_preds.append(e2e_entity_typing_outputs["titles"])
            e2e_re_pred = e2e_re_outputs["predictions"].cpu().numpy()
            e2e_re_pred[np.isnan(e2e_re_pred)] = 0
            e2e_re_preds.append(e2e_re_pred)
            e2e_re_scores.append(e2e_re_outputs["scores"].cpu().numpy())
            e2e_re_topks.append(e2e_re_outputs["topks"].cpu().numpy())
            e2e_titles_re_preds.append(e2e_re_outputs["titles"])
            e2e_entity_centric_hts.append(e2e_re_outputs["e2e_entity_centric_hts"])

            md_predictions.extend(md_outputs["predictions"])
            cr_predictions.extend(coref_outputs["predictions"])
            re_predictions.extend(
                [torch.argmax(label).item() for label in re_outputs["predictions"]]
            )
    (
        test_md_loss,
        test_md_metrics,
        test_cr_loss,
        test_cr_metrics,
        test_et_loss,
        test_et_metrics,
        test_re_loss,
        test_re_metrics,
    ) = evaluate(
        coefficients,
        md_loss,
        md_preds,
        cr_loss,
        cr_preds,
        et_loss,
        et_preds,
        id2ent,
        re_loss,
        re_preds,
        id2rel,
        re_scores,
        re_topks,
        test_loader,
        test_path,
    )

    if e2e_mode:
        (
            _,
            _,
            _,
            test_e2e_cr_metrics,
            _,
            test_e2e_et_metrics,
            _,
            test_e2e_re_metrics,
        ) = evaluate(
            coefficients,
            md_loss,
            md_preds,
            test_cr_loss,
            e2e_cr_preds,
            test_et_loss,
            e2e_et_preds,
            id2ent,
            test_re_loss,
            e2e_re_preds,
            id2rel,
            e2e_re_scores,
            e2e_re_topks,
            test_loader,
            test_path,
            e2e_mode=True,
            e2e_titles_cr_preds=e2e_titles_cr_preds,
            e2e_titles_et_preds=e2e_titles_et_preds,
            e2e_titles_re_preds=e2e_titles_re_preds,
            e2e_entity_clusters=e2e_et_clusters,
            e2e_entity_centric_hts=e2e_entity_centric_hts,
        )
    else:
        test_e2e_cr_metrics = [0, 0, 0]
        test_e2e_et_metrics = [0, 0, 0]
        test_e2e_re_metrics = [0, 0, 0]

    # Joint tasks metrics
    inference_time_per_example = inference_time / len(test_loader.dataset)
    description = f"Test Loss: {test_loss:.4f}\n\
                    Total Inference Time: {inference_time:.4f}, Avg Inference Time per Example: {inference_time_per_example:.4f}\n\
                    ---------------------\n\
                    Test MD Loss: {test_md_loss:.4f}, Test Precision: {test_md_metrics[0]:.4f}, Test Recall: {test_md_metrics[1]:.4f}, Test F1: {test_md_metrics[2]:.4f}\n\
                    Test Coref Loss: {test_cr_loss:.4f}, Test Coref Precision: {test_cr_metrics[0]:.4f}, Test Coref Recall: {test_cr_metrics[1]:.4f}, Test Coref F1: {test_cr_metrics[2]:.4f}\n\
                    Test ET Loss: {test_et_loss:.4f}, Test ET Precision: {test_et_metrics[0]:.4f}, Test ET Recall: {test_et_metrics[1]:.4f}, Test ET F1: {test_et_metrics[2]:.4f}\n\
                    Test RE Loss: {test_re_loss:.4f}, Test RE Precision: {test_re_metrics[0]:.4f}, Test RE Recall: {test_re_metrics[1]:.4f}, Test RE F1: {test_re_metrics[2]:.4f}\n\
                    ======================\n\
                    Test E2E Coref Precision: {test_e2e_cr_metrics[0]:.4f}, Test E2E Coref Recall: {test_e2e_cr_metrics[1]:.4f}, Test E2E Coref F1: {test_e2e_cr_metrics[2]:.4f}\n\
                    Test E2E ET Precision: {test_e2e_et_metrics[0]:.4f}, Test E2E ET Recall: {test_e2e_et_metrics[1]:.4f}, Test E2E ET F1: {test_e2e_et_metrics[2]:.4f}\n\
                    Test E2E RE Precision: {test_e2e_re_metrics[0]:.4f}, Test E2E RE Recall: {test_e2e_re_metrics[1]:.4f}, Test E2E RE F1: {test_e2e_re_metrics[2]:.4f}\n"
    print(description)

    mention_detection_results = {
        "predictions": [int(pred) for pred in md_predictions],
        "gt": [int(gt) for gt in md_gt],
    }
    coreference_resolution_results = {"predictions": cr_predictions, "gt": coref_gt}
    relation_extraction_results = {
        "predictions": [int(pred) for pred in re_predictions],
        "gt": [int(gt) for gt in re_gt],
    }

    dir = "error_analysis"
    os.makedirs(os.path.join(dir, run_name), exist_ok=True)
    with open(os.path.join(dir, run_name, "mention_classification.json"), "w") as f:
        json.dump(mention_detection_results, f)
    with open(os.path.join(dir, run_name, "coreference_resolution.json"), "w") as f:
        json.dump(coreference_resolution_results, f)
    with open(os.path.join(dir, run_name, "relation_extraction.json"), "w") as f:
        json.dump(relation_extraction_results, f)


if __name__ == "__main__":
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    config_path = sys.argv[1]
    config = load_json(config_path)

    # Add datetime to log dir
    pretrained_weights_path = os.path.join(
        config["log_dir"],
        f"{config['pretrained_weights']}",
    )
    os.makedirs(config["log_dir"], exist_ok=True)

    # Load ent2id
    ent2id = load_json(config["ent2id_path"])
    ent_num_classes = len(ent2id.keys())
    id2ent = {value: key for key, value in ent2id.items()}

    # Load rel2id
    rel2id = load_json(config["rel2id_path"])
    rel_num_classes = len(rel2id.keys())
    id2rel = {value: key for key, value in rel2id.items()}

    # Load encoder and tokenizer
    model_config = AutoConfig.from_pretrained(config["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModel.from_pretrained(config["model_name"], config=model_config)
    print("Model loading...")
    model = DocJEREModel(
        model,
        tokenizer,
        ent_num_classes,
        rel_num_classes,
        max_span_width=config["max_span_width"],
        max_re_height=config["max_re_height"],
        depthwise=config["depthwise"],
    )
    print("Model loaded")
    model.cuda()
    # summary(model, depth=7)

    # Load datasets
    if "coastred" in config["train_path"]:
        print("Using CoastRED dataset")
        dataset_name = "coastred"
    elif "redocred" in config["train_path"]:
        print("Using ReDocRED dataset")
        dataset_name = "redocred"
    elif "docred" in config["train_path"]:
        print("Using DocRED dataset")
        dataset_name = "docred"
    else:
        print("Using other dataset")
        dataset_name = "other"

    test_dataset = read_dataset(
        load_json(config["test_path"]),
        tokenizer,
        ent2id,
        rel2id,
        max_span_width=config["max_span_width"],
    )

    # Create data loaders
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )
    device = torch.device(config["device"])
    coefficients = [
        config["coeff_md"],
        config["coeff_cr"],
        config["coeff_et"],
        config["coeff_re"],
    ]
    test(
        model,
        test_loader,
        test_dataset,
        id2ent,
        id2rel,
        config["test_path"],
        device,
        pretrained_weights_path,
        run_name=config["run_name"],
        coefficients=coefficients,
    )
