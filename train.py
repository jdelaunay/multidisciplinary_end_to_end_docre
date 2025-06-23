import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm
import os
import sys
import datetime
from torchinfo import summary
import json
from time import time

from utils.utils import load_json, collate_fn, get_batch_inputs
from torch.utils.data import DataLoader

from model.model import DocJEREModel
from preprocess import read_dataset

from evaluate import evaluate


def save_model_and_optimizer(
    model, optimizer, scheduler, epoch, initial_n_epochs, save_path
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "initial_n_epochs": initial_n_epochs,
        },
        save_path,
    )
    print(f"Epoch {epoch}: Model saved at: {save_path}")


def set_best_metrics(best_strategy_metric, loss, score, epoch):
    best_strategy_metric["loss"] = loss
    best_strategy_metric["score"] = score
    best_strategy_metric["epoch"] = epoch
    return best_strategy_metric


def train(
    model,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    test_loader,
    id2ent,
    id2rel,
    val_path,
    test_path,
    device,
    epochs,
    save_path,
    pretrained_weights_path,
    coefficients=[1, 1, 1, 1],
    approx=0.01,
    run_name="",
):
    start_epoch = 0
    if pretrained_weights_path:
        print(f"Loading weights from {pretrained_weights_path}...")
        checkpoint = torch.load(pretrained_weights_path, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        epochs = checkpoint["initial_n_epochs"]
        print(f"Starting training from epoch {start_epoch + 1} to epoch {epochs}")
    # Task
    if sum(coefficients) == max(coefficients):
        e2e_mode = False
    else:
        e2e_mode = True

    progress_bar = tqdm(range(start_epoch, epochs), desc="Training")
    best_strategy_metric = {"loss": np.inf, "score": 0, "epoch": 0}
    for epoch in progress_bar:
        # Training loop
        model.train()
        train_loss = 0
        for batch in train_loader:
            inputs = get_batch_inputs(
                batch,
                coefficients,
                device,
            )
            optimizer.zero_grad()
            (loss, md_outputs, coref_outputs, entity_typing_outputs, re_outputs) = (
                model(**inputs)
            )
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation loop
        model.eval()
        md_loss, cr_loss, et_loss, re_loss = 1e-30, 1e-30, 1e-30, 1e-30
        md_preds, cr_preds, et_preds, re_preds = [], [], [], []
        e2e_cr_preds, e2e_re_preds, e2e_et_preds = [], [], []
        re_scores, re_topks = [], []
        e2e_re_scores, e2e_re_topks = [], []
        e2e_titles_cr_preds, md_false_positives = [], []
        e2e_titles_et_preds, e2e_et_clusters = [], []
        e2e_titles_re_preds, e2e_entity_centric_hts = [], []
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = get_batch_inputs(
                    batch,
                    coefficients,
                    device,
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
                ) = model(**inputs, eval_mode=True)
                val_loss += loss.item()
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
                e2e_et_preds.append(
                    e2e_entity_typing_outputs["predictions"].cpu().numpy()
                )
                e2e_titles_et_preds.append(e2e_entity_typing_outputs["titles"])
                e2e_et_clusters.extend(e2e_entity_typing_outputs["entity_clusters"])
                e2e_re_pred = e2e_re_outputs["predictions"].cpu().numpy()
                e2e_re_pred[np.isnan(e2e_re_pred)] = 0
                e2e_re_preds.append(e2e_re_pred)
                e2e_re_scores.append(e2e_re_outputs["scores"].cpu().numpy())
                e2e_re_topks.append(e2e_re_outputs["topks"].cpu().numpy())
                e2e_titles_re_preds.append(e2e_re_outputs["titles"])
                e2e_entity_centric_hts.append(e2e_re_outputs["e2e_entity_centric_hts"])
            val_loss /= len(val_loader)
            (
                val_md_loss,
                val_md_metrics,
                val_cr_loss,
                val_cr_metrics,
                val_et_loss,
                val_et_metrics,
                val_re_loss,
                val_re_metrics,
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
                val_loader,
                val_path,
            )

            if e2e_mode:
                (
                    _,
                    _,
                    _,
                    val_e2e_cr_metrics,
                    _,
                    val_e2e_et_metrics,
                    _,
                    val_e2e_re_metrics,
                ) = evaluate(
                    coefficients,
                    md_loss,
                    md_preds,
                    val_cr_loss,
                    e2e_cr_preds,
                    val_et_loss,
                    e2e_et_preds,
                    id2ent,
                    val_re_loss,
                    e2e_re_preds,
                    id2rel,
                    e2e_re_scores,
                    e2e_re_topks,
                    val_loader,
                    val_path,
                    e2e_mode=True,
                    e2e_titles_cr_preds=e2e_titles_cr_preds,
                    e2e_titles_et_preds=e2e_titles_et_preds,
                    e2e_titles_re_preds=e2e_titles_re_preds,
                    e2e_entity_clusters=e2e_et_clusters,
                    e2e_entity_centric_hts=e2e_entity_centric_hts,
                    md_false_positives=md_false_positives,
                )
            else:
                val_e2e_cr_metrics = [0, 0, 0]
                val_e2e_et_metrics = [0, 0, 0]
                val_e2e_re_metrics = [0, 0, 0]

        description = f"Epoch {epoch + 1} | Average Training Loss: {train_loss:.4f}, Average Validation Loss: {val_loss:.4f}, | LR: {scheduler.get_last_lr()}\n\
                        ---------------------\n\
                        Val MD Loss: {val_md_loss:.4f}, Val Precision: {val_md_metrics[0]:.4f}, Val Recall: {val_md_metrics[1]:.4f}, Val F1: {val_md_metrics[2]:4f}\n\
                        ---------------------\n\
                        Val Coref Loss: {val_cr_loss:.4f}, Val Coref Precision: {val_cr_metrics[0]:.4f}, Val Coref Recall: {val_cr_metrics[1]:.4f}, Val Coref F1: {val_cr_metrics[2]:.4f}\n\
                        ---------------------\n\
                        Val ET Loss: {val_et_loss:.4f}, Val ET Precision: {val_et_metrics[0]:.4f}, Val ET Recall: {val_et_metrics[1]:.4f}, Val ET F1: {val_et_metrics[2]:.4f}\n\
                        ---------------------\n\
                        Val RE Loss: {val_re_loss:.4f}, Val RE Precision: {val_re_metrics[0]:.4f}, Val RE Recall: {val_re_metrics[1]:.4f}, Val RE F1: {val_re_metrics[2]:.4f}\n\
                        ======================\n\
                        Val E2E Coref Precision: {val_e2e_cr_metrics[0]:.4f}, Val E2E Coref Recall: {val_e2e_cr_metrics[1]:.4f}, Val E2E Coref F1: {val_e2e_cr_metrics[2]:.4f}\n\
                        Val E2E ET Precision: {val_e2e_et_metrics[0]:.4f}, Val E2E ET Recall: {val_e2e_et_metrics[1]:.4f}, Val E2E ET F1: {val_e2e_et_metrics[2]:.4f}\n\
                        Val E2E RE Precision: {val_e2e_re_metrics[0]:.4f}, Val E2E RE Recall: {val_e2e_re_metrics[1]:.4f}, Val E2E RE F1: {val_e2e_re_metrics[2]:.4f}\n"

        progress_bar.set_description(description)

        loss = val_loss
        if all(coeff > 0 for coeff in coefficients):
            score = val_e2e_re_metrics[2]
            scheduler.step(score)

            if score >= (best_strategy_metric["score"] + approx):
                best_strategy_metric = set_best_metrics(
                    best_strategy_metric, loss, score, epoch
                )
                save_model_and_optimizer(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    epochs,
                    save_path,
                )
        else:
            # MD only
            if coefficients[0] > 0 and all(coeff == 0 for coeff in coefficients[1:]):
                score = val_md_metrics[2]
            # CR only
            elif coefficients[1] > 0 and coefficients[0] == coefficients[2] == 0:
                score = val_cr_metrics[2]
            # ET only
            elif coefficients[2] > 0 and all(
                coeff == 0
                for coeff in [coefficients[0], coefficients[1], coefficients[-1]]
            ):
                score = val_et_metrics[2]
            # RE only
            elif coefficients[3] > 0 and all(coeff == 0 for coeff in coefficients[:-1]):
                score = val_re_metrics[2]

            if score > (best_strategy_metric["score"] + approx):
                best_strategy_metric = set_best_metrics(
                    best_strategy_metric, loss, score, epoch
                )
                save_model_and_optimizer(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    epochs,
                    save_path,
                )
            else:
                scheduler.step(score)

    # Test loop
    checkpoint = torch.load(save_path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model loaded from: {save_path}")
    print("COREF THRESHOLD", model.coreference_resolution.threshold)
    model.eval()
    test_loss = 0
    inference_time = 0
    n_examples = len(test_loader.dataset)

    md_loss, cr_loss, et_loss, re_loss = 1e-30, 1e-30, 1e-30, 1e-30
    md_preds, cr_preds, et_preds, re_preds = [], [], [], []
    e2e_cr_preds, e2e_re_preds, e2e_et_preds = [], [], []
    re_scores, re_topks = [], []
    e2e_re_scores, e2e_re_topks = [], []
    e2e_titles_cr_preds, md_false_positives = [], []
    e2e_titles_et_preds, e2e_et_clusters = [], []
    e2e_titles_re_preds, e2e_entity_centric_hts = [], []

    with torch.no_grad():
        for batch in test_loader:
            t0 = time()
            inputs = get_batch_inputs(
                batch,
                coefficients,
                device,
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

    test_loss /= len(test_loader)

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
            md_false_positives=md_false_positives,
        )
    else:
        test_e2e_cr_metrics = [0, 0, 0]
        test_e2e_et_metrics = [0, 0, 0]
        test_e2e_re_metrics = [0, 0, 0]

    inference_time_per_example = inference_time / n_examples
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


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True

    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    config_path = sys.argv[1]
    config = load_json(config_path)
    print("Run name:", config["run_name"])
    print("Configurations:")
    print(config)

    run_name = (
        f"{config['run_name']}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    # Load pretrained weights if resuming
    if config["pretrained_weights"]:
        pretrained_weights_path = os.path.join(
            config["log_dir"],
            config["pretrained_weights"],
        )
    else:
        pretrained_weights_path = None

    # Add datetime to log dir
    save_path = os.path.join(
        config["log_dir"],
        f"{run_name}.pt",
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
    if "arpi" in config["train_path"]:
        print("Using ARPI dataset")
    elif "dwie" in config["train_path"]:
        print("Using DWIE dataset")
    else:
        print("Using DocRED dataset")
    train_dataset = read_dataset(
        load_json(config["train_path"]),
        tokenizer,
        ent2id,
        rel2id,
        max_span_width=config["max_span_width"],
    )
    val_dataset = read_dataset(
        load_json(config["val_path"]),
        tokenizer,
        ent2id,
        rel2id,
        max_span_width=config["max_span_width"],
    )
    test_dataset = read_dataset(
        load_json(config["test_path"]),
        tokenizer,
        ent2id,
        rel2id,
        max_span_width=config["max_span_width"],
    )

    if config["test_one_sample_only"]:
        train_dataset = train_dataset[:1]
        for i in range(1000):
            train_dataset += train_dataset[:1]
        val_dataset = train_dataset[:2]
        test_dataset = train_dataset[:2]

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )
    param_groups = [
        {
            "params": model.encoder.parameters(),
            "lr": config["encoder_learning_rate"],
            "amsgrad": True,
            "name": "encoder",
        },
        {
            "params": model.mention_detection.parameters(),
            "lr": config["md_learning_rate"],
            "amsgrad": True,
            "name": "mention_detection",
        },
        {
            "params": model.coreference_resolution.parameters(),
            "lr": config["cr_learning_rate"],
            "amsgrad": True,
            "name": "coreference_resolution",
        },
        {
            "params": model.entity_typing.parameters(),
            "lr": config["et_learning_rate"],
            "amsgrad": True,
            "name": "entity_typing",
        },
        {
            "params": model.rel_classifier.parameters(),
            "lr": config["re_learning_rate"],
            "amsgrad": True,
            "name": "relation_extraction",
        },
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=config["other_learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=config["patience"], verbose=True
    )
    device = torch.device(config["device"])
    epochs = config["epochs"]
    train(
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        test_loader,
        id2ent,
        id2rel,
        config["val_path"],
        config["test_path"],
        device,
        epochs,
        save_path,
        pretrained_weights_path,
        coefficients=[
            config["coeff_md"],
            config["coeff_cr"],
            config["coeff_et"],
            config["coeff_re"],
        ],
        run_name=run_name,
    )
