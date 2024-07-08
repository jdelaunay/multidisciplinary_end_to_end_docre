import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm
import os
import sys
import datetime

from utils.utils import load_json, collate_fn
from torch.utils.data import DataLoader

from model.model import DocJEREModel
from preprocess import read_dataset


def update_batch_metrics(precision, recall, f1, outputs, loss=None):
    precision += outputs["precision"]
    recall += outputs["recall"]
    f1 += outputs["f1"]
    if loss:
        loss += outputs["loss"].item()
        metrics = precision, recall, f1, loss
    else:
        metrics = precision, recall, f1
    return metrics


def train(model, optimizer, train_loader, val_loader, device, epochs, save_path):
    # Warmup
    total_steps = len(train_loader) * epochs

    progress_bar = tqdm(range(epochs), desc="Training")
    best_val_loss = float("inf")

    for epoch in progress_bar:
        # Training loop
        model.train()
        train_loss = 0
        md_loss, md_precision, md_recall, md_f1 = 1e-30, 0, 0, 0
        coref_loss, coref_precision, coref_recall, coref_f1 = 1e-30, 0, 0, 0
        et_loss, et_precision, et_recall, et_f1 = 1e-30, 0, 0, 0
        for batch in train_loader:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "span_idx": batch["span_idx"].to(device),
                "span_mask": batch["span_mask"].to(device),
                "span_labels": batch["span_labels"].to(device),
                "coreference_labels": batch["coreference_labels"],
                "hts": batch["hts"],
                "entity_pos": batch["entity_pos"],
                "entity_types": batch["entity_types"],
            }
            optimizer.zero_grad()
            (loss, md_outputs, coref_outputs, entity_typing_outputs) = model(**inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            md_precision, md_recall, md_f1, md_loss = update_batch_metrics(
                md_precision, md_recall, md_f1, md_outputs, loss=md_loss
            )
            coref_precision, coref_recall, coref_f1, coref_loss = update_batch_metrics(
                coref_precision, coref_recall, coref_f1, coref_outputs, loss=coref_loss
            )
            et_precision, et_recall, et_f1, et_loss = update_batch_metrics(
                et_precision, et_recall, et_f1, entity_typing_outputs, loss=et_loss
            )

        train_loss /= len(train_loader)

        train_md_metrics = [md_loss, md_precision, md_recall, md_f1]
        train_md_metrics = [item / len(train_loader) for item in train_md_metrics]
        train_coref_metrics = [coref_loss, coref_precision, coref_recall, coref_f1]
        train_coref_metrics = [item / len(train_loader) for item in train_coref_metrics]
        train_et_metrics = [et_loss, et_precision, et_recall, et_f1]
        train_et_metrics = [item / len(train_loader) for item in train_et_metrics]

        # Validation loop
        model.eval()
        md_loss, md_precision, md_recall, md_f1 = 1e-30, 0, 0, 0
        coref_loss, coref_precision, coref_recall, coref_f1 = 1e-30, 0, 0, 0
        et_loss, et_precision, et_recall, et_f1 = 1e-30, 0, 0, 0
        joint_coref_precision, joint_coref_recall, joint_coref_f1 = 0, 0, 0
        joint_et_precision, joint_et_recall, joint_et_f1 = 0, 0, 0

        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                    "span_idx": batch["span_idx"].to(device),
                    "span_mask": batch["span_mask"].to(device),
                    "span_labels": batch["span_labels"].to(device),
                    "coreference_labels": batch["coreference_labels"],
                    "hts": batch["hts"],
                    "entity_pos": batch["entity_pos"],
                    "entity_types": batch["entity_types"],
                }
                (
                    loss,
                    md_outputs,
                    coref_outputs,
                    joint_coref_outputs,
                    entity_typing_outputs,
                    joint_entity_typing_outputs,
                ) = model(**inputs, eval_mode=True)
                val_loss += loss.item()
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
                et_precision, et_recall, et_f1, et_loss = update_batch_metrics(
                    et_precision, et_recall, et_f1, entity_typing_outputs, loss=et_loss
                )
                print("joint_coref_outputs:", joint_coref_outputs)
                joint_coref_precision, joint_coref_recall, joint_coref_f1 = (
                    update_batch_metrics(
                        joint_coref_precision,
                        joint_coref_recall,
                        joint_coref_f1,
                        joint_coref_outputs,
                    )
                )
                joint_et_precision, joint_et_recall, joint_et_f1 = update_batch_metrics(
                    joint_et_precision,
                    joint_et_recall,
                    joint_et_f1,
                    joint_entity_typing_outputs,
                )
            val_loss /= len(val_loader)

            val_md_metrics = [md_loss, md_precision, md_recall, md_f1]
            val_md_metrics = [item / len(val_loader) for item in val_md_metrics]
            val_coref_metrics = [coref_loss, coref_precision, coref_recall, coref_f1]
            val_coref_metrics = [item / len(val_loader) for item in val_coref_metrics]
            val_et_metrics = [et_loss, et_precision, et_recall, et_f1]
            val_et_metrics = [item / len(val_loader) for item in val_et_metrics]
            val_joint_coref_metrics = [
                joint_coref_precision,
                joint_coref_recall,
                joint_coref_f1,
            ]
            val_joint_coref_metrics = [
                item / len(val_loader) for item in val_joint_coref_metrics
            ]
            val_joint_et_metrics = [joint_et_precision, joint_et_recall, joint_et_f1]
            val_joint_et_metrics = [
                item / len(val_loader) for item in val_joint_et_metrics
            ]
            if val_loss < best_val_loss:
                print(
                    f"Epoch {epoch + 1}: Validation loss decreased from {best_val_loss:.4f} to {val_loss:.4f}"
                )

                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                print(f"Model saved at: {save_path}")
        description = f"Epoch {epoch + 1} | Average Training Loss: {train_loss:.4f}, Average Validation Loss: {val_loss:.4f},\n\
                        ---------------------\n\
                        Train MD Loss: {train_md_metrics[0]:.4f}, Train Precision: {train_md_metrics[1]:.4f}, Train Recall: {train_md_metrics[2]:.4f}, Train F1: {train_md_metrics[3]:.4f}\n\
                        Val MD Loss: {val_md_metrics[0]:.4f}, Val Precision: {val_md_metrics[1]:.4f}, Val Recall: {val_md_metrics[2]:.4f}, Val F1: {val_md_metrics[3]:4f}\n\
                        ---------------------\n\
                        Train Coref Loss: {train_coref_metrics[0]:.4f}, Train Coref Precision: {train_coref_metrics[1]:.4f}, Train Coref Recall: {train_coref_metrics[2]:.4f}, Train Coref F1: {train_coref_metrics[3]:.4f}\n\
                        Val Coref Loss: {val_coref_metrics[0]:.4f}, Val Coref Precision: {val_coref_metrics[1]:.4f}, Val Coref Recall: {val_coref_metrics[2]:.4f}, Val Coref F1: {val_coref_metrics[3]:.4f}\n\
                        ---------------------\n\
                        Train ET Loss: {train_et_metrics[0]:.4f}, Train ET Precision: {train_et_metrics[1]:.4f}, Train ET Recall: {train_et_metrics[2]:.4f}, Train ET F1: {train_et_metrics[3]:.4f}\n\
                        Val ET Loss: {val_et_metrics[0]:.4f}, Val ET Precision: {val_et_metrics[1]:.4f}, Val ET Recall: {val_et_metrics[2]:.4f}, Val ET F1: {val_et_metrics[3]:.4f}\n\
                        ======================\n\
                        Val Joint Coref Precision: {val_joint_coref_metrics[0]:.4f}, Val Joint Coref Recall: {val_joint_coref_metrics[1]:.4f}, Val Joint Coref F1: {val_joint_coref_metrics[2]:.4f}\n\
                        Val Joint ET Precision: {val_joint_et_metrics[0]:.4f}, Val Joint ET Recall: {val_joint_et_metrics[1]:.4f}, Val Joint ET F1: {val_joint_et_metrics[2]:.4f}\n"
        progress_bar.set_description(description)


if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    config_path = sys.argv[1]
    config = load_json(config_path)

    # Add datetime to log dir
    save_path = os.path.join(
        config["log_dir"],
        f"{config['run_name']}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt",
    )
    os.makedirs(config["log_dir"], exist_ok=True)

    # Load ent2id
    ent2id = load_json(config["ent2id_path"])
    ent_num_classes = len(ent2id.keys())

    # Load encoder and tokenizer
    model_config = AutoConfig.from_pretrained(config["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModel.from_pretrained(config["model_name"], config=model_config)
    model = DocJEREModel(model, tokenizer, ent_num_classes)
    model.cuda()

    # Load datasets
    if "arpi" in config["train_path"]:
        print("Using ARPI dataset")
    else:
        print("Using DocRED dataset")
    train_dataset = read_dataset(load_json(config["train_path"]), tokenizer, ent2id)
    val_dataset = read_dataset(load_json(config["val_path"]), tokenizer, ent2id)
    # test_dataset = read_dataset(load_json(config["test_path"]), tokenizer, ent2id)

    if "sample" in config["train_path"]:
        train_dataset = train_dataset[:1]
        for i in range(150):
            train_dataset += train_dataset[:1]
        val_dataset = train_dataset[:1]

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )
    # test_loader = DataLoader(
    #     test_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn
    # )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    device = torch.device(config["device"])
    epochs = config["epochs"]
    train(model, optimizer, train_loader, val_loader, device, epochs, save_path)
