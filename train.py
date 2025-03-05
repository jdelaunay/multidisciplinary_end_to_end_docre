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

from utils.utils import load_json, collate_fn
from torch.utils.data import DataLoader

from model.model import DocJEREModel
from preprocess import read_dataset


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


def train(
    model,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    test_loader,
    device,
    epochs,
    save_path,
    pretrained_weights_path,
    patience=2,
    coeff_md=1,
    coeff_coref=1,
    coeff_et=1,
    coeff_re=1,
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
    # Task
    if coeff_md > 0 and coeff_coref == coeff_et == coeff_re == 0:
        task = "MD"
        strategy = "task_specific"
    elif coeff_coref > 0 and coeff_md == coeff_et == coeff_re == 0:
        task = "CR"
        strategy = "task_specific"
    elif coeff_et > 0 and coeff_md == coeff_coref == coeff_re == 0:
        task = "ET"
        strategy = "task_specific"
    elif coeff_re > 0 and coeff_md == coeff_coref == coeff_et == 0:
        task = "RE"
        strategy = "task_specific"
    else:
        task = "Joint"
        strategy = "joint"
        original_coeff_md = coeff_md
        original_coeff_coref = coeff_coref
        original_coeff_et = coeff_et
        original_coeff_re = coeff_re
        print(f"Joint training with {strategy} strategy")
        if (
            strategy == "md_first"
            or strategy == "cascadian"
            or strategy == "md_re_cr_et"
        ):
            coeff_coref = coeff_et = coeff_re = 0
        elif strategy == "re_md_cr_et":
            coeff_md = coeff_coref = coeff_et = 0
        elif strategy == "re+md_cr+et":
            coeff_et = coeff_coref = 0

    progress_bar = tqdm(range(epochs), desc="Training")
    best_metric = 0
    best_strategy_metric = {"score": 0, "epoch": 0}
    for epoch in progress_bar:
        epoch = start_epoch + epoch
        # Training loop
        model.train()
        train_loss = 0
        md_loss, md_precision, md_recall, md_f1 = 1e-30, 0, 0, 0
        coref_loss, coref_precision, coref_recall, coref_f1 = 1e-30, 0, 0, 0
        b3_precision, b3_recall, b3_f1 = 0, 0, 0
        pair_precision, pair_recall, pair_f1 = 0, 0, 0
        et_loss, et_precision, et_recall, et_f1 = 1e-30, 0, 0, 0
        re_loss, re_precision, re_recall, re_f1 = 1e-30, 0, 0, 0
        for batch in train_loader:
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
            optimizer.zero_grad()
            (loss, md_outputs, coref_outputs, entity_typing_outputs, re_outputs) = (
                model(**inputs)
            )
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            md_precision, md_recall, md_f1, md_loss = update_batch_metrics(
                md_precision, md_recall, md_f1, md_outputs, loss=md_loss
            )
            coref_precision, coref_recall, coref_f1, coref_loss = update_batch_metrics(
                coref_precision, coref_recall, coref_f1, coref_outputs, loss=coref_loss
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

        train_loss /= len(train_loader)

        train_md_metrics = [md_loss, md_precision, md_recall, md_f1]
        train_md_metrics = [item / len(train_loader) for item in train_md_metrics]
        train_coref_metrics = [coref_loss, coref_precision, coref_recall, coref_f1]
        train_coref_metrics = [item / len(train_loader) for item in train_coref_metrics]
        train_b3_metrics = [b3_precision, b3_recall, b3_f1]
        train_b3_metrics = [item / len(train_loader) for item in train_b3_metrics]
        train_pair_metrics = [pair_precision, pair_recall, pair_f1]
        train_pair_metrics = [item / len(train_loader) for item in train_pair_metrics]
        train_et_metrics = [et_loss, et_precision, et_recall, et_f1]
        train_et_metrics = [item / len(train_loader) for item in train_et_metrics]
        train_re_metrics = [re_loss, re_precision, re_recall, re_f1]
        train_re_metrics = [item / len(train_loader) for item in train_re_metrics]

        # Validation loop
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
                (
                    loss,
                    md_outputs,
                    coref_outputs,
                    joint_coref_outputs,
                    entity_typing_outputs,
                    joint_entity_typing_outputs,
                    re_outputs,
                    joint_re_outputs,
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
            val_loss /= len(val_loader)

            val_md_metrics = [md_loss, md_precision, md_recall, md_f1]
            val_md_metrics = [item / len(val_loader) for item in val_md_metrics]
            val_coref_metrics = [coref_loss, coref_precision, coref_recall, coref_f1]
            val_coref_metrics = [item / len(val_loader) for item in val_coref_metrics]
            val_b3_metrics = [b3_precision, b3_recall, b3_f1]
            val_b3_metrics = [item / len(val_loader) for item in val_b3_metrics]
            val_pair_metrics = [pair_precision, pair_recall, pair_f1]
            val_pair_metrics = [item / len(val_loader) for item in val_pair_metrics]
            val_et_metrics = [et_loss, et_precision, et_recall, et_f1]
            val_et_metrics = [item / len(val_loader) for item in val_et_metrics]
            val_re_metrics = [re_loss, re_precision, re_recall, re_f1]
            val_re_metrics = [item / len(val_loader) for item in val_re_metrics]

            # Joint tasks metrics
            val_joint_coref_metrics = [
                joint_coref_precision,
                joint_coref_recall,
                joint_coref_f1,
            ]
            val_joint_coref_metrics = [
                item / len(val_loader) for item in val_joint_coref_metrics
            ]
            val_joint_b3_metrics = [joint_b3_precision, joint_b3_recall, joint_b3_f1]
            val_joint_b3_metrics = [
                item / len(val_loader) for item in val_joint_b3_metrics
            ]
            val_joint_pair_metrics = [
                joint_pair_precision,
                joint_pair_recall,
                joint_pair_f1,
            ]
            val_joint_pair_metrics = [
                item / len(val_loader) for item in val_joint_pair_metrics
            ]
            val_joint_et_metrics = [joint_et_precision, joint_et_recall, joint_et_f1]
            val_joint_et_metrics = [
                item / len(val_loader) for item in val_joint_et_metrics
            ]
            val_joint_re_metrics = [joint_re_precision, joint_re_recall, joint_re_f1]
            val_joint_re_metrics = [
                item / len(val_loader) for item in val_joint_re_metrics
            ]

            # Save best model
            if task == "MD":
                metric_to_monitor = val_md_metrics[3]
            elif task == "CR":
                metric_to_monitor = val_coref_metrics[3]
            elif task == "ET":
                metric_to_monitor = val_et_metrics[3]
            elif task == "RE":
                metric_to_monitor = val_re_metrics[3]
            else:
                metric_to_monitor = val_joint_re_metrics[2]
            if metric_to_monitor > best_metric:
                print(
                    f"Epoch {epoch + 1}: Validation F1 increased from {best_metric:.4f} to {metric_to_monitor:.4f}"
                )

                best_metric = metric_to_monitor
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                    },
                    save_path,
                )
                print(f"Model saved at: {save_path}")
        description = f"Epoch {epoch + 1} | Average Training Loss: {train_loss:.4f}, Average Validation Loss: {val_loss:.4f}, | LR: {scheduler.get_last_lr()}\n\
                        ---------------------\n\
                        Train MD Loss: {train_md_metrics[0]:.4f}, Train Precision: {train_md_metrics[1]:.4f}, Train Recall: {train_md_metrics[2]:.4f}, Train F1: {train_md_metrics[3]:.4f}\n\
                        Val MD Loss: {val_md_metrics[0]:.4f}, Val Precision: {val_md_metrics[1]:.4f}, Val Recall: {val_md_metrics[2]:.4f}, Val F1: {val_md_metrics[3]:4f}\n\
                        ---------------------\n\
                        Train Coref Loss: {train_coref_metrics[0]:.4f}, Train Coref Precision: {train_coref_metrics[1]:.4f}, Train Coref Recall: {train_coref_metrics[2]:.4f}, Train Coref F1: {train_coref_metrics[3]:.4f}\n\
                        Train B3 Precision: {train_b3_metrics[0]:.4f}, Train B3 Recall: {train_b3_metrics[1]:.4f}, Train B3 F1: {train_b3_metrics[2]:.4f}\n\
                        Train Pair Precision: {train_pair_metrics[0]:.4f}, Train Pair Recall: {train_pair_metrics[1]:.4f}, Train Pair F1: {train_pair_metrics[2]:.4f}\n\
                        Val Coref Loss: {val_coref_metrics[0]:.4f}, Val Coref Precision: {val_coref_metrics[1]:.4f}, Val Coref Recall: {val_coref_metrics[2]:.4f}, Val Coref F1: {val_coref_metrics[3]:.4f}\n\
                        Val B3 Precision: {val_b3_metrics[0]:.4f}, Val B3 Recall: {val_b3_metrics[1]:.4f}, Val B3 F1: {val_b3_metrics[2]:.4f}\n\
                        Val Pair Precision: {val_pair_metrics[0]:.4f}, Val Pair Recall: {val_pair_metrics[1]:.4f}, Val Pair F1: {val_pair_metrics[2]:.4f}\n\
                        ---------------------\n\
                        Train ET Loss: {train_et_metrics[0]:.4f}, Train ET Precision: {train_et_metrics[1]:.4f}, Train ET Recall: {train_et_metrics[2]:.4f}, Train ET F1: {train_et_metrics[3]:.4f}\n\
                        Val ET Loss: {val_et_metrics[0]:.4f}, Val ET Precision: {val_et_metrics[1]:.4f}, Val ET Recall: {val_et_metrics[2]:.4f}, Val ET F1: {val_et_metrics[3]:.4f}\n\
                        ---------------------\n\
                        Train RE Loss: {train_re_metrics[0]:.4f}, Train RE Precision: {train_re_metrics[1]:.4f}, Train RE Recall: {train_re_metrics[2]:.4f}, Train RE F1: {train_re_metrics[3]:.4f}\n\
                        Val RE Loss: {val_re_metrics[0]:.4f}, Val RE Precision: {val_re_metrics[1]:.4f}, Val RE Recall: {val_re_metrics[2]:.4f}, Val RE F1: {val_re_metrics[3]:.4f}\n\
                        ======================\n\
                        Val Joint Coref Precision: {val_joint_coref_metrics[0]:.4f}, Val Joint Coref Recall: {val_joint_coref_metrics[1]:.4f}, Val Joint Coref F1: {val_joint_coref_metrics[2]:.4f}\n\
                        Val Joint B3 Precision: {val_joint_b3_metrics[0]:.4f}, Val Joint B3 Recall: {val_joint_b3_metrics[1]:.4f}, Val Joint B3 F1: {val_joint_b3_metrics[2]:.4f}\n\
                        Val Joint Pair Precision: {val_joint_pair_metrics[0]:.4f}, Val Joint Pair Recall: {val_joint_pair_metrics[1]:.4f}, Val Joint Pair F1: {val_joint_pair_metrics[2]:.4f}\n\
                        Val Joint ET Precision: {val_joint_et_metrics[0]:.4f}, Val Joint ET Recall: {val_joint_et_metrics[1]:.4f}, Val Joint ET F1: {val_joint_et_metrics[2]:.4f}\n\
                        Val Joint RE Precision: {val_joint_re_metrics[0]:.4f}, Val Joint RE Recall: {val_joint_re_metrics[1]:.4f}, Val Joint RE F1: {val_joint_re_metrics[2]:.4f}\n"

        progress_bar.set_description(description)

        if strategy == "task_specific" or (
            coeff_md > 0 and coeff_coref > 0 and coeff_et > 0 and coeff_re > 0
        ):
            scheduler.step(metric_to_monitor)
        elif coeff_md > 0 and coeff_coref == coeff_et == coeff_re == 0:
            if val_md_metrics[3] > best_strategy_metric["score"] + approx:
                best_strategy_metric["score"] = val_md_metrics[3]
                best_strategy_metric["epoch"] = epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                    },
                    save_path,
                )
                print(f"Model saved at: {save_path}")
            elif (
                epoch - best_strategy_metric["epoch"] > patience
                and strategy == "cascadian"
            ):
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                    },
                    save_path,
                )
                best_strategy_metric["score"] = 0
                best_strategy_metric["epoch"] = epoch
                coeff_coref = original_coeff_coref
            elif (
                epoch - best_strategy_metric["epoch"] > patience
                and strategy == "md_first"
            ):
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                    },
                    save_path,
                )
                best_strategy_metric["score"] = 0
                best_strategy_metric["epoch"] = epoch
                coeff_coref = original_coeff_coref
                coeff_et = original_coeff_et
                coeff_re = original_coeff_re
            elif (
                epoch - best_strategy_metric["epoch"] > patience
                and strategy == "md_re_cr_et"
            ):
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                    },
                    save_path,
                )
                best_strategy_metric["score"] = 0
                best_strategy_metric["epoch"] = epoch
                coeff_re = original_coeff_re
        elif coeff_md > 0 and coeff_coref > 0 and coeff_et == coeff_re == 0:
            if val_joint_coref_metrics[2] > best_strategy_metric["score"] + approx:
                best_strategy_metric["score"] = val_joint_coref_metrics[2]
                best_strategy_metric["epoch"] = epoch
                torch.save(model.state_dict(), save_path)
                print(f"Model saved at: {save_path}")
            elif epoch - best_strategy_metric["epoch"] > patience:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                    },
                    save_path,
                )
                best_strategy_metric["score"] = 0
                best_strategy_metric["epoch"] = epoch
                coeff_et = original_coeff_et
        elif coeff_md > 0 and coeff_coref > 0 and coeff_et > 0 and coeff_re == 0:
            if val_joint_et_metrics[2] > best_strategy_metric["score"] + approx:
                best_strategy_metric["score"] = val_joint_et_metrics[2]
                best_strategy_metric["epoch"] = epoch
                torch.save(model.state_dict(), save_path)
                print(f"Model saved at: {save_path}")
            elif epoch - best_strategy_metric["epoch"] > patience:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                    },
                    save_path,
                )
                best_strategy_metric["score"] = 0
                best_strategy_metric["epoch"] = epoch
                coeff_re = original_coeff_re

        # RE first
        elif coeff_re > 0 and coeff_md == coeff_coref == coeff_et == 0:
            if val_re_metrics[3] > best_strategy_metric["score"] + approx:
                best_strategy_metric["score"] = val_re_metrics[3]
                best_strategy_metric["epoch"] = epoch
                torch.save(model.state_dict(), save_path)
                print(f"Model saved at: {save_path}")
            elif (
                epoch - best_strategy_metric["epoch"] > patience
                and strategy == "re_md_cr_et"
            ):
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                    },
                    save_path,
                )
                best_strategy_metric["score"] = 0
                best_strategy_metric["epoch"] = epoch
                coeff_md = original_coeff_md
        elif coeff_re > 0 and coeff_md > 0 and coeff_coref == coeff_et == 0:
            if (
                val_md_metrics[3] + val_re_metrics[3]
                > best_strategy_metric["score"] + 2 * approx
            ):
                best_strategy_metric["score"] = val_md_metrics[3] + val_re_metrics[3]
                best_strategy_metric["epoch"] = epoch
                torch.save(model.state_dict(), save_path)
                print(f"Model saved at: {save_path}")
            elif (
                epoch - best_strategy_metric["epoch"] > patience
                and strategy == "re_md_cr_et"
            ):
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                    },
                    save_path,
                )
                best_strategy_metric["score"] = 0
                best_strategy_metric["epoch"] = epoch
                coeff_coref = original_coeff_coref
            elif (
                epoch - best_strategy_metric["epoch"] > patience
                and strategy == "re+md_cr+et"
            ):
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                    },
                    save_path,
                )
                best_strategy_metric["score"] = 0
                best_strategy_metric["epoch"] = epoch
                coeff_coref = original_coeff_coref
                coeff_et = original_coeff_et
            elif (
                epoch - best_strategy_metric["epoch"] > patience
                and strategy == "md_re_cr_et"
            ):
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                    },
                    save_path,
                )
                best_strategy_metric["score"] = 0
                best_strategy_metric["epoch"] = epoch
                coeff_coref = original_coeff_coref

        elif coeff_re > 0 and coeff_md > 0 and coeff_coref > 0 and coeff_et == 0:
            if (
                val_md_metrics[3] + val_coref_metrics[3] + val_re_metrics[3]
                > best_strategy_metric["score"] + 3 * approx
            ):
                best_strategy_metric["score"] = (
                    val_md_metrics[3] + val_coref_metrics[3] + val_re_metrics[3]
                )
                best_strategy_metric["epoch"] = epoch
                torch.save(model.state_dict(), save_path)
                print(f"Model saved at: {save_path}")
            elif epoch - best_strategy_metric["epoch"] > patience:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                    },
                    save_path,
                )
                best_strategy_metric["score"] = 0
                best_strategy_metric["epoch"] = epoch
                coeff_et = original_coeff_et

    # Test loop
    checkpoint = torch.load(save_path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
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
            et_gt.extend(
                [
                    torch.argmax(label).item()
                    for x in batch["entity_types"]
                    for label in x
                ]
            )
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
            coref_precision, coref_recall, coref_f1, coref_loss = update_batch_metrics(
                coref_precision,
                coref_recall,
                coref_f1,
                coref_outputs,
                loss=coref_loss,
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
            et_predictions.extend(
                [label.item() for label in entity_typing_outputs["predictions"]]
            )
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
    test_joint_b3_metrics = [item / len(test_loader) for item in test_joint_b3_metrics]
    test_joint_pair_metrics = [
        joint_pair_precision,
        joint_pair_recall,
        joint_pair_f1,
    ]
    test_joint_pair_metrics = [
        item / len(test_loader) for item in test_joint_pair_metrics
    ]
    test_joint_et_metrics = [joint_et_precision, joint_et_recall, joint_et_f1]
    test_joint_et_metrics = [item / len(test_loader) for item in test_joint_et_metrics]
    test_joint_re_metrics = [joint_re_precision, joint_re_recall, joint_re_f1]
    test_joint_re_metrics = [item / len(test_loader) for item in test_joint_re_metrics]

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
        "gt": [int(gt) for gt in md_gt],
    }
    coreference_resolution_results = {"predictions": coref_predictions, "gt": coref_gt}
    entity_typing_results = {"predictions": et_predictions, "gt": et_gt}
    relation_extraction_results = {
        "predictions": [int(pred) for pred in re_predictions],
        "gt": [int(gt) for gt in re_gt],
    }
    dir = "error_analysis"
    os.makedirs(os.path.join(dir, run_name), exist_ok=True)
    with open(os.path.join(dir, run_name, "mention_detection.json"), "w") as f:
        json.dump(mention_detection_results, f)
    with open(os.path.join(dir, run_name, "coreference_resolution.json"), "w") as f:
        json.dump(coreference_resolution_results, f)
    with open(os.path.join(dir, run_name, "entity_typing.json"), "w") as f:
        json.dump(entity_typing_results, f)
    with open(os.path.join(dir, run_name, "relation_extraction.json"), "w") as f:
        json.dump(relation_extraction_results, f)


if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.backends.cuda.matmul.allow_tf32 = True

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

    # Load rel2id
    rel2id = load_json(config["rel2id_path"])
    rel_num_classes = len(rel2id.keys())

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
        re_loss_type=config["re_loss_type"],
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
        for i in range(10):
            train_dataset += train_dataset[:1]
        val_dataset = train_dataset[:2]

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
        },
        {
            "params": model.mention_detection.parameters(),
            "lr": config["md_learning_rate"],
            "amsgrad": True,
        },
        {
            "params": model.coreference_resolution.parameters(),
            "lr": config["cr_learning_rate"],
            "amsgrad": True,
        },
        {
            "params": model.entity_typing.parameters(),
            "lr": config["et_learning_rate"],
            "amsgrad": True,
        },
        {
            "params": model.rel_classifier.parameters(),
            "lr": config["re_learning_rate"],
            "amsgrad": True,
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
        device,
        epochs,
        save_path,
        pretrained_weights_path,
        patience=config["patience"],
        coeff_md=config["coeff_md"],
        coeff_coref=config["coeff_cr"],
        coeff_et=config["coeff_et"],
        coeff_re=config["coeff_re"],
        run_name=run_name,
    )
