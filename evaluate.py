import torch
import numpy as np
import json
from tqdm import tqdm


def compute_metrics_multi_class2md(predicted_labels, true_labels):
    """
    Compute precision, recall, and F1 score for multi-class classification.

    Args:
        logits (torch.Tensor): The predicted logits from the model.
        targets (torch.Tensor): The true labels.

    Returns:
        tuple: A tuple containing the precision, recall, and F1 score.

    """
    predicted_labels = (predicted_labels > 0).long()
    true_labels = (true_labels > 0).long()
    assert predicted_labels.size() == true_labels.size()

    true_positive = torch.sum(
        (predicted_labels == true_labels) & (true_labels != 0)
    ).item()
    false_positive = torch.sum(
        (predicted_labels != true_labels) & (predicted_labels != 0)
    ).item()
    false_negative = torch.sum((predicted_labels == 0) & (true_labels != 0)).item()

    if true_positive == 0:
        precision = 0
        recall = 0
        f1_score = 0
    else:
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score


def evaluate_multi_class2md(preds, gt):
    md_preds = torch.cat(preds, dim=0).to("cpu")
    span_labels = torch.cat(
        [labels for batch in gt for labels in batch],
        dim=0,
    ).to("cpu")
    mc_precision, mc_recall, mc_f1 = compute_metrics_multi_class2md(
        md_preds,
        span_labels,
    )
    return mc_precision, mc_recall, mc_f1


def compute_cr_tp_fp_fn(
    predicted_clusters, ground_truth_clusters, e2e_md_false_positives=0
):
    """
    Compute recall, precision, and F1 score between predicted entity clusters and ground truth entity clusters.

    Args:
        predicted_clusters (list): List of predicted entity clusters.
        ground_truth_clusters (list): List of ground truth entity clusters.

    Returns:
        float: Recall score.
        float: Precision score.
        float: F1 score.

    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    predicted_clusters = [[pos for pos in entity] for entity in predicted_clusters]
    ground_truth_clusters = [
        [pos for pos in entity] for entity in ground_truth_clusters
    ]
    for pred_cluster in predicted_clusters:
        if pred_cluster in ground_truth_clusters:
            true_positives += 1
        else:
            false_positives += 1

    for true_cluster in ground_truth_clusters:
        if true_cluster not in predicted_clusters:
            false_negatives += 1

    return (
        true_positives,
        false_positives,
        false_negatives,
    )


def get_clusters_per_features(preds, cr_batch_indexes=None):
    if cr_batch_indexes is not None and any(cr_batch_indexes):
        e2e_preds = []
        for indexes, batch in zip(cr_batch_indexes, preds):
            for i, f_clusters in enumerate(batch):
                if i in indexes:
                    e2e_preds.append(f_clusters)
                else:
                    e2e_preds.append([])
        preds = e2e_preds
    else:
        preds = [f_clusters for batch in preds for f_clusters in batch]
    return preds


def evaluate_coreference(features, preds, titles_preds=None, md_false_positives=None):
    """
    Evaluate coreference resolution performance.

    Args:
        features (list): List of features.
        preds (list): List of predicted clusters.

    Returns:
        tuple: Precision, recall, and F1 score.
    """
    gt_clusters = [f["entity_clusters"] for f in features]
    tp, fp, fn = 0, 0, 0
    if md_false_positives is not None:
        fp += sum([sum(batch) for batch in md_false_positives])
    if titles_preds is not None:
        for i, f in enumerate(features):
            f_gt = gt_clusters[i]
            if f["title"] in titles_preds:
                f_preds = preds[titles_preds.index(f["title"])]
                tp_, fp_, fn_ = compute_cr_tp_fp_fn(f_preds, f_gt)
                tp += tp_
                fp += fp_
                fn += fn_
            else:
                fn += len(f_gt)

    else:
        for f_preds, f_gt in zip(preds, gt_clusters):
            tp_, fp_, fn_ = compute_cr_tp_fp_fn(f_preds, f_gt)
            tp += tp_
            fp += fp_
            fn += fn_
    if tp == 0:
        precision = recall = f1 = 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1


def to_official_entity_types(
    preds, features, id2ent, titles_preds=None, e2e_entity_clusters=None
):
    clusters, title = [], []

    if titles_preds and e2e_entity_clusters:
        print("len(e2e_entity_clusters):", len(e2e_entity_clusters))
        for i, f in enumerate(features):
            if f["title"] in titles_preds:
                f_clusters = e2e_entity_clusters[titles_preds.index(f["title"])]
                clusters += f_clusters
                title += [f["title"] for _ in f_clusters]
    else:
        for f in features:
            f_clusters = f["entity_clusters"]
            clusters += f_clusters
            title += [f["title"] for _ in f_clusters]

    official_res = []
    for i in range(len(preds)):  # for each predicted type
        pred = preds[i]

        curr_result = {
            "title": title[i],
            "cluster": clusters[i],
            "type": id2ent[pred],
        }
        official_res.append(curr_result)
    return official_res


def official_entity_types_evaluate(tmp, ds, id2ent):
    truth = ds
    std = []
    titleset = set([])

    for x in truth:
        title = x["title"]
        titleset.add(title)

        for cluster, cluster_type in zip(x["entity_clusters"], x["entity_types"]):
            std.append((title, cluster, id2ent[np.argmax(cluster_type)]))

    tot_entities = len(std)
    tmp.sort(key=lambda x: (x["title"], x["cluster"], x["type"]))

    if tmp:
        submission_answer = [tmp[0]]

        for i in range(1, len(tmp)):
            x = tmp[i]
            y = tmp[i - 1]
            if (x["title"], x["cluster"], x["type"]) != (
                y["title"],
                y["cluster"],
                y["type"],
            ):
                submission_answer.append(tmp[i])

        tp, fp = 0, 0

        for x in submission_answer:
            title = x["title"]
            type = x["type"]
            cluster = x["cluster"]
            if (title, cluster, type) in std:
                tp += 1
            else:
                fp += 1

        precision = tp / (tp + fp) if len(submission_answer) > 0 else 0
        recall = tp / tot_entities if tot_entities > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0
        )
    else:
        precision = recall = f1 = 0
        print("No predictions found.")
    return precision, recall, f1


def evaluate_entity_types(
    features,
    ent_preds,
    id2ent,
    path,
    titles_preds=None,
    e2e_entity_clusters=None,
):
    res = to_official_entity_types(
        ent_preds,
        features,
        id2ent,
        titles_preds=titles_preds,
        e2e_entity_clusters=e2e_entity_clusters,
    )
    return official_entity_types_evaluate(res, features, id2ent)


def get_re_extra_false_positives(cr_preds, md_false_positives):
    md_false_positives = [
        f_md_false_positives
        for batch in md_false_positives
        for f_md_false_positives in batch
    ]
    extra_re_false_positives = [
        f_md_false_positives * len(cr_preds[i])
        + f_md_false_positives * (f_md_false_positives - 1)
        for i, f_md_false_positives in enumerate(md_false_positives)
    ]
    return sum(extra_re_false_positives)


def extract_relative_score(scores: list, topks: list) -> list:
    """
    Get relative score from topk predictions.
    Input:
        :scores: a list containing scores of topk predictions.
        :topks: a list containing relation labels of topk predictions.
    Output:
        :scores: a list containing relative scores of topk predictions.
    """

    na_score = scores[-1].item() - 1
    if 0 in topks:
        na_score = scores[np.where(topks == 0)].item()

    scores -= na_score

    return scores


def to_official(
    preds: np.array,
    features: list,
    id2rel: dict,
    titles_preds: np.array = None,
    e2e_entity_centric_hts: list = None,
    scores: np.array = None,
    topks: np.array = None,
):
    """
    Convert the predictions to official format for evaluating.
    Input:
        :preds: list of dictionaries, each dictionary entry is a predicted relation triple from the original document. Keys: ['title', 'h_idx', 't_idx', 'r', 'evidence', 'score'].
        :features: list of features within each document. Identical to the lists obtained from pre-processing.
    Output:
        :official_res: official results used for evaluation.
        :res: topk results to be dumped into file, which can be further used during fushion.
    """
    h_idx, t_idx, title = [], [], []
    if titles_preds and e2e_entity_centric_hts:
        for i, f in enumerate(features):
            if f["title"] in titles_preds:
                hts = e2e_entity_centric_hts[titles_preds.index(f["title"])]
                h_idx += [ht[0] for ht in hts]
                t_idx += [ht[1] for ht in hts]
                title += [f["title"] for _ in hts]
    else:
        for f in features:
            hts = f["entity_centric_hts"]
            h_idx += [ht[0] for ht in hts]
            t_idx += [ht[1] for ht in hts]
            title += [f["title"] for _ in hts]

    official_res = []
    for i in range(preds.shape[0]):  # for each entity pair
        # if scores.size > 0:
        #     score = extract_relative_score(scores[i], topks[i])
        #     pred = topks[i]
        # else:
        pred = preds[i]
        pred = np.nonzero(pred)[0].tolist()

        for p in pred:  # for each predicted relation label (topk)
            if p != 0 and p in np.nonzero(preds[i])[0].tolist():
                curr_result = {
                    "title": title[i],
                    "h_idx": h_idx[i],
                    "t_idx": t_idx[i],
                    "r": id2rel[p],
                }
                official_res.append(curr_result)
    return official_res


def official_evaluate(
    tmp,
    path,
):

    truth = json.load(open(path))

    std = []
    titleset = set([])

    title2vectexSet = {}

    for x in truth:
        title = x["title"]
        titleset.add(title)

        vertexSet = x["vertexSet"]
        title2vectexSet[title] = vertexSet

        if "labels" not in x:  # official test set from DocRED
            continue

        for label in x["labels"]:
            r = label["r"]
            h_idx = label["h"]
            t_idx = label["t"]
            std.append((title, r, h_idx, t_idx))

    std = set(std)
    tot_relations = len(std)
    tmp.sort(key=lambda x: (x["title"], x["h_idx"], x["t_idx"], x["r"]))
    std = sorted(
        std, key=lambda x: (x[0], x[2], x[3], x[1])
    )  # sort by title, h_idx, t_idx, r

    if tmp:
        submission_answer = [tmp[0]]
        for i in range(1, len(tmp)):
            x = tmp[i]
            y = tmp[i - 1]
            if (x["title"], x["h_idx"], x["t_idx"], x["r"]) != (
                y["title"],
                y["h_idx"],
                y["t_idx"],
                y["r"],
            ):
                submission_answer.append(tmp[i])

        tp_re, fp_re = 0, 0

        for x in submission_answer:
            title = x["title"]
            r = x["r"]
            h_idx = x["h_idx"]
            t_idx = x["t_idx"]
            if (title, r, h_idx, t_idx) in std:
                tp_re += 1
            else:
                fp_re += 1

        re_precision = tp_re / (tp_re + fp_re) if len(submission_answer) > 0 else 0
        re_recall = tp_re / tot_relations if tot_relations > 0 else 0
        re_f1 = (
            2 * re_precision * re_recall / (re_precision + re_recall)
            if re_precision + re_recall > 0
            else 0
        )
    else:
        re_precision = re_recall = re_f1 = 0
        print("No predictions found.")
    return re_precision, re_recall, re_f1


def evaluate_relations(
    features,
    rel_preds,
    id2rel,
    scores,
    topks,
    path,
    titles_preds=None,
    e2e_entity_centric_hts=None,
):

    rel_preds = np.concatenate(rel_preds, axis=0)
    if titles_preds is not None:
        titles_preds = [title for titles in titles_preds for title in titles]
    # scores = np.concatenate(scores, axis=0)
    # topks = np.concatenate(topks, axis=0)

    res = to_official(
        rel_preds,
        features,
        id2rel,
        titles_preds=titles_preds,
        e2e_entity_centric_hts=e2e_entity_centric_hts,
        # scores=scores,
        # topks=topks,
    )
    return official_evaluate(
        res,
        path,
    )


def evaluate(
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
    dataloader,
    path,
    e2e_mode=False,
    e2e_titles_cr_preds=None,
    e2e_titles_et_preds=None,
    e2e_titles_re_preds=None,
    e2e_entity_clusters=None,
    e2e_entity_centric_hts=None,
    md_false_positives=None,
):
    if coefficients[0] > 0 and not e2e_mode:
        md_loss /= len(dataloader)
        md_precision, md_recall, md_f1 = evaluate_multi_class2md(
            md_preds,
            [batch["span_labels"] for batch in dataloader],
        )
    else:
        md_precision, md_recall, md_f1 = 0, 0, 0
    md_metrics = [md_precision, md_recall, md_f1]

    if coefficients[1] > 0:
        cr_loss /= len(dataloader)
        # cr_preds = get_clusters_per_features(cr_preds, cr_batch_indexes)
        if e2e_titles_cr_preds is not None:
            e2e_titles_cr_preds = [
                title for batch in e2e_titles_cr_preds for title in batch
            ]
        cr_preds = [f_clusters for batch in cr_preds for f_clusters in batch]
        cr_precision, cr_recall, cr_f1 = evaluate_coreference(
            dataloader.dataset,
            cr_preds,
            titles_preds=e2e_titles_cr_preds,
            md_false_positives=md_false_positives,
        )
    else:
        cr_precision, cr_recall, cr_f1 = 0, 0, 0
    cr_metrics = [cr_precision, cr_recall, cr_f1]

    if coefficients[2] > 0:
        et_loss /= len(dataloader)
        et_preds = [f_clusters for batch in et_preds for f_clusters in batch]
        if e2e_titles_et_preds is not None:
            e2e_titles_et_preds = [
                title for batch in e2e_titles_et_preds for title in batch
            ]
        et_precision, et_recall, et_f1 = evaluate_entity_types(
            dataloader.dataset,
            et_preds,
            id2ent=id2ent,
            path=path,
            titles_preds=e2e_titles_et_preds,
            e2e_entity_clusters=e2e_entity_clusters,
        )
    else:
        et_precision, et_recall, et_f1 = 0, 0, 0
    et_metrics = [et_precision, et_recall, et_f1]

    if e2e_entity_centric_hts is not None:
        e2e_entity_centric_hts = [
            ht for hts in e2e_entity_centric_hts for ht in hts if ht
        ]
        re_preds = [pred for pred in re_preds if len(pred) > 0]

    if (coefficients[3] > 0) and (len(re_preds) > 0):
        re_loss /= len(dataloader)

        re_precision, re_recall, re_f1 = evaluate_relations(
            dataloader.dataset,
            re_preds,
            id2rel,
            re_scores,
            re_topks,
            path,
            titles_preds=e2e_titles_re_preds,
            e2e_entity_centric_hts=e2e_entity_centric_hts,
        )
    else:
        re_precision, re_recall, re_f1 = 0, 0, 0
    re_metrics = [re_precision, re_recall, re_f1]
    return (
        md_loss,
        md_metrics,
        cr_loss,
        cr_metrics,
        et_loss,
        et_metrics,
        re_loss,
        re_metrics,
    )
