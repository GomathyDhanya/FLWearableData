import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from config import DEVICE
from model import new_model


@torch.no_grad()
def losses(state, dataset):
    model = new_model(state)
    model.eval()
    loader = DataLoader(dataset, batch_size=2048, shuffle=False)
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    values = []

    for x, y in loader:
        values.append(loss_fn(model(x.to(DEVICE)), y.to(DEVICE)).cpu().numpy())

    return np.concatenate(values)


def loss_mia(member_losses, nonmember_losses):
    scores = np.concatenate([-member_losses, -nonmember_losses])
    labels = np.concatenate([np.ones(len(member_losses)), np.zeros(len(nonmember_losses))])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    balanced = 0.5 * (tpr + 1 - fpr)
    best = int(np.argmax(balanced))

    def tpr_at(target):
        values = tpr[fpr <= target]
        return float(values.max()) if len(values) else 0.0

    return {
        "balanced_accuracy": float(balanced[best]),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "threshold": float(thresholds[best]),
        "tpr_at_1pct_fpr": tpr_at(0.01),
        "tpr_at_5pct_fpr": tpr_at(0.05),
        "tpr_at_10pct_fpr": tpr_at(0.10),
        "fpr": fpr,
        "tpr": tpr,
    }


def evaluate_mia(run, train_dataset, test_dataset):
    global_state = run["global_checkpoints"][1]
    local_states = run["local_checkpoints"][1]

    train_global = losses(global_state, train_dataset)
    test_global = losses(global_state, test_dataset)
    global_result = loss_mia(train_global, test_global)

    rows = []
    curves = []

    for client_id, (indices, state) in enumerate(zip(run["client_splits"], local_states)):
        train_local = losses(state, train_dataset)
        test_local = losses(state, test_dataset)
        mask = np.ones(len(train_dataset), dtype=bool)
        mask[indices] = False
        result = loss_mia(train_local[indices], np.concatenate([train_local[mask], test_local]))
        rows.append({"client_id": client_id, **{k: v for k, v in result.items() if k not in ["fpr", "tpr"]}})
        curves.append((result["fpr"], result["tpr"]))

    local = pd.DataFrame(rows)
    mean_auc = local["roc_auc"].mean()
    rep = int((local["roc_auc"] - mean_auc).abs().idxmin())

    return {
        "global": global_result,
        "local": local,
        "representative_local_client": int(local.loc[rep, "client_id"]),
        "representative_local_curve": curves[rep],
    }
