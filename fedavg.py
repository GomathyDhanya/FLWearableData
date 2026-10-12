import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from config import DEVICE, SEED, MAX_ROUNDS, MIN_ROUNDS, PATIENCE, MIN_DELTA, seed_everything
from model import new_model, state_to_cpu, evaluate


def split_iid(y, n_clients, seed=SEED):
    if n_clients == 1:
        return [np.arange(len(y), dtype=np.int64)]

    splitter = StratifiedKFold(n_splits=n_clients, shuffle=True, random_state=seed)
    folds = [idx for _, idx in splitter.split(np.zeros(len(y)), y)]
    size = min(map(len, folds))
    rng = np.random.default_rng(seed)
    return [np.sort(rng.permutation(idx)[:size]) for idx in folds]


def train_local(global_state, dataset, indices, epochs, batch_size, lr):
    model = new_model(global_state)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

    return state_to_cpu(model)


def average_models(states):
    return {
        key: torch.stack([state[key].float() for state in states]).mean(dim=0)
        for key in states[0]
    }


def run_fedavg(train_dataset, test_dataset, y_train, n_clients, local_epochs, batch_size, lr):
    seed_everything()
    clients = split_iid(y_train, n_clients)
    global_model = new_model()
    global_state = state_to_cpu(global_model)

    history = []
    best_accuracy = -float("inf")
    best_round = 0
    best_state = None
    wait = 0
    stopped_reason = "max_rounds"
    global_checkpoints = {}
    local_checkpoints = {}

    for round_idx in range(1, MAX_ROUNDS + 1):
        local_states = [
            train_local(global_state, train_dataset, idx, local_epochs, batch_size, lr)
            for idx in clients
        ]

        global_state = average_models(local_states)
        global_model.load_state_dict(global_state)
        train_metrics = evaluate(global_model, train_dataset)
        test_metrics = evaluate(global_model, test_dataset)

        history.append({
            "round": round_idx,
            "train_loss": train_metrics["loss"],
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
        })

        if round_idx == 1:
            global_checkpoints[1] = copy.deepcopy(global_state)
            local_checkpoints[1] = copy.deepcopy(local_states)

        if test_metrics["accuracy"] >= best_accuracy + MIN_DELTA:
            best_accuracy = test_metrics["accuracy"]
            best_round = round_idx
            best_state = copy.deepcopy(global_state)
            wait = 0
        else:
            wait += 1

        if round_idx == 1 or round_idx % 10 == 0:
            print(f"round={round_idx:3d} acc={test_metrics['accuracy']:.4f} f1={test_metrics['macro_f1']:.4f}")

        if round_idx >= MIN_ROUNDS and wait >= PATIENCE:
            stopped_reason = "converged"
            break

    return {
        "config": {
            "n_clients": n_clients,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "lr": lr,
        },
        "history": pd.DataFrame(history),
        "final_state": copy.deepcopy(global_state),
        "best_state": best_state,
        "best_round": best_round,
        "best_test_accuracy": best_accuracy,
        "stopped_round": history[-1]["round"],
        "stopped_reason": stopped_reason,
        "client_splits": clients,
        "global_checkpoints": global_checkpoints,
        "local_checkpoints": local_checkpoints,
    }
