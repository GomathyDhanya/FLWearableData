import copy
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tenseal as ts
from opacus import PrivacyEngine
from torch.utils.data import DataLoader, Subset
from config import DEVICE, MAX_ROUNDS, MIN_ROUNDS, PATIENCE, MIN_DELTA, MAX_GRAD_NORM, seed_everything
from model import new_model, state_to_cpu, evaluate
from fedavg import split_iid

POLY_DEGREE = 8192
CHUNK_SIZE = POLY_DEGREE // 2


def flatten_state(state):
    keys = list(state)
    shapes = [state[k].shape for k in keys]
    sizes = [state[k].numel() for k in keys]
    vec = torch.cat([state[k].reshape(-1).float() for k in keys]).numpy()
    return vec, keys, shapes, sizes


def vector_to_state(vec, reference, keys, shapes, sizes):
    state = {}
    offset = 0
    for key, shape, size in zip(keys, shapes, sizes):
        state[key] = torch.tensor(vec[offset:offset + size].reshape(shape), dtype=reference[key].dtype)
        offset += size
    return state


def make_contexts():
    secret = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=POLY_DEGREE,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    secret.global_scale = 2 ** 40
    public = ts.context_from(secret.serialize(
        save_public_key=True,
        save_secret_key=False,
        save_galois_keys=False,
        save_relin_keys=True,
    ))
    return secret, public


def average_he(global_state, local_states, public, secret):
    global_vec, keys, shapes, sizes = flatten_state(global_state)
    deltas = []

    for state in local_states:
        local_vec, _, _, _ = flatten_state(state)
        delta = local_vec - global_vec
        deltas.append([delta[i:i + CHUNK_SIZE] for i in range(0, len(delta), CHUNK_SIZE)])

    t0 = time.perf_counter()
    encrypted = [[ts.ckks_vector(public, chunk.tolist()) for chunk in chunks] for chunks in deltas]
    encryption_time = time.perf_counter() - t0
    communication_bytes = sum(len(chunk.serialize()) for client in encrypted for chunk in client)

    t0 = time.perf_counter()
    averaged = []
    for j in range(len(encrypted[0])):
        total = encrypted[0][j]
        for i in range(1, len(encrypted)):
            total = total + encrypted[i][j]
        averaged.append(total * (1 / len(encrypted)))
    aggregation_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    chunks = [np.asarray(ts.ckks_vector_from(secret, x.serialize()).decrypt(), dtype=np.float32) for x in averaged]
    avg_delta = np.concatenate(chunks)[:len(global_vec)]
    decryption_time = time.perf_counter() - t0

    return vector_to_state(global_vec + avg_delta, global_state, keys, shapes, sizes), {
        "encryption_time_sec": encryption_time,
        "aggregation_time_sec": aggregation_time,
        "decryption_time_sec": decryption_time,
        "communication_bytes": communication_bytes,
    }


def train_local_dp(global_state, dataset, indices, epochs, batch_size, lr, sigma):
    model = new_model(global_state)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=batch_size, shuffle=True)
    engine = PrivacyEngine(accountant="rdp")
    model, optimizer, loader = engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=sigma,
        max_grad_norm=MAX_GRAD_NORM,
    )

    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            optimizer.step()

    base = model._module if hasattr(model, "_module") else model
    return state_to_cpu(base)


def run_he_dp(train_dataset, test_dataset, y_train, n_clients, local_epochs, batch_size, lr, sigma):
    seed_everything()
    clients = split_iid(y_train, n_clients)
    secret, public = make_contexts()
    global_model = new_model()
    global_state = state_to_cpu(global_model)

    history = []
    overhead = []
    best_accuracy = -float("inf")
    best_round = 0
    best_state = None
    wait = 0
    stopped_reason = "max_rounds"
    global_checkpoints = {}
    local_checkpoints = {}

    for round_idx in range(1, MAX_ROUNDS + 1):
        t0 = time.perf_counter()
        local_states = [
            train_local_dp(global_state, train_dataset, idx, local_epochs, batch_size, lr, sigma)
            for idx in clients
        ]
        local_time = time.perf_counter() - t0

        global_state, info = average_he(global_state, local_states, public, secret)
        global_model.load_state_dict(global_state)
        test_metrics = evaluate(global_model, test_dataset)

        history.append({
            "round": round_idx,
            "test_accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
        })
        overhead.append({"round": round_idx, "local_training_time_sec": local_time, **info})

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
            print(f"HE+DP round={round_idx:3d} acc={test_metrics['accuracy']:.4f} f1={test_metrics['macro_f1']:.4f}")

        if round_idx >= MIN_ROUNDS and wait >= PATIENCE:
            stopped_reason = "converged"
            break

    return {
        "config": {
            "n_clients": n_clients,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "dp_sigma": sigma,
            "use_he": True,
        },
        "history": pd.DataFrame(history),
        "overhead": pd.DataFrame(overhead),
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
