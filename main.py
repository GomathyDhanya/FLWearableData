import pickle
import shutil
import time
import numpy as np
import pandas as pd
from config import DEVICE, RESULTS_DIR, MAX_ROUNDS, HE_DP_SIGMA
from data import load_data
from fedavg import run_fedavg, average_models
from mia import evaluate_mia
from plots import save_sweep, save_mia_clients, save_mia_roc, save_privacy_convergence, save_privacy_mia, save_overhead

BASELINE = {"n_clients": 20, "local_epochs": 5, "batch_size": 32, "lr": 0.01}
EXPERIMENTS = {
    "clients_1": {**BASELINE, "n_clients": 1},
    "clients_10": {**BASELINE, "n_clients": 10},
    "clients_20": BASELINE,
    "clients_50": {**BASELINE, "n_clients": 50},
    "epochs_1": {**BASELINE, "local_epochs": 1},
    "epochs_10": {**BASELINE, "local_epochs": 10},
    "batch_16": {**BASELINE, "batch_size": 16},
    "batch_64": {**BASELINE, "batch_size": 64},
    "lr_001": {**BASELINE, "lr": 0.001},
    "lr_05": {**BASELINE, "lr": 0.05},
}


def load_pickle(path, default):
    if not path.exists():
        return default
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(value, path):
    with open(path, "wb") as f:
        pickle.dump(value, f)


def run_task_a(data):
    path = RESULTS_DIR / "task_a_runs.pkl"
    runs = load_pickle(path, {})

    for name, config in EXPERIMENTS.items():
        if name not in runs:
            print("\n", name, config)
            runs[name] = run_fedavg(data.train_dataset, data.test_dataset, data.y_train, **config)
            save_pickle(runs, path)

    rows = []
    for name, run in runs.items():
        h = run["history"]
        rows.append({
            "experiment": name,
            **run["config"],
            "stopped_round": run["stopped_round"],
            "stop_reason": run["stopped_reason"],
            "best_round": run["best_round"],
            "best_test_accuracy": run["best_test_accuracy"],
            "final_test_accuracy": h.iloc[-1]["test_accuracy"],
            "final_macro_f1": h.iloc[-1]["macro_f1"],
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "task_a_convergence_summary.csv", index=False)
    print(summary.to_string(index=False))

    save_sweep(RESULTS_DIR / "taskA_clients.png", "Number of clients", [(f"K={k}", runs[f"clients_{k}"]) for k in [1, 10, 20, 50]])
    save_sweep(RESULTS_DIR / "taskA_epochs.png", "Local epochs", [("E=1", runs["epochs_1"]), ("E=5", runs["clients_20"]), ("E=10", runs["epochs_10"])])
    save_sweep(RESULTS_DIR / "taskA_batch.png", "Mini-batch size", [("B=16", runs["batch_16"]), ("B=32", runs["clients_20"]), ("B=64", runs["batch_64"])])
    save_sweep(RESULTS_DIR / "taskA_lr.png", "Learning rate", [("LR=.001", runs["lr_001"]), ("LR=.01", runs["clients_20"]), ("LR=.05", runs["lr_05"])])
    return runs


def run_task_b(data, runs):
    path = RESULTS_DIR / "task_a_mia_round1.pkl"
    mia = load_pickle(path, {})

    for name, run in runs.items():
        if name not in mia:
            print("MIA:", name)
            mia[name] = evaluate_mia(run, data.train_dataset, data.test_dataset)
            save_pickle(mia, path)

    rows = []
    for name, result in mia.items():
        cfg = runs[name]["config"]
        g = result["global"]
        rows.append({
            "experiment": name,
            "model_type": "global",
            "client_id": np.nan,
            **cfg,
            "balanced_accuracy": g["balanced_accuracy"],
            "roc_auc": g["roc_auc"],
            "tpr_at_1pct_fpr": g["tpr_at_1pct_fpr"],
            "tpr_at_5pct_fpr": g["tpr_at_5pct_fpr"],
            "tpr_at_10pct_fpr": g["tpr_at_10pct_fpr"],
        })

        for _, local in result["local"].iterrows():
            rows.append({
                "experiment": name,
                "model_type": "local",
                "client_id": int(local["client_id"]),
                **cfg,
                "balanced_accuracy": local["balanced_accuracy"],
                "roc_auc": local["roc_auc"],
                "tpr_at_1pct_fpr": local["tpr_at_1pct_fpr"],
                "tpr_at_5pct_fpr": local["tpr_at_5pct_fpr"],
                "tpr_at_10pct_fpr": local["tpr_at_10pct_fpr"],
            })

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "task_b_all_models_mia.csv", index=False)
    print(table.to_string(index=False))
    save_mia_clients(RESULTS_DIR / "mia_clients.png", mia)
    save_mia_roc(RESULTS_DIR / "mia_roc.png", mia["clients_20"])
    return mia


def run_extension(data, runs):
    from privacy import run_he_dp

    run_path = RESULTS_DIR / "he_dp_extension_runs.pkl"
    saved = load_pickle(run_path, {})
    if "HE + DP" not in saved:
        saved["HE + DP"] = run_he_dp(data.train_dataset, data.test_dataset, data.y_train, **BASELINE, sigma=HE_DP_SIGMA)
        save_pickle(saved, run_path)

    extension_runs = {"FedAvg": runs["clients_20"], "HE + DP": saved["HE + DP"]}

    mia_path = RESULTS_DIR / "he_dp_extension_mia_round1.pkl"
    extension_mia = load_pickle(mia_path, {})
    for name, run in extension_runs.items():
        if name not in extension_mia:
            extension_mia[name] = evaluate_mia(run, data.train_dataset, data.test_dataset)
            save_pickle(extension_mia, mia_path)

    rows = []
    for name in ["FedAvg", "HE + DP"]:
        run = extension_runs[name]
        mia = extension_mia[name]
        last = run["history"].iloc[-1]
        rows.append({
            "Method": name,
            "Test accuracy": last["test_accuracy"],
            "Macro F1": last["macro_f1"],
            "Global balanced accuracy": mia["global"]["balanced_accuracy"],
            "Global ROC AUC": mia["global"]["roc_auc"],
            "Global TPR@1%": mia["global"]["tpr_at_1pct_fpr"],
            "Global TPR@5%": mia["global"]["tpr_at_5pct_fpr"],
            "Global TPR@10%": mia["global"]["tpr_at_10pct_fpr"],
            "Local ROC AUC mean": mia["local"]["roc_auc"].mean(),
        })

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "fedavg_vs_he_dp.csv", index=False)
    print(results.round(4).to_string(index=False))

    overhead = extension_runs["HE + DP"]["overhead"]
    t0 = time.perf_counter()
    average_models(extension_runs["FedAvg"]["local_checkpoints"][1])
    plain_time = time.perf_counter() - t0

    overhead_table = pd.DataFrame([
        {"component": "Plain aggregation", "seconds": plain_time},
        {"component": "HE encryption", "seconds": overhead["encryption_time_sec"].mean()},
        {"component": "HE aggregation", "seconds": overhead["aggregation_time_sec"].mean()},
        {"component": "HE decryption", "seconds": overhead["decryption_time_sec"].mean()},
    ])
    overhead_table.to_csv(RESULTS_DIR / "he_dp_overhead.csv", index=False)

    save_privacy_convergence(RESULTS_DIR / "privacy_convergence.png", extension_runs)
    save_privacy_mia(RESULTS_DIR / "privacy_mia.png", extension_mia)
    save_overhead(RESULTS_DIR / "privacy_overhead.png", overhead_table)

    mean_bytes = overhead["communication_bytes"].mean()
    print(f"Encrypted communication/round: {mean_bytes / 1024 / 1024:.2f} MiB")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Device:", DEVICE)
    print("Max rounds:", MAX_ROUNDS)

    data, counts = load_data()
    print("Train:", data.X_train.shape)
    print("Test:", data.X_test.shape)
    print(counts.to_string())

    runs = run_task_a(data)
    run_task_b(data, runs)
    run_extension(data, runs)

    archive = shutil.make_archive("fedavg_assignment_results", "zip", RESULTS_DIR)
    print("Results:", archive)


if __name__ == "__main__":
    main()
