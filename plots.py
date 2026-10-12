import numpy as np
import matplotlib.pyplot as plt


def save_sweep(path, title, entries):
    plt.figure(figsize=(7.5, 4.8))
    for label, run in entries:
        h = run["history"]
        plt.plot(h["round"], h["test_accuracy"], label=label)
    plt.xlabel("Communication round")
    plt.ylabel("Test accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_mia_clients(path, mia):
    ks = [1, 10, 20, 50]
    global_auc = [mia[f"clients_{k}"]["global"]["roc_auc"] for k in ks]
    local_mean = [mia[f"clients_{k}"]["local"]["roc_auc"].mean() for k in ks]
    local_std = [mia[f"clients_{k}"]["local"]["roc_auc"].std(ddof=0) for k in ks]

    plt.figure(figsize=(7, 4.5))
    plt.plot(ks, global_auc, marker="o", label="Global model AUC")
    plt.errorbar(ks, local_mean, yerr=local_std, marker="o", capsize=4, label="Local mean AUC ± std")
    plt.axhline(0.5, linestyle="--", linewidth=1, label="Random guess")
    plt.xlabel("Number of clients")
    plt.ylabel("MIA ROC AUC")
    plt.title("Round-1 global vs local membership leakage")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_mia_roc(path, result):
    g = result["global"]
    lfpr, ltpr = result["representative_local_curve"]
    rep_id = result["representative_local_client"]
    rep_auc = result["local"].loc[result["local"]["client_id"] == rep_id, "roc_auc"].iloc[0]

    plt.figure(figsize=(6, 5))
    plt.plot(g["fpr"], g["tpr"], label=f"Global (AUC={g['roc_auc']:.3f})")
    plt.plot(lfpr, ltpr, label=f"Representative local (AUC={rep_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random guess")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("LOSS MIA ROC, baseline configuration")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_privacy_convergence(path, runs):
    plt.figure(figsize=(7.5, 4.8))
    for name in ["FedAvg", "HE + DP"]:
        h = runs[name]["history"]
        plt.plot(h["round"], h["test_accuracy"], label=name)
    plt.xlabel("Communication round")
    plt.ylabel("Test accuracy")
    plt.title("FedAvg vs HE + DP convergence")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_privacy_mia(path, mia):
    methods = ["FedAvg", "HE + DP"]
    global_auc = [mia[m]["global"]["roc_auc"] for m in methods]
    local_auc = [mia[m]["local"]["roc_auc"].mean() for m in methods]
    x = np.arange(len(methods))
    width = 0.36

    plt.figure(figsize=(7, 4.8))
    plt.bar(x - width / 2, global_auc, width, label="Global model AUC")
    plt.bar(x + width / 2, local_auc, width, label="Local model mean AUC")
    plt.axhline(0.5, linestyle="--", linewidth=1, label="Random guess")
    plt.xticks(x, methods)
    plt.ylabel("Round-1 LOSS MIA ROC AUC")
    plt.title("Membership inference: FedAvg vs HE + DP")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_overhead(path, table):
    plt.figure(figsize=(8.5, 4.8))
    plt.bar(table["component"], table["seconds"])
    plt.ylabel("Seconds")
    plt.title("HE + DP cryptographic overhead")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
