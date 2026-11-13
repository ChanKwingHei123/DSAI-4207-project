import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BOOTSTRAP_RESAMPLES = 10000


def run_paired_bootstrap(control_scores, treatment_scores, seed: int = 42):
    rng = np.random.default_rng(seed)
    control = np.asarray(control_scores, dtype=float)
    treatment = np.asarray(treatment_scores, dtype=float)
    delta = treatment - control

    boot_indices = rng.choice(len(delta), size=(BOOTSTRAP_RESAMPLES, len(delta)), replace=True)
    boot_means = delta[boot_indices].mean(axis=1)

    low_ci = float(np.percentile(boot_means, 2.5))
    high_ci = float(np.percentile(boot_means, 97.5))
    mean_diff = float(delta.mean())

    p_val = float(2 * min((boot_means >= 0).mean(), (boot_means <= 0).mean()))
    p_val = max(p_val, 1.0 / BOOTSTRAP_RESAMPLES)
    return mean_diff, low_ci, high_ci, p_val


def apply_holm_bonferroni(p_values):
    p_arr = np.asarray(p_values, dtype=float)
    sorted_indices = np.argsort(p_arr)
    m = len(p_arr)
    adjusted = np.zeros(m)

    running_max = 0.0
    for rank, idx in enumerate(sorted_indices):
        adj = min(p_arr[idx] * (m - rank), 1.0)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted


def generate_tables_and_figures(df: pd.DataFrame):
    models = ["Qwen3-8B", "Qwen3-4B", "Llama-3.1-8B-Instruct"]
    doses = [0, 1, 2, 4, 8]

    results_summary = []
    raw_p_vals = []

    for model in models:
        baseline = df[(df["model"] == model) & (df["dose"] == 0)]["em_score"].values
        for dose in doses[1:]:
            for c_type in ["random", "hard"]:
                subset = df[(df["model"] == model) & (df["dose"] == dose) & (df["context_type"] == c_type)]
                treatment = subset["em_score"].values
                diff, low, high, p = run_paired_bootstrap(baseline, treatment)
                results_summary.append({
                    "Model": model,
                    "Dose": dose,
                    "Type": c_type,
                    "EM": float(treatment.mean()),
                    "Delta": diff,
                    "CI_Low": low,
                    "CI_High": high,
                    "Raw_P": p,
                })
                raw_p_vals.append(p)

    adj_p_vals = apply_holm_bonferroni(raw_p_vals)
    for row, adj_p in zip(results_summary, adj_p_vals):
        row["Adjusted_P"] = float(adj_p)

    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("report/figures", exist_ok=True)

    df_metrics = pd.DataFrame(results_summary)
    df_metrics.to_csv("results/tables/main_results.csv", index=False)

    strata = df.groupby(["model", "dose", "context_type", "closed_book_correct"])["em_score"].mean().unstack()
    strata.columns = ["Closed_Book_Incorrect_RAG_EM", "Closed_Book_Correct_RAG_EM"]
    strata.to_csv("results/tables/strata_results.csv")

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, model in zip(axes, models):
        model_df = df_metrics[df_metrics["Model"] == model]
        base_val = df[(df["model"] == model) & (df["dose"] == 0)]["em_score"].mean()

        hard_df = model_df[model_df["Type"] == "hard"]
        ax.plot([0] + list(hard_df["Dose"]), [base_val] + list(hard_df["EM"]),
                marker="o", linestyle="-", color="#d95f02", linewidth=2.5, label="Topically Related (Hard)")

        rand_df = model_df[model_df["Type"] == "random"]
        ax.plot([0] + list(rand_df["Dose"]), [base_val] + list(rand_df["EM"]),
                marker="s", linestyle="--", color="#7570b3", linewidth=2.0, label="Disjoint (Random)")

        ax.set_title(f"{model} Context Performance Drop", fontsize=13, fontweight="bold")
        ax.set_xlabel("Volumetric Distractor Dose ($n$ Passages)", fontsize=11)
        ax.set_xticks(doses)
        ax.grid(True, linestyle=":", alpha=0.6)

    axes[0].set_ylabel("SQuAD Exact Match (EM) Score", fontsize=12)
    axes[0].legend(loc="lower left", frameon=True)

    plt.tight_layout()
    plt.savefig("report/figures/dose_response.pdf", bbox_inches="tight", dpi=300)
    print("Exported tables to results/tables/ and plot to report/figures/dose_response.pdf")
