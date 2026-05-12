"""
Rebuttal analysis plots.
Generates two figures:
  1. Combined plot (line + equivalence-class scatter) for disjoint7_6 across models
     -> leakage_experiment_diverse_k6_equivalence_class_disjoint7.pdf
  2. Line plot for disjoint5_6 (nh6_nl3 only)
     -> rebuttal_plots/leakage_experiment_diverse_k6_disjoint5.pdf
"""

import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

K = 6
EVAL_PATH = f"{ROOT_DIR}/models/eval/diverse/fixed/nalph_26_seqlen_6_fnlen_6_taskmaxlen_{K}/direct"
EVAL_PATH_ICML = f"{ROOT_DIR}/models/eval/diverse/fixed/nalph_26_seqlen_6_fnlen_6_taskmaxlen_{K}/direct"

STRATEGY_PREFIX = "disjoint7_6"
RATIOS = [0.0, 0.1, 0.2, 0.5, 0.6, 0.7, 1.0]
MODELS = ["nh6_nl3", "nh12_nl12", "gemma1"]
POS_TYPES = ["abs", "rel_global"]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _acc_path_dir(model_name, strategy_prefix, perc):
    if model_name == "nh6_nl3":
        base = EVAL_PATH_ICML
    elif model_name == "nh12_nl12":
        base = EVAL_PATH
    else:
        return f"{EVAL_PATH}/train_{strategy_prefix}_{perc}/eval_{strategy_prefix}_{perc}/gemma1/"
    return f"{base}/train_{strategy_prefix}_{perc}/eval_{strategy_prefix}_{perc}/{'{p}'}/{model_name}/seed_0"


def _latest_accs_file(directory):
    """Return the most-recent accs file in *directory* (sorted by epoch number)."""
    files = [f for f in os.listdir(directory) if f.endswith(".pkl") and "accs" in f]
    if len(files) > 1:
        files = sorted(files, key=lambda x: int(x.split("_")[-1].split(".")[0]), reverse=True)
    return os.path.join(directory, files[0])


def _leakage_perc(strategy_prefix, ratio):
    path = (
        f"{ROOT_DIR}/data/equivalence_classes/"
        f"{strategy_prefix.split('_')[0]}/{K}/{ratio}/number_leaked_test_tasks.json"
    )
    with open(path) as f:
        return json.load(f)["number_leaked_test_tasks"]


def load_disjoint7_results():
    """Load train / val (train_heldout) / test accuracy for all models and pos types."""
    def _empty():
        return {m: {"abs": [], "rel_global": []} for m in MODELS}

    train_results = _empty()
    val_results   = _empty()   # train_heldout split
    test_results  = _empty()
    leakage_list  = _empty()

    for model_name in MODELS:
        for ratio in RATIOS:
            perc = int(ratio * 100)
            for p in POS_TYPES:
                if model_name == "gemma1":
                    d = (
                        f"{EVAL_PATH}/train_{STRATEGY_PREFIX}_{perc}"
                        f"/eval_{STRATEGY_PREFIX}_{perc}/gemma1/"
                    )
                else:
                    base = EVAL_PATH_ICML if model_name == "nh6_nl3" else EVAL_PATH
                    d = (
                        f"{base}/train_{STRATEGY_PREFIX}_{perc}"
                        f"/eval_{STRATEGY_PREFIX}_{perc}/{p}/{model_name}/seed_0"
                    )
                if not os.path.exists(d):
                    continue
                print(_latest_accs_file(d))
                with open(_latest_accs_file(d), "rb") as f:
                    accs = pickle.load(f)

                train_results[model_name][p].append(accs["train"].sharp_accuracy)
                val_results[model_name][p].append(accs["train_heldout"].sharp_accuracy)
                test_results[model_name][p].append(accs["test"].sharp_accuracy)
                leakage_list[model_name][p].append(_leakage_perc(STRATEGY_PREFIX, ratio))

        # sort all splits by leakage percentage (abs pos type as reference)
        idx = np.argsort(leakage_list[model_name]["abs"])
        for p in POS_TYPES:
            leakage_list[model_name][p]  = [leakage_list[model_name][p][i]  for i in idx]
            train_results[model_name][p] = [train_results[model_name][p][i] for i in idx]
            val_results[model_name][p]   = [val_results[model_name][p][i]   for i in idx]
            test_results[model_name][p]  = [test_results[model_name][p][i]  for i in idx]

    return train_results, val_results, test_results, leakage_list


def _load_cec():
    """Load and cluster the composition equivalence classes (model-agnostic)."""
    import sys
    sys.path.insert(0, ROOT_DIR)
    from src.data.equivalence_classes.composition_equivalence_classes import CompositionEquivalenceClasses
    cec = CompositionEquivalenceClasses(K, threshold=0.01)
    cec.load_ce_metric()
    cec.convert_ce_metric_to_matrix()
    cec.cluster_ce_metric_matrix()
    return cec.learner.unique_pairs, cec.learner.cluster_labels


def load_cluster_data_for_model(model_name, unique_pairs, cluster_labels):
    """
    Load per-cluster accuracy data for *model_name* across all RATIOS.

    Accuracy lookup:
      functions_info.pkl (function_tuple → cid) used as a FORWARD map so that
      each unique_pair is looked up directly — avoids the reverse-dict ambiguity
      where multiple function tuples (including identity-based ones) share the
      same combination_id, which would cause the reversed dict to silently drop
      entries and mislabel shared clusters as non-shared.

    Shared determination:
      train_functions.pkl is used directly (lists the actual training function
      compositions for this split) to avoid any combination_id intermediary.
    """
    from collections import defaultdict

    # Use rel_global for nGPT models, abs (only option) for gemma1
    p = "rel_global"
    up_set = set(unique_pairs)   # for fast membership test
    

    rel_plot_data = []

    for ratio in RATIOS:
        perc = int(ratio * 100)

        # --- functions_info: function_tuple → combination_id (forward map) ---
        data_path = (
            f"{ROOT_DIR}/data/diverse/fixed/nalph_26_seqlen_6_fnlen_6_taskmaxlen_{K}"
            f"/direct/{STRATEGY_PREFIX}_{perc}/functions_info.pkl"
        )
        if not os.path.exists(data_path):
            print(f"Warning: missing {data_path}")
            continue
        with open(data_path, "rb") as f:
            functions_info = pickle.load(f)   # function_tuple → cid

        # --- accs.pkl ---
        if model_name == "gemma1":
            d = (
                f"{EVAL_PATH}/train_{STRATEGY_PREFIX}_{perc}"
                f"/eval_{STRATEGY_PREFIX}_{perc}/gemma1/"
            )
        else:
            base = EVAL_PATH_ICML if model_name == "nh6_nl3" else EVAL_PATH
            d = (
                f"{base}/train_{STRATEGY_PREFIX}_{perc}"
                f"/eval_{STRATEGY_PREFIX}_{perc}/{p}/{model_name}/seed_0"
            )
        if not os.path.exists(d):
            continue
        with open(_latest_accs_file(d), "rb") as f:
            accs = pickle.load(f)

        test_comb_acc = accs["test"].combination_sharp_acc or {}
        train_comb_acc = accs["train"].combination_sharp_acc or {}

        # function_tuple → accuracy  (only for unique_pairs members that appear in test)
        test_combination_accs = {
            ft: test_comb_acc[cid]
            for ft, cid in functions_info.items()
            if cid in test_comb_acc
        }

        train_combination_accs = {
            ft: train_comb_acc[cid]
            for ft, cid in functions_info.items()
            if cid in train_comb_acc
        }

        # --- aggregate per cluster ---
        cluster_to_accs   = defaultdict(list)
        cluster_to_shared = {int(cid): False for cid in set(cluster_labels)}
        cluster_size_map  = {int(cid): 0     for cid in set(cluster_labels)}
        cluster_to_ft = {int(cid): [] for cid in set(cluster_labels)}
        
        for idx, raw_cid in enumerate(cluster_labels):
            cid = int(raw_cid)
            cluster_size_map[cid] += 1
            ft = unique_pairs[idx]
            if ft in test_combination_accs:
                cluster_to_ft[cid].append(ft)
                cluster_to_accs[cid].append(test_combination_accs[ft])
            if ft in train_combination_accs:
                cluster_to_shared[cid] = True
            
        # unique cluster ids
        
    
        lp = _leakage_perc(STRATEGY_PREFIX, ratio)

        for cid, accs_list in cluster_to_accs.items():
            if not accs_list:
                continue
            rel_plot_data.append((
                lp,
                float(np.mean(accs_list)),
                float(np.std(accs_list)),
                cluster_size_map[cid],
                cluster_to_shared[cid],
            ))
            # if not cluster_to_shared[cid]:
            #     print(cid, cluster_size_map[cid],  float(np.mean(accs_list)), accs_list)

    return rel_plot_data


# ---------------------------------------------------------------------------
# Plot 1 – combined figure (line plot + cluster scatter)
# ---------------------------------------------------------------------------

MODEL_COLORS = {
    "nh6_nl3":   {"abs": "tab:blue",  "rel_global": "tab:red"},
    "nh12_nl12": {"abs": "tab:green", "rel_global": "tab:orange"},
    "gemma1":    {"abs": "tab:purple", "rel_global": "tab:brown"},
}
MODEL_LABELS = {
    "nh6_nl3":   "nGPT (6h3l)",
    "nh12_nl12": "nGPT (12h12l)",
    "gemma1":    "Gemma-1",
}


def _scatter_panel(ax, rel_plot_data, font_size, title):
    """Draw a single equivalence-class scatter panel onto *ax*."""
    size_scale_factor = 50

    shared     = [(lp, ma, sa, sz) for lp, ma, sa, sz, is_s in rel_plot_data if     is_s]
    non_shared = [(lp, ma, sa, sz) for lp, ma, sa, sz, is_s in rel_plot_data if not is_s]

    def _scatter(data, color, edge_color, alpha, label):
        if not data:
            return
        lps, mas, sas, szs = zip(*data)
        lps_pct = [x * 100 for x in lps]
        mkr     = [s * size_scale_factor for s in szs]
        ax.errorbar(lps_pct, mas, yerr=sas, fmt="none",
                    ecolor=color, alpha=0.3, capsize=5, linewidth=3)
        ax.scatter(lps_pct, mas, s=mkr, c=color, alpha=alpha,
                   label=label, edgecolors=edge_color, linewidth=3)

    _scatter(shared,     "green", "darkgreen", 0.6, "Shared Eq. class")
    _scatter(non_shared, "red",   "darkred",   0.3, "Non-shared Eq. class")

    all_lps_pct = sorted(set(lp * 100 for lp, *_ in rel_plot_data))
    ax.set_xticks(all_lps_pct)
    ax.set_xticklabels([f"{int(x)}%" for x in all_lps_pct], rotation=45)
    ax.set_title(title)
    ax.set_xlabel("% of test sequences with shared equivalences")
    ax.set_ylabel("Mean Accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)

    # colour legend
    color_handles = [
        Line2D([0], [0], marker="o", color="w", label="Shared Eq. class",
               markerfacecolor="green", markeredgecolor="darkgreen",
               markersize=20, linewidth=2),
        Line2D([0], [0], marker="o", color="w", label="Non-shared Eq. class",
               markerfacecolor="red", markeredgecolor="darkred",
               markersize=20, linewidth=2),
    ]
    color_legend = ax.legend(
        handles=color_handles, loc="upper center",
        bbox_to_anchor=(0.5, -0.25), ncol=2,
        fontsize=font_size * 0.75, frameon=False,
    )

    # size legend
    unique_sizes = sorted(set(sz for *_, sz, _ in rel_plot_data))
    size_handles = [
        Line2D([0], [0], marker="o", color="w", label=f"Size: {sz}",
               markerfacecolor="gray", markeredgecolor="black",
               markersize=np.sqrt(sz * size_scale_factor) / 3, linewidth=2)
        for sz in unique_sizes
    ]
    ax.add_artist(color_legend)
    ax.legend(handles=size_handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.38), ncol=5,
              fontsize=font_size * 0.75, frameon=False)


def plot_combined(train_results, val_results, test_results, leakage_list,
                  cluster_data_per_model, save_dir=ROOT_DIR):
    """
    1×2 figure:
      [0] Overall accuracy line plot (all models, train / val / test)
      [1] Eq. class scatter – nh12_nl12

    Encoding in panel 0:
      Color      → model + pos type  (same palette as MODEL_COLORS)
      Line style → split:  Train ":", Val "--", Test "-"
      Marker     → split:  Train "^", Val "s",  Test "o"  (same across all colors)
    """
    import math

    SPLIT_STYLE = {
        # "train": dict(linestyle=":",  marker="^"),
        # "val":   dict(linestyle="--", marker="s"),
        "test":  dict(linestyle="-",  marker="o"),
    }

    font_size = 45
    plt.rcParams.update({
        "font.size": font_size,
        "legend.fontsize": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
    })

    fig, axs = plt.subplots(1, 2, figsize=(40, 12))

    # --- Panel 0: overall accuracy line plot ---
    all_x_positions = sorted(set(
        round(x * 100, 2)
        for m in MODELS
        for x in leakage_list[m]["abs"]
    ))

    for model_name in MODELS:
        lp_abs = leakage_list[model_name]["abs"]
        lp_rel = leakage_list[model_name]["rel_global"]
        if not lp_abs and not lp_rel:
            continue
        abs_label = MODEL_LABELS[model_name] if model_name == "gemma1" \
                    else f"{MODEL_LABELS[model_name]} (Abs)"

        for split, results, label_suffix in [
            # ("train", train_results, abs_label),
            # ("val",   val_results,   abs_label),
            ("test",  test_results,  abs_label),
        ]:
            if lp_abs and results[model_name]["abs"]:
                x_abs = [x * 100 for x in lp_abs]
                axs[0].plot(x_abs, results[model_name]["abs"],
                            color=MODEL_COLORS[model_name]["abs"],
                            **SPLIT_STYLE[split],
                            markersize=font_size / 2, linewidth=2)

            if model_name != "gemma1" and lp_rel and results[model_name]["rel_global"]:
                x_rel = [x * 100 for x in lp_rel]
                axs[0].plot(x_rel, results[model_name]["rel_global"],
                            color=MODEL_COLORS[model_name]["rel_global"],
                            **SPLIT_STYLE[split],
                            markersize=font_size / 2, linewidth=2)

    axs[0].set_xticks(all_x_positions, [f"{int(x)}%" for x in all_x_positions], rotation=45)
    axs[0].set_title("Overall accuracy – disjoint7, K=6")
    axs[0].set_xlabel("% of test sequences with shared equivalences")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_ylim(-0.05, 1.05)
    axs[0].grid(True)

    # --- Legend: 3 columns, 3 rows ---
    # col1 and col2: model/pos entries split in half
    # col3: train/val/test split entries
    color_entries = []
    for model_name in MODELS:
        abs_label = MODEL_LABELS[model_name] if model_name == "gemma1" \
                    else f"{MODEL_LABELS[model_name]} (Abs)"
        color_entries.append(
            Line2D([0], [0], color=MODEL_COLORS[model_name]["abs"],
                   linewidth=3, label=abs_label)
        )
        if model_name != "gemma1":
            color_entries.append(
                Line2D([0], [0], color=MODEL_COLORS[model_name]["rel_global"],
                       linewidth=3, label=f"{MODEL_LABELS[model_name]} (Rel)")
            )

    # split_entries = [
    #     Line2D([0], [0], color="gray", label="Train",
    #            **SPLIT_STYLE["train"], linewidth=3, markersize=font_size * 0.4),
    #     Line2D([0], [0], color="gray", label="Val (heldout)",
    #            **SPLIT_STYLE["val"],   linewidth=3, markersize=font_size * 0.4),
    #     Line2D([0], [0], color="gray", label="Test",
    #            **SPLIT_STYLE["test"],  linewidth=3, markersize=font_size * 0.4),
    # ]

    # Split color_entries evenly across col1 and col2
    half = math.ceil(len(color_entries) / 2)
    col1 = color_entries[:half]   # e.g. rows 0,1,2 of col1
    col2 = color_entries[half:]   # e.g. rows 0,1,2 of col2
    # col3 = split_entries          # rows 0,1,2 of col3

    # All columns must have the same number of rows (= max_len)
    # max_len = max(len(col1), len(col2), len(col3))
    max_len = max(len(col1), len(col2))
    invisible = Line2D([0], [0], color="none", label=" ")

    col1 = col1 + [invisible] * (max_len - len(col1))
    col2 = col2 + [invisible] * (max_len - len(col2))
    # col3 = col3 + [invisible] * (max_len - len(col3))

    # Interleave row by row: (col1[0], col2[0], col3[0], col1[1], col2[1], col3[1], ...)
    # With ncol=3, matplotlib fills left-to-right then top-to-bottom,
    # so this layout gives us exactly 3 columns and max_len rows.
    # interleaved = [h for row in zip(col1, col2, col3) for h in row]
    interleaved = [h for row in zip(col1, col2) for h in row]

    axs[0].legend(
        handles=interleaved,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        fontsize=font_size * 0.75,
        frameon=False,
        handletextpad=1.2,
        columnspacing=2.0,
        title="Model / Pos. Embed.",                                    
        title_fontsize=font_size * 0.75,
    )

    # --- Panel 1: eq class scatter for nh12_nl12 ---
    ax = axs[1]
    data = cluster_data_per_model.get("nh12_nl12", [])
    if not data:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=font_size)
        ax.set_title(f"{MODEL_LABELS['nh12_nl12']}\nEq. class distribution")
    else:
        _scatter_panel(ax, data, font_size,
                       title=f"{MODEL_LABELS['nh12_nl12']}\nEq. class distribution")

    out_path = os.path.join(
        save_dir, "leakage_experiment_diverse_k6_equivalence_class_disjoint7_nh12_nl12.pdf"
    )
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading disjoint7_6 results...")
    train_results, val_results, test_results, leakage_list = load_disjoint7_results()

    print("Loading CE metric cluster assignments...")
    unique_pairs, cluster_labels = _load_cec()

    cluster_data_per_model = {}
    for model_name in MODELS:
        print(f"Loading cluster data for {model_name}...")
        cluster_data_per_model[model_name] = load_cluster_data_for_model(
            model_name, unique_pairs, cluster_labels
        )

    print("Generating combined plot (line + 3× cluster scatter)...")
    plot_combined(train_results, val_results, test_results, leakage_list,
                  cluster_data_per_model, save_dir=ROOT_DIR)
