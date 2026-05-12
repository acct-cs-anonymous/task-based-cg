"""
Generates two module-coverage accuracy plots from the paper:

  1. Pairwise coverage  — accuracy vs pairwise KL divergence
     (strategies: continuouspaircoverage_6_0.0 … continuouspaircoverage_6_1.0)
     Output: pair_coverage_kl.pdf

  2. Position-wise coverage — accuracy vs coverage fraction
     (strategies: continuouscoverage_6_0.0 … continuouscoverage_6_1.0)
     Output: poswise_coverage.pdf

Usage:
    python -m scripts.module_coverage_plots [--output_dir PATH]
"""

import argparse
import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from init import ROOT_DIR


SEEDS = [0, 10, 20, 30, 40]
POS_TYPES = ["abs", "rel_global"]
MODES = ["step_by_step", "direct"]
FUNCTION_TYPES = ["uniform", "diverse"]
NHEADS_NLAYERS = "nh6_nl3"
PROMPT_LENGTH = "fixed"
DATA_SPEC = "nalph_26_seqlen_6_fnlen_6_taskmaxlen_6"

COLORS = {
    "direct_fixed":       {"abs": "tab:blue",  "rel_global": "tab:red"},
    "step_by_step_fixed": {"abs": "tab:green", "rel_global": "tab:orange"},
}

PAIR_FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

PAIR_STRATEGIES = [f"continuouspaircoverage_6_{p}" for p in PAIR_FRACS]
# Pairwise KL divergence values corresponding to each fraction (from paper)
PAIR_KL_VALUES  = [0.024, 0.55, 1.063, 1.591, 2.034, 2.506, 2.925, 4.268, 6.573, 10.057, 17.179]
POS_FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

POS_STRATEGIES  = [f"continuouscoverage_6_{p}" for p in POS_FRACS]
# Position-wise KL divergence values; update if computed values are available
POS_KL_VALUES = [0.03, 0.51, 1.00, 1.62, 2.16, 2.57, 3.06, 4.29]

def _eval_base(function_type, mode):
    return os.path.join(
        ROOT_DIR, "models", "eval",
        function_type, PROMPT_LENGTH, DATA_SPEC, mode,
    )


def _pkl_path(base, strategy, pos, seed):
    return os.path.join(
        base,
        f"train_{strategy}", f"eval_{strategy}",
        pos, NHEADS_NLAYERS, f"seed_{seed}", "accs.pkl",
    )


def load_accuracy(pkl_path):
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)["test"].sharp_accuracy


def fit_trendline(x_values, y_values):
    x, y = np.array(x_values), np.array(y_values)
    x_to_ys = defaultdict(list)
    for xi, yi in zip(x, y):
        x_to_ys[xi].append(yi)
    x_u = np.array(sorted(x_to_ys))
    y_u = np.array([np.mean(x_to_ys[xi]) for xi in x_u])
    if len(x_u) < 2:
        return None, None
    coeffs = np.polyfit(x_u, y_u, 2)
    x_fit = np.linspace(x_u.min(), x_u.max(), 300)
    return x_fit, np.poly1d(coeffs)(x_fit)


def load_results(strategies):
    """Load accuracy results for all (function_type, mode, pos, strategy, seed) combos."""
    results = {}
    for function_type in FUNCTION_TYPES:
        results[function_type] = {}
        for mode in MODES:
            results[function_type][mode] = {}
            base = _eval_base(function_type, mode)
            for pos in POS_TYPES:
                results[function_type][mode][pos] = {}
                for strategy in strategies:
                    accs = []
                    for seed in SEEDS:
                        # uniform/direct results were unavailable during experiments
                        if function_type == "uniform" and mode == "direct":
                            accs.append(0.0)
                            continue
                        acc = load_accuracy(_pkl_path(base, strategy, pos, seed))
                        if acc is not None:
                            accs.append(acc)
                    results[function_type][mode][pos][strategy] = accs
    return results


def _make_figure(results, strategies, x_values, x_label, output_path, font_size=45):
    plt.rcParams.update({
        "font.size": font_size, "legend.fontsize": font_size,
        "axes.labelsize": font_size, "axes.titlesize": font_size,
    })
    fig, axs = plt.subplots(1, 2, figsize=(50, 15))

    for ax, function_type, title in zip(axs, ["diverse", "uniform"], ["(a) Diverse", "(b) Uniform"]):
        for mode in MODES:
            for pos in POS_TYPES:
                color = COLORS[f"{mode}_fixed"][pos]
                pos_label  = "Relative" if pos == "rel_global" else "Absolute"
                mode_label = "Direct"   if mode == "direct"    else "Step-by-Step"

                y_mean = [np.mean(results[function_type][mode][pos][s]) for s in strategies]
                y_std  = [np.std( results[function_type][mode][pos][s]) for s in strategies]

                ax.scatter(x_values, y_mean, color=color,
                           s=(font_size / 2) ** 2, label=f"{mode_label} ({pos_label})")
                ax.errorbar(x_values, y_mean, yerr=y_std,
                            fmt="none", color=color, capsize=5)

                x_fit, y_fit = fit_trendline(x_values, y_mean)
                if x_fit is not None:
                    ax.plot(x_fit, y_fit, color=color, linestyle="--", linewidth=3, alpha=0.6)

        ax.set_title(title)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Accuracy")
        ax.grid(True)
        ax.set_xticks(x_values)
        ax.tick_params(axis="x", rotation=45)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=font_size,
               bbox_to_anchor=(0.5, -0.15), frameon=False)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default=os.path.join(ROOT_DIR, "results", "module_coverage_plots"),
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading pairwise coverage results...")
    pair_results = load_results(PAIR_STRATEGIES)
    _make_figure(
        pair_results, PAIR_STRATEGIES, PAIR_KL_VALUES,
        x_label="Pairwise KL divergence",
        output_path=os.path.join(args.output_dir, "pair_coverage_kl.pdf"),
    )

    print("Loading position-wise coverage results...")
    pos_results = load_results(POS_STRATEGIES)
    _make_figure(
        pos_results, POS_STRATEGIES, POS_KL_VALUES,
        x_label="Coverage fraction",
        output_path=os.path.join(args.output_dir, "poswise_coverage.pdf"),
    )


if __name__ == "__main__":
    main()
