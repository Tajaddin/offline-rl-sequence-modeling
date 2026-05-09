"""
Final comparison script — generates summary tables and plots for all models.

Reads saved training histories and evaluation results to produce:
  1. Training loss comparison plot
  2. D4RL score comparison bar chart
  3. Summary table (printed and saved)

Usage:
  python compare.py --out_dir outputs
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="outputs")
    args = parser.parse_args()

    # Load histories
    bc_hist = load_json(os.path.join(args.out_dir, "bc_history.json"))
    dt_hist = load_json(os.path.join(args.out_dir, "dt_history.json"))
    odt_hist = load_json(os.path.join(args.out_dir, "odt_history.json"))
    online_hist = load_json(os.path.join(args.out_dir, "online_history.json"))
    foundation = load_json(os.path.join(args.out_dir, "foundation", "foundation_comparison.json"))
    improvement = load_json(os.path.join(args.out_dir, "improved", "improvement_summary.json"))

    # ── Figure 1: Training Loss Curves ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    if bc_hist:
        axes[0].plot(bc_hist["train_loss"], label="Train", color="blue")
        axes[0].plot(bc_hist["val_loss"], label="Val", color="orange")
        axes[0].set_title("Behavior Cloning (Baseline)")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("MSE Loss")
        axes[0].legend()

    if dt_hist:
        axes[1].plot(dt_hist["train_loss"], label="Train", color="blue")
        axes[1].plot(dt_hist["val_loss"], label="Val", color="orange")
        axes[1].set_title("Decision Transformer")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MSE Loss")
        axes[1].legend()

    if odt_hist:
        axes[2].plot(odt_hist["train_loss"], label="Train", color="blue")
        axes[2].plot(odt_hist["val_loss"], label="Val", color="orange")
        axes[2].set_title("Online Decision Transformer")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("NLL + Entropy Loss")
        axes[2].legend()

    plt.suptitle("Training Loss Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "all_loss_curves.png"), dpi=150)
    plt.close()
    print("Saved: all_loss_curves.png")

    # ── Figure 2: D4RL Score Progression ──
    fig, ax = plt.subplots(figsize=(10, 6))

    eval_every = 5
    max_epochs = 50

    if bc_hist and bc_hist["eval_normalized"]:
        epochs = list(range(eval_every, max_epochs + 1, eval_every))[:len(bc_hist["eval_normalized"])]
        ax.plot(epochs, bc_hist["eval_normalized"], "s-", label="BC (Baseline)",
                color="gray", markersize=8, linewidth=2)

    if dt_hist and dt_hist["eval_normalized"]:
        epochs = list(range(eval_every, max_epochs + 1, eval_every))[:len(dt_hist["eval_normalized"])]
        ax.plot(epochs, dt_hist["eval_normalized"], "o-", label="Decision Transformer",
                color="blue", markersize=8, linewidth=2)

    if odt_hist and odt_hist["eval_normalized"]:
        epochs = list(range(eval_every, max_epochs + 1, eval_every))[:len(odt_hist["eval_normalized"])]
        ax.plot(epochs, odt_hist["eval_normalized"], "^-", label="ODT (Offline Phase)",
                color="green", markersize=8, linewidth=2)

    # add online phase
    if online_hist and online_hist.get("online_normalized"):
        online_x = list(range(max_epochs + 1,
                              max_epochs + 1 + len(online_hist["online_normalized"])))
        ax.plot(online_x, online_hist["online_normalized"], "^--",
                label="ODT (Online Fine-tuning)", color="red", markersize=8, linewidth=2)
        ax.axvline(x=max_epochs + 0.5, color="red", linestyle=":", alpha=0.5)
        ax.text(max_epochs + 1, ax.get_ylim()[0] + 1, "Online\nPhase",
                fontsize=9, color="red", alpha=0.7)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("D4RL Normalized Score", fontsize=12)
    ax.set_title("D4RL Score Progression During Training", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "d4rl_progression.png"), dpi=150)
    plt.close()
    print("Saved: d4rl_progression.png")

    # ── Figure 3: Final comparison bar chart ──
    models = []
    scores = []
    colors = []

    if bc_hist and bc_hist["eval_normalized"]:
        models.append("BC\n(Baseline)")
        scores.append(max(bc_hist["eval_normalized"]))
        colors.append("gray")

    if dt_hist and dt_hist["eval_normalized"]:
        models.append("DT\n(Proposed)")
        scores.append(max(dt_hist["eval_normalized"]))
        colors.append("steelblue")

    if odt_hist and odt_hist["eval_normalized"]:
        models.append("ODT\n(Offline)")
        scores.append(max(odt_hist["eval_normalized"]))
        colors.append("seagreen")

    if online_hist and online_hist.get("online_normalized"):
        models.append("ODT\n(Online)")
        scores.append(max(online_hist["online_normalized"]))
        colors.append("indianred")

    if foundation:
        for name, data in foundation.items():
            if "Foundation" in name:
                models.append("Foundation\n(HF DT)")
                scores.append(data["d4rl_score"])
                colors.append("goldenrod")

    if improvement:
        for tech_name, data in improvement.items():
            label = tech_name.replace("_", "\n")
            models.append(f"DT+\n{label}")
            scores.append(data["best_d4rl_score"])
            colors.append("mediumpurple")

    if models:
        fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.5), 6))
        bars = ax.bar(models, scores, color=colors, edgecolor="black", linewidth=0.5)
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{score:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
        ax.set_ylabel("D4RL Normalized Score", fontsize=12)
        ax.set_title("Model Comparison — Best D4RL Scores", fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "model_comparison_bar.png"), dpi=150)
        plt.close()
        print("Saved: model_comparison_bar.png")

    # ── Print summary table ──
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Model':<30} | {'Best D4RL Score':>15}")
    print("-" * 50)

    if bc_hist and bc_hist["eval_normalized"]:
        print(f"{'BC (Baseline)':<30} | {max(bc_hist['eval_normalized']):>15.1f}")
    if dt_hist and dt_hist["eval_normalized"]:
        print(f"{'Decision Transformer':<30} | {max(dt_hist['eval_normalized']):>15.1f}")
    if odt_hist and odt_hist["eval_normalized"]:
        print(f"{'ODT (Offline Only)':<30} | {max(odt_hist['eval_normalized']):>15.1f}")
    if online_hist and online_hist.get("online_normalized"):
        print(f"{'ODT (After Online FT)':<30} | {max(online_hist['online_normalized']):>15.1f}")
    if foundation:
        for name, data in foundation.items():
            print(f"{name:<30} | {data['d4rl_score']:>15.1f}")
    if improvement:
        for name, data in improvement.items():
            print(f"{'DT + ' + name:<30} | {data['best_d4rl_score']:>15.1f}")

    print("=" * 50)
    print(f"\nAll plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
