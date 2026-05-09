"""
Final BC chart with all 4 training seeds (1, 7, 13, 42).
  Left  – train/val loss with per-seed lines + mean line
  Right – per-epoch D4RL per seed + final headline mean across 4 seeds
"""
import json, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

SEEDS = [1, 7, 13, 42]
HIST_PATHS = {
    1: "outputs/bc_seed1/bc_history.json",
    7: "outputs/bc_seed7/bc_history.json",
    13: "outputs/bc_seed13/bc_history.json",
    42: "outputs/bc_history.json",  # original seed-42 run
}
EVAL_PATHS = {
    1: "outputs/bc_seed1_eval.json",
    7: "outputs/bc_seed7_eval.json",
    13: "outputs/bc_seed13_eval.json",
    42: "outputs/bc_final_eval.json",  # 8-seed x 25-ep on bc_epoch50
}

# load
hists, evals = {}, {}
for s in SEEDS:
    with open(HIST_PATHS[s]) as f:
        hists[s] = json.load(f)
    with open(EVAL_PATHS[s]) as f:
        evals[s] = json.load(f)

# stack loss curves
train = np.array([hists[s]["train_loss"] for s in SEEDS])
val = np.array([hists[s]["val_loss"] for s in SEEDS])
epochs = np.arange(1, train.shape[1] + 1)

# per-epoch D4RL (single-seed eval during training, 1 value every 5 epochs)
eval_epochs = np.arange(5, 5 * train.shape[1] // 5 + 1, 5)[:len(hists[SEEDS[0]]["eval_normalized"])]
d4rl_per_seed = np.array([hists[s]["eval_normalized"] for s in SEEDS])

# headline cross-seed aggregate at ep50
ckpt_means = np.array([evals[s]["d4rl_mean"] for s in SEEDS])
ckpt_stds = np.array([evals[s]["d4rl_std"] for s in SEEDS])
agg_mean = ckpt_means.mean()
agg_std = ckpt_means.std(ddof=1)
print(f"Cross-seed headline: D4RL {agg_mean:.2f} +/- {agg_std:.2f} "
      f"(seed values: {ckpt_means.round(2).tolist()})")

# colors for seeds
colors = {1: "#1f77b4", 7: "#2ca02c", 13: "#ff7f0e", 42: "#d62728"}

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
plt.rcParams.update({"font.size": 11})

# ── Left: loss curves per seed ─────────────────────────────
ax = axes[0]
for s in SEEDS:
    ax.plot(epochs, hists[s]["train_loss"], color=colors[s], lw=1.6,
            alpha=0.55, label=f"seed {s} train")
    ax.plot(epochs, hists[s]["val_loss"], color=colors[s], lw=1.6,
            ls="--", alpha=0.85, label=f"seed {s} val")
# mean over all 4 seeds
ax.plot(epochs, train.mean(0), color="black", lw=2.4, label="mean train")
ax.plot(epochs, val.mean(0), color="black", lw=2.4, ls="--",
        label="mean val")
ax.fill_between(epochs, train.mean(0) - train.std(0),
                train.mean(0) + train.std(0), color="black", alpha=0.08)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (MSE)")
ax.set_title("BC Training — Loss Curves (4 seeds)",
             fontsize=12, weight="bold")
ax.legend(loc="upper right", fontsize=8, ncol=2, frameon=False)
ax.grid(alpha=0.3)
ax.annotate(f"Mean final train: {train[:,-1].mean():.3f}\n"
            f"Mean final val:   {val[:,-1].mean():.3f}",
            xy=(50, val[:,-1].mean()), xytext=(28, 0.16),
            fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff8dc",
                      ec="#999", alpha=0.9))

# ── Right: per-epoch D4RL per seed ─────────────────────────
ax = axes[1]
for s in SEEDS:
    ax.plot(eval_epochs, hists[s]["eval_normalized"], "o-",
            color=colors[s], lw=1.8, ms=6, alpha=0.85,
            label=f"seed {s}")
ax.set_xlabel("Epoch")
ax.set_ylabel("D4RL Normalized Score")
ax.set_title("BC Evaluation — D4RL per Training Seed",
             fontsize=12, weight="bold")
ax.set_ylim(-3, 75)
ax.set_xlim(2, 62)
ax.grid(alpha=0.3)

# headline callout - star at ep55 to the right of training data
ax.errorbar([55], [agg_mean], yerr=[agg_std],
            fmt="*", color="black", ms=20, capsize=6, lw=2,
            label="Cross-seed final ± std")
ax.legend(loc="upper right", fontsize=8.5, frameon=True, ncol=2)
ax.text(
    3, 5,
    f"4 train seeds × 100 eval rollouts each\n"
    f"D4RL {agg_mean:.1f} ± {agg_std:.1f}    "
    f"(best: {ckpt_means.max():.1f}, worst: {ckpt_means.min():.1f})",
    fontsize=9.5, weight="bold", va="bottom",
    bbox=dict(boxstyle="round,pad=0.4", fc="#ffe4e1", ec="#d62728",
              alpha=0.95))

plt.tight_layout()
plt.savefig("outputs/bc_curves_4seed.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved -> outputs/bc_curves_4seed.png")

# also write a summary JSON
summary = {
    "training_seeds": SEEDS,
    "per_seed_eval_at_epoch50": {
        str(s): {"d4rl_mean": float(evals[s]["d4rl_mean"]),
                 "d4rl_std": float(evals[s]["d4rl_std"]),
                 "return_mean": float(evals[s]["return_mean"]),
                 "return_std": float(evals[s]["return_std"])}
        for s in SEEDS
    },
    "cross_seed_d4rl_mean": float(agg_mean),
    "cross_seed_d4rl_std": float(agg_std),
    "cross_seed_d4rl_min": float(ckpt_means.min()),
    "cross_seed_d4rl_max": float(ckpt_means.max()),
    "final_train_loss_mean": float(train[:,-1].mean()),
    "final_train_loss_std": float(train[:,-1].std(ddof=1)),
    "final_val_loss_mean": float(val[:,-1].mean()),
    "final_val_loss_std": float(val[:,-1].std(ddof=1)),
}
with open("outputs/bc_4seed_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Saved -> outputs/bc_4seed_summary.json")
