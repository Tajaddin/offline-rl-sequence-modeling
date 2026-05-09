"""
Build the final BC results chart:
  Left  – train/val loss curves (50 epochs, single seed)
  Right – per-epoch D4RL across multiple eval seeds (mean + min/max band)

Sources:
  outputs/bc_history.json          (loss curves + single-seed per-epoch eval)
  outputs/bc_checkpoint_scan.json  (multi-seed eval at ep10/20/30/40/50)
  outputs/bc_final_eval.json       (headline 8-seed x 25-ep on best ckpt)
"""
import json, numpy as np, matplotlib.pyplot as plt

with open("outputs/bc_history.json") as f:
    hist = json.load(f)
with open("outputs/bc_checkpoint_scan.json") as f:
    scan = json.load(f)
with open("outputs/bc_final_eval.json") as f:
    final = json.load(f)

# Loss curves
train_loss = hist["train_loss"]
val_loss = hist["val_loss"]
epochs = list(range(1, len(train_loss) + 1))

# Single-seed per-epoch D4RL during training (one value every 5 epochs)
train_eval_epochs = list(range(5, 5 * len(hist["eval_normalized"]) + 1, 5))
train_eval_d4rl = hist["eval_normalized"]

# Multi-seed D4RL at ep10/20/30/40/50 (5 seeds × 10 eps)
ckpt_to_epoch = {f"outputs/bc_epoch{e}.pth": e for e in [10, 20, 30, 40, 50]}
ms_eps, ms_mean, ms_min, ms_max = [], [], [], []
for ckpt, ep in ckpt_to_epoch.items():
    r = scan["results"][ckpt]
    ms_eps.append(ep)
    ms_mean.append(r["d4rl_mean"])
    ms_min.append(r["d4rl_min"])
    ms_max.append(r["d4rl_max"])
order = np.argsort(ms_eps)
ms_eps = np.array(ms_eps)[order]
ms_mean = np.array(ms_mean)[order]
ms_min = np.array(ms_min)[order]
ms_max = np.array(ms_max)[order]

# Plot
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
plt.rcParams.update({"font.size": 11})

# ── Left: loss curves ─────────────────────────────────────────
ax = axes[0]
ax.plot(epochs, train_loss, color="#1f77b4", lw=2, label="Train MSE")
ax.plot(epochs, val_loss, color="#d62728", lw=2, label="Val MSE")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (MSE)")
ax.set_title("BC Training — Loss Curves (50 epochs, seed 42)",
             fontsize=12, weight="bold")
ax.legend(loc="upper right", frameon=False)
ax.grid(alpha=0.3)
ax.annotate(f"Final train: {train_loss[-1]:.3f}\n"
            f"Final val: {val_loss[-1]:.3f}",
            xy=(epochs[-1], val_loss[-1]),
            xytext=(30, 0.15), fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff8dc",
                      ec="#999", alpha=0.9))

# ── Right: D4RL evaluation ─────────────────────────────────────
ax = axes[1]
ax.plot(train_eval_epochs, train_eval_d4rl, "o--", color="#999",
        alpha=0.7, lw=1.2, ms=5, label="Single-seed eval (10 eps)")
ax.fill_between(ms_eps, ms_min, ms_max, color="#1f77b4", alpha=0.18,
                label="Multi-seed range (5 seeds × 10 eps)")
ax.plot(ms_eps, ms_mean, "o-", color="#1f77b4", lw=2.5, ms=8,
        label="Multi-seed mean")
# Headline final eval marker
ax.errorbar([50], [final["d4rl_mean"]],
            yerr=[final["d4rl_std"]],
            fmt="*", color="#d62728", ms=18, capsize=6, lw=2,
            label=f"Final test ({len(final['seeds'])} seeds × "
                  f"{final['episodes_per_seed']} eps)")
ax.set_xlabel("Checkpoint epoch")
ax.set_ylabel("D4RL Normalized Score")
ax.set_title("BC Evaluation — D4RL Score Across Training",
             fontsize=12, weight="bold")
ax.set_ylim(-2, 60)
ax.legend(loc="lower right", fontsize=9, frameon=True)
ax.grid(alpha=0.3)

ax.annotate(
    f"Headline: D4RL {final['d4rl_mean']:.2f} ± {final['d4rl_std']:.2f}\n"
    f"(200 rollouts, range {final['d4rl_min']:.1f}–{final['d4rl_max']:.1f})",
    xy=(50, final["d4rl_mean"]),
    xytext=(13, 55), fontsize=10, weight="bold",
    bbox=dict(boxstyle="round,pad=0.4", fc="#ffe4e1", ec="#d62728",
              alpha=0.95),
    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.4))

plt.tight_layout()
plt.savefig("outputs/bc_curves_final.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved -> outputs/bc_curves_final.png")
