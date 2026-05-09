"""
Performance Improvement Techniques for Decision Transformer.

Implements and evaluates 3 techniques (rubric section C, 15 points):
  Technique 1: State & Return-to-Go Normalization
  Technique 2: Learning Rate Warm-up + Cosine Decay
  Technique 3: Increased Context Length (K=30 vs K=20)

Each technique is trained independently so its effect can be isolated.
All tensor operations run on GPU (--device cuda:0).

Usage:
  python train_improved.py --device cuda:0 --out_dir outputs/improved
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.model import DecisionTransformer
from src.dataloader import (
    load_d4rl_dataset, segment_into_trajectories, ENV_META, get_data_stats,
)
from src.utils import set_seed, evaluate_dt, d4rl_normalized_score
from train import NumpyEncoder


# ────────────────────────────────────────────────────────────────
# Normalized DT Dataset (Technique 1)
# ────────────────────────────────────────────────────────────────

class NormalizedDTDataset(torch.utils.data.Dataset):
    """DT dataset with state and return-to-go normalization.

    Technique 1: Normalizing inputs stabilizes training and helps the
    transformer attend to meaningful patterns rather than raw magnitudes.
    """

    def __init__(self, trajectories, context_len=20,
                 state_mean=None, state_std=None, rtg_scale=None):
        self.context_len = context_len
        self.trajectories = trajectories

        # compute normalization stats from data
        all_states = np.concatenate([t["observations"] for t in trajectories])
        self.state_mean = state_mean if state_mean is not None else all_states.mean(axis=0)
        self.state_std = state_std if state_std is not None else all_states.std(axis=0) + 1e-6

        all_rtgs = np.concatenate([t["returns_to_go"] for t in trajectories])
        self.rtg_scale = rtg_scale if rtg_scale is not None else np.abs(all_rtgs).max()
        if self.rtg_scale < 1e-6:
            self.rtg_scale = 1.0

        self.indices = []
        for i, traj in enumerate(trajectories):
            for start in range(len(traj["observations"])):
                self.indices.append((i, start))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        traj_idx, start = self.indices[idx]
        traj = self.trajectories[traj_idx]

        end = start + 1
        begin = max(0, end - self.context_len)

        states = traj["observations"][begin:end].astype(np.float32)
        actions = traj["actions"][begin:end].astype(np.float32)
        rtg = traj["returns_to_go"][begin:end].astype(np.float32).reshape(-1, 1)
        timesteps = np.arange(begin, end, dtype=np.int64)

        # normalize
        states = (states - self.state_mean) / self.state_std
        rtg = rtg / self.rtg_scale

        seq_len = states.shape[0]
        pad_len = self.context_len - seq_len
        if pad_len > 0:
            states = np.concatenate([np.zeros((pad_len, states.shape[1]), dtype=np.float32), states])
            actions = np.concatenate([np.zeros((pad_len, actions.shape[1]), dtype=np.float32), actions])
            rtg = np.concatenate([np.zeros((pad_len, 1), dtype=np.float32), rtg])
            timesteps = np.concatenate([np.zeros(pad_len, dtype=np.int64), timesteps])

        return (
            torch.tensor(states), torch.tensor(actions),
            torch.tensor(rtg), torch.tensor(timesteps),
        )


# ────────────────────────────────────────────────────────────────
# Warm-up + Cosine scheduler (Technique 2)
# ────────────────────────────────────────────────────────────────

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warm-up for first `warmup_steps` then cosine decay.

    Technique 2: Prevents large gradient updates in early training when
    weights are random, reducing training instability.
    """

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            factor = step / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            factor = max(self.min_lr / self.base_lrs[0],
                         0.5 * (1.0 + np.cos(np.pi * progress)))
        return [base_lr * factor for base_lr in self.base_lrs]


# ────────────────────────────────────────────────────────────────
# Training function
# ────────────────────────────────────────────────────────────────

def train_dt_improved(args, device, technique_name, context_len, use_norm, use_warmup):
    """Train DT with specified improvement technique(s)."""
    meta = ENV_META[args.dataset]
    state_dim, act_dim = meta["state_dim"], meta["act_dim"]

    data = load_d4rl_dataset(args.data_root, args.dataset)
    trajectories = segment_into_trajectories(data)
    stats = get_data_stats(trajectories)

    print(f"\n{'='*60}")
    print(f"  Technique: {technique_name}")
    print(f"  Context length: {context_len}, Normalization: {use_norm}, Warmup: {use_warmup}")
    print(f"{'='*60}")
    print(f"Dataset: {stats['num_trajectories']} trajectories, {stats['num_transitions']} transitions")

    # dataset
    norm_stats = {}
    if use_norm:
        dataset = NormalizedDTDataset(trajectories, context_len=context_len)
        norm_stats = {
            "state_mean": dataset.state_mean,
            "state_std": dataset.state_std,
            "rtg_scale": float(dataset.rtg_scale),
        }
    else:
        from src.dataloader import DTDataset
        dataset = DTDataset(trajectories, context_len=context_len)

    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    model = DecisionTransformer(
        state_dim, act_dim,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        max_len=context_len, dropout=args.dropout,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    if use_warmup:
        total_steps = args.epochs * len(train_loader)
        warmup_steps = int(0.1 * total_steps)
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
        step_per_batch = True
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
        step_per_batch = False

    history = {"train_loss": [], "val_loss": [], "eval_return": [], "eval_normalized": []}
    best_val = float("inf")
    tag = technique_name.lower().replace(" ", "_").replace("+", "_")

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for states, actions, rtgs, timesteps in tqdm(
                train_loader, desc=f"[{technique_name}] Epoch {epoch}/{args.epochs}", leave=False):
            states = states.to(device)
            actions = actions.to(device)
            rtgs = rtgs.to(device)
            timesteps = timesteps.to(device)

            pred = model(states, actions, rtgs, timesteps)
            loss = nn.functional.mse_loss(pred, actions)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if step_per_batch:
                scheduler.step()
            losses.append(loss.item())

        if not step_per_batch:
            scheduler.step()
        train_loss = np.mean(losses)
        history["train_loss"].append(train_loss)

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for states, actions, rtgs, timesteps in val_loader:
                states, actions = states.to(device), actions.to(device)
                rtgs, timesteps = rtgs.to(device), timesteps.to(device)
                pred = model(states, actions, rtgs, timesteps)
                val_losses.append(nn.functional.mse_loss(pred, actions).item())
        val_loss = np.mean(val_losses)
        history["val_loss"].append(val_loss)

        print(f"[Epoch {epoch}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            save_data = {"model_state": model.state_dict()}
            if use_norm:
                save_data["state_mean"] = norm_stats["state_mean"]
                save_data["state_std"] = norm_stats["state_std"]
                save_data["rtg_scale"] = norm_stats["rtg_scale"]
            torch.save(save_data, os.path.join(args.out_dir, f"best_dt_{tag}.pth"))

        if epoch % args.eval_every == 0:
            # for normalized models, we need to wrap the evaluation
            if use_norm:
                mean_ret, std_ret = evaluate_dt_normalized(
                    model, meta["env_name"], device,
                    target_return=args.target_return / norm_stats["rtg_scale"],
                    state_mean=norm_stats["state_mean"],
                    state_std=norm_stats["state_std"],
                    num_episodes=args.eval_episodes,
                    max_ep_len=1000, context_len=context_len,
                    state_dim=state_dim, act_dim=act_dim,
                )
            else:
                mean_ret, std_ret = evaluate_dt(
                    model, meta["env_name"], device,
                    target_return=args.target_return,
                    num_episodes=args.eval_episodes,
                    max_ep_len=1000, context_len=context_len,
                    state_dim=state_dim, act_dim=act_dim,
                )
            norm_score = d4rl_normalized_score(args.dataset, mean_ret)
            history["eval_return"].append(float(mean_ret))
            history["eval_normalized"].append(float(norm_score))
            print(f"  -> Eval return: {mean_ret:.1f} +/- {std_ret:.1f}  "
                  f"(D4RL score: {norm_score:.1f})")

    with open(os.path.join(args.out_dir, f"{tag}_history.json"), "w") as f:
        json.dump(history, f, indent=2, cls=NumpyEncoder)

    return history


# ────────────────────────────────────────────────────────────────
# Normalized evaluation helper
# ────────────────────────────────────────────────────────────────

def evaluate_dt_normalized(model, env_name, device, target_return,
                           state_mean, state_std,
                           num_episodes=10, max_ep_len=1000,
                           context_len=20, state_dim=11, act_dim=3):
    """Evaluate DT that was trained with normalized inputs."""
    import gymnasium as gym

    model.eval()
    returns = []
    s_mean = torch.tensor(state_mean, dtype=torch.float32, device=device)
    s_std = torch.tensor(state_std, dtype=torch.float32, device=device)

    for _ in range(num_episodes):
        env = gym.make(env_name)
        state, _ = env.reset()

        states = torch.zeros(1, context_len, state_dim, device=device)
        actions = torch.zeros(1, context_len, act_dim, device=device)
        rtgs = torch.zeros(1, context_len, 1, device=device)
        timesteps = torch.zeros(1, context_len, dtype=torch.long, device=device)

        state_t = torch.tensor(state, device=device, dtype=torch.float32)
        states[0, 0] = (state_t - s_mean) / s_std
        rtgs[0, 0] = target_return

        ep_return = 0.0
        for t in range(max_ep_len):
            idx = min(t, context_len - 1)

            with torch.no_grad():
                action = model.get_action(
                    states[:, :idx + 1], actions[:, :idx + 1],
                    rtgs[:, :idx + 1], timesteps[:, :idx + 1],
                )

            action_np = action.squeeze(0).cpu().numpy()
            action_np = np.clip(action_np, -1.0, 1.0)
            state, reward, terminated, truncated, _ = env.step(action_np)
            ep_return += reward

            if terminated or truncated:
                break

            next_t = min(t + 1, context_len - 1)
            state_t = torch.tensor(state, device=device, dtype=torch.float32)
            if t + 1 < context_len:
                states[0, t + 1] = (state_t - s_mean) / s_std
                actions[0, t] = action.squeeze(0)
                rtgs[0, t + 1] = rtgs[0, t] - reward
                timesteps[0, t + 1] = t + 1
            else:
                states = torch.roll(states, -1, dims=1)
                actions = torch.roll(actions, -1, dims=1)
                rtgs = torch.roll(rtgs, -1, dims=1)
                timesteps = torch.roll(timesteps, -1, dims=1)
                states[0, -1] = (state_t - s_mean) / s_std
                actions[0, -2] = action.squeeze(0)
                rtgs[0, -1] = rtgs[0, -2] - reward
                timesteps[0, -1] = t + 1

        env.close()
        returns.append(ep_return)

    model.train()
    return np.mean(returns), np.std(returns)


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train DT with improvement techniques")
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--dataset", type=str, default="hopper-medium-replay-v2")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=0.25)
    p.add_argument("--target_return", type=float, default=3600.0)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--out_dir", type=str, default="outputs/improved")
    p.add_argument("--technique", type=str, default="all",
                   choices=["1", "2", "3", "all"],
                   help="Which technique to run: 1=normalization, 2=warmup, 3=context_len, all=all three")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    results = {}

    # Technique 1: State & RTG Normalization
    if args.technique in ["1", "all"]:
        print("\n" + "=" * 60)
        print("TECHNIQUE 1: State & Return-to-Go Normalization")
        print("=" * 60)
        h1 = train_dt_improved(args, device,
                               technique_name="Normalization",
                               context_len=20, use_norm=True, use_warmup=False)
        results["normalization"] = h1

    # Technique 2: Learning Rate Warm-up
    if args.technique in ["2", "all"]:
        print("\n" + "=" * 60)
        print("TECHNIQUE 2: Learning Rate Warm-up + Cosine Decay")
        print("=" * 60)
        h2 = train_dt_improved(args, device,
                               technique_name="LR Warmup",
                               context_len=20, use_norm=False, use_warmup=True)
        results["lr_warmup"] = h2

    # Technique 3: Increased Context Length (K=30)
    if args.technique in ["3", "all"]:
        print("\n" + "=" * 60)
        print("TECHNIQUE 3: Increased Context Length (K=30)")
        print("=" * 60)
        h3 = train_dt_improved(args, device,
                               technique_name="Context K=30",
                               context_len=30, use_norm=False, use_warmup=False)
        results["context_k30"] = h3

    # ── Summary comparison ──
    print("\n" + "=" * 60)
    print("IMPROVEMENT TECHNIQUES SUMMARY")
    print("=" * 60)
    print(f"{'Technique':<30} | {'Best D4RL Score':>15} | {'Final Val Loss':>14}")
    print("-" * 65)

    # baseline reference
    baseline_path = os.path.join("outputs", "dt_history.json")
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            bh = json.load(f)
        baseline_best = max(bh["eval_normalized"]) if bh["eval_normalized"] else 0
        baseline_val = bh["val_loss"][-1] if bh["val_loss"] else 0
        print(f"{'Baseline DT (no improvements)':<30} | {baseline_best:>15.1f} | {baseline_val:>14.4f}")

    for name, hist in results.items():
        best_score = max(hist["eval_normalized"]) if hist["eval_normalized"] else 0
        final_val = hist["val_loss"][-1] if hist["val_loss"] else 0
        print(f"{name:<30} | {best_score:>15.1f} | {final_val:>14.4f}")

    # save summary
    summary_path = os.path.join(args.out_dir, "improvement_summary.json")
    summary = {}
    for name, hist in results.items():
        summary[name] = {
            "best_d4rl_score": max(hist["eval_normalized"]) if hist["eval_normalized"] else 0,
            "final_val_loss": hist["val_loss"][-1] if hist["val_loss"] else 0,
            "all_eval_scores": hist["eval_normalized"],
        }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    # ── Generate comparison plot ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            bh = json.load(f)
        axes[0].plot(bh["val_loss"], label="Baseline DT", linestyle="--", color="gray")

    colors = {"normalization": "blue", "lr_warmup": "green", "context_k30": "red"}
    labels = {"normalization": "T1: Normalization", "lr_warmup": "T2: LR Warmup", "context_k30": "T3: Context K=30"}
    for name, hist in results.items():
        axes[0].plot(hist["val_loss"], label=labels.get(name, name), color=colors.get(name, "black"))
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation Loss")
    axes[0].set_title("Validation Loss Comparison")
    axes[0].legend()

    # D4RL scores
    eval_epochs = list(range(args.eval_every, args.epochs + 1, args.eval_every))
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            bh = json.load(f)
        be = eval_epochs[:len(bh["eval_normalized"])]
        axes[1].plot(be, bh["eval_normalized"], "o--", label="Baseline DT", color="gray")

    for name, hist in results.items():
        ep = eval_epochs[:len(hist["eval_normalized"])]
        axes[1].plot(ep, hist["eval_normalized"], "o-",
                     label=labels.get(name, name), color=colors.get(name, "black"))
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("D4RL Normalized Score")
    axes[1].set_title("D4RL Score Comparison")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "improvement_comparison.png"), dpi=150)
    plt.close()
    print(f"\nPlot saved to {args.out_dir}/improvement_comparison.png")


if __name__ == "__main__":
    main()
