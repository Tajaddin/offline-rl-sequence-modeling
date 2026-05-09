"""
Training script for all three models:
  - bc   : Behavior Cloning (baseline)
  - dt   : Decision Transformer
  - odt  : Online Decision Transformer

All tensor operations run on GPU (--device cuda:0).

Usage:
  python train.py --model bc  --epochs 50 --device cuda:0
  python train.py --model dt  --epochs 50 --device cuda:0
  python train.py --model odt --epochs 50 --online_epochs 10 --device cuda:0
"""

import os
import argparse
import json


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.model import BehaviorCloning, DecisionTransformer, OnlineDecisionTransformer
from src.dataloader import (
    load_d4rl_dataset, segment_into_trajectories, BCDataset, DTDataset,
    get_data_stats, ENV_META,
)
from src.utils import (
    set_seed, evaluate_bc, evaluate_dt, d4rl_normalized_score,
    collect_online_rollout,
)


# ────────────────────────────────────────────────────────────────
# Argument parsing
# ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train offline RL models")
    p.add_argument("--model", type=str, default="dt",
                   choices=["bc", "dt", "odt"])
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--dataset", type=str, default="hopper-medium-replay-v2")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)

    # architecture
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--context_len", type=int, default=20)
    p.add_argument("--dropout", type=float, default=0.1)

    # training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=0.25)
    p.add_argument("--val_split", type=float, default=0.1)

    # ODT online fine-tuning
    p.add_argument("--online_epochs", type=int, default=10,
                   help="Online fine-tuning epochs (ODT only)")
    p.add_argument("--online_rollouts", type=int, default=10,
                   help="Rollouts per online epoch")
    p.add_argument("--entropy_weight", type=float, default=0.01)
    p.add_argument("--target_return", type=float, default=3600.0,
                   help="Target return-to-go for DT/ODT evaluation")

    # evaluation
    p.add_argument("--eval_every", type=int, default=5,
                   help="Evaluate in env every N epochs")
    p.add_argument("--eval_episodes", type=int, default=10)

    # output
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--save_every", type=int, default=10)

    return p.parse_args()


# ────────────────────────────────────────────────────────────────
# Training loops
# ────────────────────────────────────────────────────────────────

def train_bc(args, device):
    """Train the Behavior Cloning baseline."""
    meta = ENV_META[args.dataset]
    state_dim, act_dim = meta["state_dim"], meta["act_dim"]

    # data
    data = load_d4rl_dataset(args.data_root, args.dataset)
    trajectories = segment_into_trajectories(data)
    stats = get_data_stats(trajectories)
    print(f"Dataset: {stats['num_trajectories']} trajectories, "
          f"{stats['num_transitions']} transitions")
    print(f"Return: {stats['avg_return']:.1f} +/- {stats['std_return']:.1f} "
          f"(min={stats['min_return']:.1f}, max={stats['max_return']:.1f})")

    dataset = BCDataset(trajectories, context_len=args.context_len)
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    # model
    model = BehaviorCloning(
        state_dim, act_dim,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        max_len=args.context_len, dropout=args.dropout,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # training
    history = {"train_loss": [], "val_loss": [], "eval_return": [], "eval_normalized": []}
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for states, actions in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}",
                                    leave=False):
            states, actions = states.to(device), actions.to(device)
            pred = model(states)
            loss = nn.functional.mse_loss(pred, actions)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()
        train_loss = np.mean(losses)
        history["train_loss"].append(train_loss)

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for states, actions in val_loader:
                states, actions = states.to(device), actions.to(device)
                pred = model(states)
                val_losses.append(nn.functional.mse_loss(pred, actions).item())
        val_loss = np.mean(val_losses)
        history["val_loss"].append(val_loss)

        print(f"[Epoch {epoch}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        # save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(),
                       os.path.join(args.out_dir, "best_model_bc.pth"))

        # periodic checkpoint
        if epoch % args.save_every == 0:
            torch.save(model.state_dict(),
                       os.path.join(args.out_dir, f"bc_epoch{epoch}.pth"))

        # environment evaluation
        if epoch % args.eval_every == 0:
            mean_ret, std_ret = evaluate_bc(
                model, meta["env_name"], device,
                num_episodes=args.eval_episodes,
                max_ep_len=1000, context_len=args.context_len,
                state_dim=state_dim,
            )
            norm = d4rl_normalized_score(args.dataset, mean_ret)
            history["eval_return"].append(mean_ret)
            history["eval_normalized"].append(norm)
            print(f"  → Eval return: {mean_ret:.1f} +/- {std_ret:.1f}  "
                  f"(D4RL score: {norm:.1f})")

    save_results(args, history, "bc")
    return model


def train_dt(args, device, model_type="dt"):
    """Train Decision Transformer or Online Decision Transformer (offline phase)."""
    meta = ENV_META[args.dataset]
    state_dim, act_dim = meta["state_dim"], meta["act_dim"]

    # data
    data = load_d4rl_dataset(args.data_root, args.dataset)
    trajectories = segment_into_trajectories(data)
    stats = get_data_stats(trajectories)
    print(f"Dataset: {stats['num_trajectories']} trajectories, "
          f"{stats['num_transitions']} transitions")
    print(f"Return: {stats['avg_return']:.1f} +/- {stats['std_return']:.1f}")

    dataset = DTDataset(trajectories, context_len=args.context_len)
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    # model
    ModelClass = OnlineDecisionTransformer if model_type == "odt" else DecisionTransformer
    model = ModelClass(
        state_dim, act_dim,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        max_len=args.context_len, dropout=args.dropout,
    ).to(device)
    print(f"Model ({model_type.upper()}): {sum(p.numel() for p in model.parameters()):,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    history = {"train_loss": [], "val_loss": [], "eval_return": [], "eval_normalized": []}
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for states, actions, rtgs, timesteps in tqdm(
                train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            states = states.to(device)
            actions = actions.to(device)
            rtgs = rtgs.to(device)
            timesteps = timesteps.to(device)

            if model_type == "odt":
                mean, log_std = model(states, actions, rtgs, timesteps)
                std = log_std.exp()
                dist = torch.distributions.Normal(mean, std)
                log_prob = dist.log_prob(actions).sum(dim=-1)
                nll_loss = -log_prob.mean()
                entropy = dist.entropy().sum(dim=-1).mean()
                loss = nll_loss - args.entropy_weight * entropy
            else:
                pred = model(states, actions, rtgs, timesteps)
                loss = nn.functional.mse_loss(pred, actions)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()
        train_loss = np.mean(losses)
        history["train_loss"].append(train_loss)

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for states, actions, rtgs, timesteps in val_loader:
                states = states.to(device)
                actions = actions.to(device)
                rtgs = rtgs.to(device)
                timesteps = timesteps.to(device)

                if model_type == "odt":
                    mean, log_std = model(states, actions, rtgs, timesteps)
                    std = log_std.exp()
                    dist = torch.distributions.Normal(mean, std)
                    val_losses.append(-dist.log_prob(actions).sum(-1).mean().item())
                else:
                    pred = model(states, actions, rtgs, timesteps)
                    val_losses.append(nn.functional.mse_loss(pred, actions).item())
        val_loss = np.mean(val_losses)
        history["val_loss"].append(val_loss)

        print(f"[Epoch {epoch}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(),
                       os.path.join(args.out_dir, f"best_model_{model_type}.pth"))

        if epoch % args.save_every == 0:
            torch.save(model.state_dict(),
                       os.path.join(args.out_dir, f"{model_type}_epoch{epoch}.pth"))

        if epoch % args.eval_every == 0:
            mean_ret, std_ret = evaluate_dt(
                model, meta["env_name"], device,
                target_return=args.target_return,
                num_episodes=args.eval_episodes,
                max_ep_len=1000, context_len=args.context_len,
                state_dim=state_dim, act_dim=act_dim,
            )
            norm = d4rl_normalized_score(args.dataset, mean_ret)
            history["eval_return"].append(mean_ret)
            history["eval_normalized"].append(norm)
            print(f"  → Eval return: {mean_ret:.1f} +/- {std_ret:.1f}  "
                  f"(D4RL score: {norm:.1f})")

    save_results(args, history, model_type)

    # ── Online fine-tuning (ODT only) ──
    if model_type == "odt" and args.online_epochs > 0:
        print("\n=== Online Fine-Tuning Phase ===")
        online_optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.1,
                                             weight_decay=args.weight_decay)
        online_history = {"online_loss": [], "online_return": [], "online_normalized": []}

        for oe in range(1, args.online_epochs + 1):
            # collect rollouts
            new_trajs = []
            ep_returns = []
            for _ in range(args.online_rollouts):
                traj = collect_online_rollout(
                    model, meta["env_name"], device,
                    target_return=args.target_return,
                    max_ep_len=1000, context_len=args.context_len,
                    state_dim=state_dim, act_dim=act_dim,
                )
                new_trajs.append(traj)
                ep_returns.append(traj["rewards"].sum())

            mean_ret = np.mean(ep_returns)
            norm = d4rl_normalized_score(args.dataset, mean_ret)
            online_history["online_return"].append(mean_ret)
            online_history["online_normalized"].append(norm)
            print(f"[Online {oe}] rollout return: {mean_ret:.1f} "
                  f"(D4RL: {norm:.1f})")

            # fine-tune on collected data (hindsight-relabeled)
            online_ds = DTDataset(new_trajs, context_len=args.context_len)
            online_loader = DataLoader(online_ds, batch_size=args.batch_size,
                                       shuffle=True, num_workers=0, pin_memory=True)

            model.train()
            ol_losses = []
            for states, actions, rtgs, timesteps in online_loader:
                states = states.to(device)
                actions = actions.to(device)
                rtgs = rtgs.to(device)
                timesteps = timesteps.to(device)

                mean, log_std = model(states, actions, rtgs, timesteps)
                std = log_std.exp()
                dist = torch.distributions.Normal(mean, std)
                log_prob = dist.log_prob(actions).sum(dim=-1)
                nll_loss = -log_prob.mean()
                entropy = dist.entropy().sum(dim=-1).mean()
                loss = nll_loss - args.entropy_weight * entropy

                online_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                online_optimizer.step()
                ol_losses.append(loss.item())

            online_history["online_loss"].append(np.mean(ol_losses))

        torch.save(model.state_dict(),
                   os.path.join(args.out_dir, "best_model_odt_online.pth"))

        with open(os.path.join(args.out_dir, "online_history.json"), "w") as f:
            json.dump(online_history, f, indent=2, cls=NumpyEncoder)

    return model


# ────────────────────────────────────────────────────────────────
# Results saving
# ────────────────────────────────────────────────────────────────

def save_results(args, history, model_name):
    with open(os.path.join(args.out_dir, f"{model_name}_history.json"), "w") as f:
        json.dump(history, f, indent=2, cls=NumpyEncoder)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name.upper()} Loss Curves")
    axes[0].legend()

    if history["eval_normalized"]:
        eval_epochs = list(range(args.eval_every, args.epochs + 1, args.eval_every))
        eval_epochs = eval_epochs[:len(history["eval_normalized"])]
        axes[1].plot(eval_epochs, history["eval_normalized"], "o-")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("D4RL Normalized Score")
        axes[1].set_title(f"{model_name.upper()} Evaluation")

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"{model_name}_curves.png"), dpi=150)
    plt.close()
    print(f"Results saved to {args.out_dir}/")


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    print(f"\n{'='*50}")
    print(f"Training: {args.model.upper()}")
    print(f"{'='*50}\n")

    if args.model == "bc":
        train_bc(args, device)
    elif args.model == "dt":
        train_dt(args, device, model_type="dt")
    elif args.model == "odt":
        train_dt(args, device, model_type="odt")


if __name__ == "__main__":
    main()
