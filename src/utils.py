"""
Evaluation and utility functions.
"""

import random
import numpy as np
import torch
import gymnasium as gym


# ────────────────────────────────────────────────────────────────
# Reproducibility
# ────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ────────────────────────────────────────────────────────────────
# D4RL scoring
# ────────────────────────────────────────────────────────────────

REF_SCORES = {
    "hopper-medium-replay-v2": {"random": -20.272305, "expert": 3234.3},
}


def d4rl_normalized_score(dataset_name, raw_score):
    """Convert raw episodic return to D4RL 0-100 normalized score."""
    ref = REF_SCORES[dataset_name]
    return (raw_score - ref["random"]) / (ref["expert"] - ref["random"]) * 100.0


# ────────────────────────────────────────────────────────────────
# Environment evaluation
# ────────────────────────────────────────────────────────────────

def evaluate_bc(model, env_name, device, num_episodes=10, max_ep_len=1000,
                context_len=20, state_dim=11):
    """Evaluate a BehaviorCloning model in the environment."""
    model.eval()
    returns = []

    for _ in range(num_episodes):
        env = gym.make(env_name)
        state, _ = env.reset()
        states = torch.zeros(1, context_len, state_dim, device=device)
        ep_return = 0.0

        for t in range(max_ep_len):
            # shift context window
            idx = min(t, context_len - 1)
            if t < context_len:
                states[0, t] = torch.tensor(state, device=device)
            else:
                states = torch.roll(states, -1, dims=1)
                states[0, -1] = torch.tensor(state, device=device)

            with torch.no_grad():
                action = model.get_action(states)

            action_np = action.squeeze(0).cpu().numpy()
            action_np = np.clip(action_np, -1.0, 1.0)
            state, reward, terminated, truncated, _ = env.step(action_np)
            ep_return += reward

            if terminated or truncated:
                break

        env.close()
        returns.append(ep_return)

    model.train()
    return np.mean(returns), np.std(returns)


def evaluate_dt(model, env_name, device, target_return, num_episodes=10,
                max_ep_len=1000, context_len=20, state_dim=11, act_dim=3,
                deterministic=True):
    """Evaluate a DecisionTransformer or OnlineDecisionTransformer."""
    model.eval()
    returns = []

    for _ in range(num_episodes):
        env = gym.make(env_name)
        state, _ = env.reset()

        states = torch.zeros(1, context_len, state_dim, device=device)
        actions = torch.zeros(1, context_len, act_dim, device=device)
        rtgs = torch.zeros(1, context_len, 1, device=device)
        timesteps = torch.zeros(1, context_len, dtype=torch.long, device=device)

        states[0, 0] = torch.tensor(state, device=device)
        rtgs[0, 0] = target_return

        ep_return = 0.0
        for t in range(max_ep_len):
            idx = min(t, context_len - 1)

            with torch.no_grad():
                if hasattr(model, "action_mean"):
                    # OnlineDecisionTransformer
                    action = model.get_action(
                        states[:, :idx + 1],
                        actions[:, :idx + 1],
                        rtgs[:, :idx + 1],
                        timesteps[:, :idx + 1],
                        deterministic=deterministic,
                    )
                else:
                    # DecisionTransformer
                    action = model.get_action(
                        states[:, :idx + 1],
                        actions[:, :idx + 1],
                        rtgs[:, :idx + 1],
                        timesteps[:, :idx + 1],
                    )

            action_np = action.squeeze(0).cpu().numpy()
            action_np = np.clip(action_np, -1.0, 1.0)
            state, reward, terminated, truncated, _ = env.step(action_np)
            ep_return += reward

            if terminated or truncated:
                break

            # update context for next step
            next_t = min(t + 1, context_len - 1)
            if t + 1 < context_len:
                states[0, t + 1] = torch.tensor(state, device=device)
                actions[0, t] = action.squeeze(0)
                rtgs[0, t + 1] = rtgs[0, t] - reward
                timesteps[0, t + 1] = t + 1
            else:
                # shift window
                states = torch.roll(states, -1, dims=1)
                actions = torch.roll(actions, -1, dims=1)
                rtgs = torch.roll(rtgs, -1, dims=1)
                timesteps = torch.roll(timesteps, -1, dims=1)
                states[0, -1] = torch.tensor(state, device=device)
                actions[0, -2] = action.squeeze(0)
                rtgs[0, -1] = rtgs[0, -2] - reward
                timesteps[0, -1] = t + 1

        env.close()
        returns.append(ep_return)

    model.train()
    return np.mean(returns), np.std(returns)


# ────────────────────────────────────────────────────────────────
# Online rollout collection (for ODT)
# ────────────────────────────────────────────────────────────────

def collect_online_rollout(model, env_name, device, target_return,
                           max_ep_len=1000, context_len=20,
                           state_dim=11, act_dim=3):
    """Collect a single rollout for online fine-tuning with hindsight relabeling."""
    model.eval()
    env = gym.make(env_name)
    state, _ = env.reset()

    traj = {"observations": [], "actions": [], "rewards": []}

    states_buf = torch.zeros(1, context_len, state_dim, device=device)
    actions_buf = torch.zeros(1, context_len, act_dim, device=device)
    rtgs_buf = torch.zeros(1, context_len, 1, device=device)
    timesteps_buf = torch.zeros(1, context_len, dtype=torch.long, device=device)

    states_buf[0, 0] = torch.tensor(state, device=device)
    rtgs_buf[0, 0] = target_return

    for t in range(max_ep_len):
        idx = min(t, context_len - 1)

        with torch.no_grad():
            action = model.get_action(
                states_buf[:, :idx + 1],
                actions_buf[:, :idx + 1],
                rtgs_buf[:, :idx + 1],
                timesteps_buf[:, :idx + 1],
                deterministic=False,
            )

        action_np = action.squeeze(0).cpu().numpy()
        action_np = np.clip(action_np, -1.0, 1.0)

        traj["observations"].append(state.copy())
        traj["actions"].append(action_np.copy())

        state, reward, terminated, truncated, _ = env.step(action_np)
        traj["rewards"].append(reward)

        if terminated or truncated:
            break

        next_t = min(t + 1, context_len - 1)
        if t + 1 < context_len:
            states_buf[0, t + 1] = torch.tensor(state, device=device)
            actions_buf[0, t] = action.squeeze(0)
            rtgs_buf[0, t + 1] = rtgs_buf[0, t] - reward
            timesteps_buf[0, t + 1] = t + 1
        else:
            states_buf = torch.roll(states_buf, -1, dims=1)
            actions_buf = torch.roll(actions_buf, -1, dims=1)
            rtgs_buf = torch.roll(rtgs_buf, -1, dims=1)
            timesteps_buf = torch.roll(timesteps_buf, -1, dims=1)
            states_buf[0, -1] = torch.tensor(state, device=device)
            actions_buf[0, -2] = action.squeeze(0)
            rtgs_buf[0, -1] = rtgs_buf[0, -2] - reward
            timesteps_buf[0, -1] = t + 1

    env.close()

    # convert to numpy
    traj["observations"] = np.array(traj["observations"], dtype=np.float32)
    traj["actions"] = np.array(traj["actions"], dtype=np.float32)
    traj["rewards"] = np.array(traj["rewards"], dtype=np.float32)
    traj["terminals"] = np.zeros(len(traj["rewards"]), dtype=bool)
    traj["timeouts"] = np.zeros(len(traj["rewards"]), dtype=bool)

    # hindsight return relabeling
    rewards = traj["rewards"]
    rtg = np.zeros_like(rewards)
    rtg[-1] = rewards[-1]
    for i in reversed(range(len(rewards) - 1)):
        rtg[i] = rewards[i] + rtg[i + 1]
    traj["returns_to_go"] = rtg

    model.train()
    return traj
