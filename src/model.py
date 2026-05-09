"""
Transformer-based models for Offline Reinforcement Learning.

Models:
  1. BehaviorCloning  – causal transformer that imitates dataset actions (baseline)
  2. DecisionTransformer – return-conditioned sequence model (proposed)
  3. OnlineDecisionTransformer – stochastic DT with online fine-tuning (proposed)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────
# Building blocks
# ────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_len, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0),
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # (3, B, H, T, D)
        q, k, v = qkv.unbind(0)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj_drop(self.proj(out))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, max_len, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ────────────────────────────────────────────────────────────────
# 1. Behavior Cloning (Baseline)
# ────────────────────────────────────────────────────────────────

class BehaviorCloning(nn.Module):
    """Causal transformer that predicts actions from state history only.
    Ignores returns — just imitates average dataset behaviour."""

    def __init__(self, state_dim, act_dim, d_model=128, n_heads=4,
                 n_layers=3, max_len=20, dropout=0.1):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.d_model = d_model
        self.max_len = max_len

        self.state_embed = nn.Linear(state_dim, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[TransformerBlock(d_model, n_heads, max_len, dropout)
              for _ in range(n_layers)]
        )
        self.ln = nn.LayerNorm(d_model)
        self.action_head = nn.Linear(d_model, act_dim)

    def forward(self, states):
        """
        states: (B, T, state_dim)
        returns: predicted actions (B, T, act_dim)
        """
        B, T, _ = states.shape
        pos = torch.arange(T, device=states.device).unsqueeze(0)
        x = self.drop(self.state_embed(states) + self.pos_embed(pos))
        x = self.ln(self.blocks(x))
        return self.action_head(x)

    def get_action(self, states):
        """Inference: predict action for most recent state."""
        actions = self.forward(states)
        return actions[:, -1, :]


# ────────────────────────────────────────────────────────────────
# 2. Decision Transformer (Deterministic)
# ────────────────────────────────────────────────────────────────

class DecisionTransformer(nn.Module):
    """Return-conditioned transformer.
    Sequence: [R_1, s_1, a_1, R_2, s_2, a_2, ...]
    Predicts actions at the action-token positions."""

    def __init__(self, state_dim, act_dim, d_model=128, n_heads=4,
                 n_layers=3, max_len=20, max_ep_len=1000, dropout=0.1):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.d_model = d_model
        self.max_len = max_len

        # modality embeddings
        self.state_embed = nn.Linear(state_dim, d_model)
        self.action_embed = nn.Linear(act_dim, d_model)
        self.return_embed = nn.Linear(1, d_model)

        # positional / timestep embedding
        self.timestep_embed = nn.Embedding(max_ep_len, d_model)
        self.pos_embed = nn.Embedding(3 * max_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[TransformerBlock(d_model, n_heads, 3 * max_len, dropout)
              for _ in range(n_layers)]
        )
        self.ln = nn.LayerNorm(d_model)

        # prediction heads
        self.action_head = nn.Linear(d_model, act_dim)

    def forward(self, states, actions, returns_to_go, timesteps):
        """
        All inputs: (B, T, *)
        returns_to_go: (B, T, 1)
        timesteps:     (B, T)  integer
        """
        B, T, _ = states.shape

        time_emb = self.timestep_embed(timesteps)           # (B, T, d)

        s_emb = self.state_embed(states) + time_emb         # (B, T, d)
        a_emb = self.action_embed(actions) + time_emb       # (B, T, d)
        r_emb = self.return_embed(returns_to_go) + time_emb # (B, T, d)

        # interleave: [R_1, s_1, a_1, R_2, s_2, a_2, ...]
        stacked = torch.stack([r_emb, s_emb, a_emb], dim=2)  # (B, T, 3, d)
        stacked = stacked.reshape(B, 3 * T, self.d_model)     # (B, 3T, d)

        pos = torch.arange(3 * T, device=states.device).unsqueeze(0)
        x = self.drop(stacked + self.pos_embed(pos))

        x = self.ln(self.blocks(x))                            # (B, 3T, d)

        # extract state-position tokens (indices 1, 4, 7, ...)
        # action is predicted from the state token
        x_states = x[:, 1::3, :]                               # (B, T, d)
        action_preds = self.action_head(x_states)               # (B, T, act_dim)
        return action_preds

    def get_action(self, states, actions, returns_to_go, timesteps):
        """Inference: predict the next action given context window."""
        action_preds = self.forward(states, actions, returns_to_go, timesteps)
        return action_preds[:, -1, :]


# ────────────────────────────────────────────────────────────────
# 3. Online Decision Transformer (Stochastic)
# ────────────────────────────────────────────────────────────────

class OnlineDecisionTransformer(nn.Module):
    """Stochastic DT that outputs a Gaussian policy.
    Supports offline pre-training + online fine-tuning with
    hindsight return relabeling."""

    def __init__(self, state_dim, act_dim, d_model=128, n_heads=4,
                 n_layers=3, max_len=20, max_ep_len=1000, dropout=0.1):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.d_model = d_model
        self.max_len = max_len

        self.state_embed = nn.Linear(state_dim, d_model)
        self.action_embed = nn.Linear(act_dim, d_model)
        self.return_embed = nn.Linear(1, d_model)

        self.timestep_embed = nn.Embedding(max_ep_len, d_model)
        self.pos_embed = nn.Embedding(3 * max_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[TransformerBlock(d_model, n_heads, 3 * max_len, dropout)
              for _ in range(n_layers)]
        )
        self.ln = nn.LayerNorm(d_model)

        # stochastic action head → mean and log_std
        self.action_mean = nn.Linear(d_model, act_dim)
        self.action_log_std = nn.Linear(d_model, act_dim)

        self.min_log_std = -5.0
        self.max_log_std = 2.0

    def forward(self, states, actions, returns_to_go, timesteps):
        B, T, _ = states.shape

        time_emb = self.timestep_embed(timesteps)
        s_emb = self.state_embed(states) + time_emb
        a_emb = self.action_embed(actions) + time_emb
        r_emb = self.return_embed(returns_to_go) + time_emb

        stacked = torch.stack([r_emb, s_emb, a_emb], dim=2)
        stacked = stacked.reshape(B, 3 * T, self.d_model)

        pos = torch.arange(3 * T, device=states.device).unsqueeze(0)
        x = self.drop(stacked + self.pos_embed(pos))
        x = self.ln(self.blocks(x))

        x_states = x[:, 1::3, :]

        mean = self.action_mean(x_states)
        log_std = self.action_log_std(x_states)
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)

        return mean, log_std

    def get_action(self, states, actions, returns_to_go, timesteps,
                   deterministic=False):
        mean, log_std = self.forward(states, actions, returns_to_go, timesteps)
        mean = mean[:, -1, :]
        log_std = log_std[:, -1, :]

        if deterministic:
            return mean

        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        return action

    def log_prob(self, states, actions, returns_to_go, timesteps):
        mean, log_std = self.forward(states, actions, returns_to_go, timesteps)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(actions).sum(dim=-1)

    def entropy(self, states, actions, returns_to_go, timesteps):
        _, log_std = self.forward(states, actions, returns_to_go, timesteps)
        std = log_std.exp()
        dist = torch.distributions.Normal(torch.zeros_like(std), std)
        return dist.entropy().sum(dim=-1)
