from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Categorical

from v6rl.config import GRU_HIDDEN, N_ACTIONS, SIZE_FLOOR
from v6rl.obs import OBS_DIM


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 256, gru_hidden: int = GRU_HIDDEN):
        super().__init__()
        self.gru_hidden = gru_hidden
        self.enc = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
        )
        self.gru = nn.GRU(hidden, gru_hidden, batch_first=True)
        self.actor = nn.Linear(gru_hidden, N_ACTIONS)
        self.critic = nn.Linear(gru_hidden, 1)
        self.lot = nn.Linear(gru_hidden, 1)
        self.size_ab = nn.Linear(gru_hidden, 2)
        self.aux_ride = nn.Linear(gru_hidden, 1)
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.constant_(self.lot.bias, 1.4)
        with torch.no_grad():
            self.size_ab.bias[0] = 2.0
            self.size_ab.bias[1] = -1.0

    def init_hidden(self, batch: int = 1, device=None) -> torch.Tensor:
        return torch.zeros(1, batch, self.gru_hidden, device=device)

    def _heads(self, z: torch.Tensor, mask: torch.Tensor) -> dict:
        logits = self.actor(z).masked_fill(~mask, -1e9)
        raw_ab = self.size_ab(z)
        alpha = F.softplus(raw_ab[..., 0]) + 1.0
        beta = F.softplus(raw_ab[..., 1]) + 1.0
        size_dist = Beta(alpha, beta)
        span = 1.0 - SIZE_FLOOR
        size_sample = SIZE_FLOOR + span * size_dist.rsample()
        size_mean = SIZE_FLOOR + span * size_dist.mean
        return {
            "dist": Categorical(logits=logits),
            "value": self.critic(z).squeeze(-1),
            "lot_frac": torch.sigmoid(self.lot(z)).squeeze(-1),
            "size_scale": size_sample,
            "size_mean": size_mean,
            "size_dist": size_dist,
            "actor_logits": logits,
            "aux_logit": self.aux_ride(z).squeeze(-1),
        }

    def step(self, obs: torch.Tensor, mask: torch.Tensor, hidden: torch.Tensor | None = None):
        x = self.enc(obs).unsqueeze(1)
        if hidden is None:
            hidden = self.init_hidden(obs.size(0), obs.device)
        y, hidden = self.gru(x, hidden)
        return self._heads(y.squeeze(1), mask), hidden

    def unroll(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        hidden: torch.Tensor | None = None,
        reset_before: torch.Tensor | None = None,
    ):
        x = self.enc(obs)
        if hidden is None:
            hidden = self.init_hidden(1, obs.device)
        if reset_before is None or not bool(reset_before.any()):
            y, hidden = self.gru(x.unsqueeze(0), hidden)
            return self._heads(y.squeeze(0), mask), hidden
        flags = reset_before.detach().cpu().numpy().astype(bool)
        parts = []
        start = 0
        t_len = x.size(0)
        h = hidden
        for t in range(t_len):
            if t > 0 and flags[t]:
                y, h = self.gru(x[start:t].unsqueeze(0), h)
                parts.append(y.squeeze(0))
                h = self.init_hidden(1, obs.device)
                start = t
        y, h = self.gru(x[start:t_len].unsqueeze(0), h)
        parts.append(y.squeeze(0))
        return self._heads(torch.cat(parts, 0), mask), h
