from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from v6rl.config import N_ACTIONS, TrainConfig
from v6rl.network import PolicyNet
from v6rl.obs import OBS_DIM


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    t_steps = len(rewards)
    adv = np.zeros(t_steps, dtype=np.float32)
    last = 0.0
    for t in reversed(range(t_steps)):
        next_v = last_value if t == t_steps - 1 else values[t + 1]
        next_alive = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * next_alive - values[t]
        last = delta + gamma * lam * next_alive * last
        adv[t] = last
    return adv, (adv + values).astype(np.float32)


class PPO:
    def __init__(self, config: TrainConfig, obs_dim: int = OBS_DIM):
        self.cfg = config
        self.device = torch.device(config.device)
        self.net = PolicyNet(obs_dim, hidden=config.hidden).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=config.lr)
        self.hidden: torch.Tensor | None = None

    def reset_hidden(self) -> None:
        self.hidden = None

    @torch.inference_mode()
    def value(self, obs: np.ndarray) -> float:
        self.net.eval()
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        m = torch.ones(1, N_ACTIONS, dtype=torch.bool, device=self.device)
        out, _ = self.net.step(o, m, self.hidden)
        return float(out["value"].item())

    @torch.inference_mode()
    def act(self, obs: np.ndarray, mask: np.ndarray, greedy: bool = False):
        self.net.eval()
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        m = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        out, self.hidden = self.net.step(o, m, self.hidden)
        dist = out["dist"]
        action = out["actor_logits"].argmax(dim=-1) if greedy else dist.sample()
        size = out["size_mean"] if greedy else out["size_scale"]
        lot = out["lot_frac"]
        lot_v = np.array([float(lot.reshape(-1)[0].item())], dtype=np.float32)
        size_v = np.array([float(size.reshape(-1)[0].item())], dtype=np.float32)
        return {
            "action": int(action.item()),
            "logp": float(dist.log_prob(action).item()),
            "value": float(out["value"].item()),
            "lot_frac": lot_v,
            "size_scale": size_v,
            "actor_logits": out["actor_logits"][0].detach().cpu().numpy(),
            "aux_logit": float(out["aux_logit"].item()),
        }

    @staticmethod
    def logp_of(logits: np.ndarray, action: int) -> float:
        t = torch.as_tensor(logits, dtype=torch.float32)
        return float(Categorical(logits=t).log_prob(torch.tensor(action)).item())

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        self.net.train()
        adv, ret = compute_gae(
            batch["rewards"],
            batch["values"],
            batch["dones"],
            float(batch["last_value"]),
            self.cfg.gamma,
            self.cfg.gae_lambda,
        )
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        t_obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        t_mask = torch.as_tensor(batch["masks"], dtype=torch.bool, device=self.device)
        t_act = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
        t_old = torch.as_tensor(batch["logp"], dtype=torch.float32, device=self.device)
        t_adv = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        t_ret = torch.as_tensor(ret, dtype=torch.float32, device=self.device)
        t_aux = torch.as_tensor(batch["aux_y"], dtype=torch.float32, device=self.device)
        t_ss = torch.as_tensor(batch["size_y"], dtype=torch.float32, device=self.device)
        t_lot = torch.as_tensor(batch["lot_y"], dtype=torch.float32, device=self.device)
        dones = batch["dones"]
        n = t_obs.size(0)
        mb = min(self.cfg.minibatch, n)
        last = {}
        aux_hist: list[float] = []
        for _ in range(self.cfg.ppo_epochs):
            hidden = None
            for start in range(0, n, mb):
                end = min(start + mb, n)
                sl = slice(start, end)
                reset = torch.zeros(end - start, dtype=torch.bool, device=self.device)
                for j, t in enumerate(range(start, end)):
                    if t > 0 and dones[t - 1] > 0.5:
                        reset[j] = True
                if start > 0 and dones[start - 1] > 0.5:
                    hidden = None
                out, hidden = self.net.unroll(t_obs[sl], t_mask[sl], hidden, reset)
                hidden = hidden.detach() if hidden is not None else None
                logp = out["dist"].log_prob(t_act[sl])
                entropy = out["dist"].entropy()
                value = out["value"]
                ratio = (logp - t_old[sl]).exp()
                surr1 = ratio * t_adv[sl]
                surr2 = ratio.clamp(1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * t_adv[sl]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (value - t_ret[sl]).pow(2).mean()
                valid_aux = t_aux[sl] >= 0
                if valid_aux.any():
                    aux_loss = F.binary_cross_entropy_with_logits(out["aux_logit"][valid_aux], t_aux[sl][valid_aux])
                else:
                    aux_loss = torch.zeros((), device=self.device)
                size_loss = (out["size_mean"] - t_ss[sl]).pow(2).mean()
                lot_ok = t_lot[sl] >= 0
                if lot_ok.any():
                    lot_loss = (out["lot_frac"][lot_ok] - t_lot[sl][lot_ok]).pow(2).mean()
                else:
                    lot_loss = torch.zeros((), device=self.device)
                ent_mean = entropy.mean()
                c_ent = float(self.cfg.entropy_coef)
                floor = float(self.cfg.entropy_floor)
                ent_now = float(ent_mean.detach())
                if ent_now < floor:
                    c_ent = c_ent * (floor / (ent_now + 1e-6))
                loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    - c_ent * ent_mean
                    + self.cfg.aux_coef * aux_loss
                    + self.cfg.size_coef * size_loss
                    + self.cfg.lot_coef * lot_loss
                )
                if not torch.isfinite(loss):
                    continue
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()
                aux_hist.append(float(aux_loss.item()))
                last = {
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(ent_mean.item()),
                    "entropy_coef": float(c_ent),
                    "aux_loss": float(aux_loss.item()),
                    "size_loss": float(size_loss.item()),
                    "lot_loss": float(lot_loss.item()),
                    "loss": float(loss.item()),
                    "aux_nan": 0.0,
                }
        if aux_hist:
            last["aux_loss_mean"] = float(np.mean(aux_hist))
            last["aux_loss_max"] = float(np.max(aux_hist))
        return last
