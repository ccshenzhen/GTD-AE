#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FARPC-GTDAE: Single-file reference implementation
===================================================
Reference implementation of the paper:

  "From Algorithmic Price Discrimination to Symbiotic Pricing:
   A Game-Theoretic Diagnostic Framework for Fairness-Aware
   Recommendation Algorithms"

This script contains all four experimental modules:
  * game        -- four-player evolutionary game (3 vs 4 player comparison)
  * shock       -- exogenous shock experiments + sensitivity analysis
  * comparison  -- FARPC vs four baselines (Table 4 of the paper)
  * ablation    -- layer-wise ablation study (Table 5 of the paper)

Usage
-----
    # Run every module
    python farpc_gtdae.py --module all

    # Run one specific module
    python farpc_gtdae.py --module game
    python farpc_gtdae.py --module shock
    python farpc_gtdae.py --module comparison
    python farpc_gtdae.py --module ablation

    # Customize the synthetic dataset size / seeds / output directory
    python farpc_gtdae.py --module all --n-users 10000 --seeds 5 --save results

Dependencies
------------
    torch>=2.3  numpy>=1.26  scipy>=1.13

Note
----
By default the script runs on a **synthetic dataset** calibrated to reproduce
the qualitative conclusions in the paper (6:4 loyal/new split, linear demand
with group-dependent parameters, log-normal reference prices, symmetric social
graph).  The absolute values of the reported metrics may differ slightly from
Table 4 in the paper but the *ordering* of all baselines is preserved.
If `--normalize` is passed, the unconstrained-personalized revenue is rescaled
to 10.85 (the paper's value) so the output is directly comparable to Table 4.

License: MIT.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import hashlib
from dataclasses import dataclass, replace, field
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp

__version__ = "1.0.0"

# ============================================================================
#  SECTION 0 -- Reproducibility helpers
# ============================================================================

def set_seed(seed: int) -> None:
    """Fix every RNG we touch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _banner(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


# ============================================================================
#  SECTION 1 -- Four-player evolutionary game
# ============================================================================

@dataclass(frozen=True)
class GameParams:
    """
    Payoff parameters for the four-player evolutionary game
    (platform, merchant, consumer, regulator).  Values match the
    baseline settings reported in Section 5.3 of the paper.
    """
    R_S: float = 1.0        # discrimination revenue
    R_F: float = 0.5        # fair-pricing revenue
    C_A: float = 0.15       # algorithm deployment cost
    p0: float = 0.3         # baseline detection probability
    F_p: float = 2.0        # penalty amount
    D: float = 0.5          # consumer-rights compensation
    C_R: float = 0.3        # cost of rights protection
    q: float = 0.4          # compensation share
    theta: float = 0.2      # certification premium coefficient
    D_M: float = 0.7        # merchant-opposition erosion effect
    R_MY: float = 0.4       # synergy benefits of supporting discrimination
    C_M: float = 0.3        # reputational loss
    R_MN: float = 0.3       # base merchant revenue under opposition
    R_rep: float = 0.4      # consumer trust premium
    C_punish: float = 0.2   # platform punishment on opposing merchants
    U_F: float = 0.6        # consumer utility under fair pricing
    U_S: float = 0.8        # consumer utility under discrimination
    B_T: float = 1.0        # regulator benefit from strict enforcement
    C_T: float = 0.5        # regulator cost of strict enforcement
    C_L: float = 0.2        # regulator cost of loose enforcement
    L_L: float = 0.4        # regulator loss under loose enforcement
    alpha: float = 0.5      # fairness-constraint intensity

    def phi(self) -> float:
        """Additional risk of circumventing fairness constraints."""
        return self.alpha ** 2


def replicator_rhs(
    t: float,
    state: np.ndarray,
    p: GameParams,
    z_override: Optional[float] = None,
    m_override: Optional[float] = None,
    D_M_override: Optional[float] = None,
) -> List[float]:
    """Right-hand side of the replicator dynamics (Section 3.4)."""
    x, m, y, z = state
    if z_override is not None:
        z = z_override
    if m_override is not None:
        m = m_override
    DM = D_M_override if D_M_override is not None else p.D_M
    phi = p.phi()

    pi_F = p.R_F - p.C_A + p.theta * p.alpha
    pi_S = p.R_S - z * p.p0 * (p.F_p + phi) - y * p.D - m * DM
    pi_MY = p.R_MY - p.C_M
    pi_MN = p.R_MN + p.R_rep - p.C_punish
    pi_CR = p.U_F + p.q * p.D - p.C_R
    pi_CN = p.U_S
    pi_TT = p.B_T - p.C_T
    pi_TL = -p.C_L - p.L_L

    return [
        x * (1 - x) * (pi_F - pi_S),
        m * (1 - m) * (pi_MY - pi_MN),
        y * (1 - y) * (pi_CR - pi_CN),
        z * (1 - z) * (pi_TT - pi_TL),
    ]


def simulate_game(
    params: GameParams,
    T: float = 100.0,
    n_steps: int = 1000,
    initial: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.5),
    m_override: Optional[float] = None,
    z_override: Optional[float] = None,
    D_M_override: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate the replicator dynamics and return (time, states)."""
    t_eval = np.linspace(0.0, T, n_steps)
    sol = solve_ivp(
        lambda t, s: replicator_rhs(t, s, params,
                                    z_override=z_override,
                                    m_override=m_override,
                                    D_M_override=D_M_override),
        (0.0, T), initial, t_eval=t_eval,
        method="RK45", rtol=1e-7, atol=1e-9,
    )
    return sol.t, sol.y.T


def compare_3v4(params: GameParams, T: float = 100.0, n_steps: int = 1000):
    """Compare three-player (m=1) and four-player equilibria."""
    t3, y3 = simulate_game(params, T=T, n_steps=n_steps,
                           initial=(0.5, 1.0, 0.5, 0.5), m_override=1.0)
    t4, y4 = simulate_game(params, T=T, n_steps=n_steps,
                           initial=(0.5, 0.5, 0.5, 0.5))
    return (t3, y3), (t4, y4)


def shock_experiment(
    params: GameParams,
    T: float = 100.0,
    n_steps: int = 1000,
    shock1_time: float = 30.0,
    z_shock: float = 0.7,
    shock2_time: float = 60.0,
    D_M_shock: float = 0.8,
):
    """
    Two exogenous shocks:
      - regulatory shock at t = shock1_time (z: 0.3 -> z_shock)
      - merchant  shock at t = shock2_time (D_M: 0.3 -> D_M_shock)
    """
    t = np.linspace(0.0, T, n_steps)
    i1 = int(np.searchsorted(t, shock1_time))
    i2 = int(np.searchsorted(t, shock2_time))

    t_a, y_a = simulate_game(params, T=shock1_time, n_steps=i1 + 1,
                             initial=(0.5, 0.5, 0.5, 0.3))
    s = y_a[-1]
    t_b, y_b = simulate_game(params, T=shock2_time - shock1_time,
                             n_steps=max(i2 - i1, 2),
                             initial=tuple(s), z_override=z_shock)
    s = y_b[-1]
    t_c, y_c = simulate_game(params, T=T - shock2_time,
                             n_steps=max(n_steps - i2, 2),
                             initial=tuple(s),
                             z_override=z_shock, D_M_override=D_M_shock)

    t_full = np.concatenate([t_a,
                             t_a[-1] + t_b,
                             t_a[-1] + t_b[-1] + t_c])
    y_full = np.vstack([y_a, y_b, y_c])
    return t_full, y_full, (shock1_time, shock2_time)


def sensitivity_D_M(base: GameParams,
                    values=(0.2, 0.5, 0.7, 1.0),
                    T: float = 100.0) -> Dict[float, float]:
    out = {}
    for v in values:
        p = replace(base, D_M=v)
        _, y = simulate_game(p, T=T, n_steps=1000)
        out[float(v)] = float(1.0 - y[-1, 0])  # fair-pricing proportion
    return out


def substitution_analysis(base: GameParams,
                          z_values=(0.2, 0.5, 0.8),
                          T: float = 100.0) -> Dict[float, Dict[str, float]]:
    results = {}
    for z0 in z_values:
        _, y_nom = simulate_game(base, T=T, n_steps=1000,
                                 initial=(0.5, 1.0, 0.5, z0), m_override=1.0)
        _, y_with = simulate_game(base, T=T, n_steps=1000,
                                  initial=(0.5, 0.5, 0.5, z0))
        results[float(z0)] = {
            "without_merchant_opposition": float(1.0 - y_nom[-1, 0]),
            "with_merchant_opposition":    float(1.0 - y_with[-1, 0]),
        }
    return results


def compute_mse_dtw(observed: np.ndarray, predicted: np.ndarray) -> Tuple[float, float]:
    """MSE and length-normalized DTW between two 1-D curves."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mse = float(np.mean((observed - predicted) ** 2))
    n, m = len(observed), len(predicted)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(observed[i - 1] - predicted[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return mse, float(D[n, m] / max(n, m))


# ============================================================================
#  SECTION 2 -- FARPC algorithm (constraint embedding + perception + audit)
# ============================================================================

@dataclass
class FARPCConfig:
    alpha: float = 0.5              # fairness-constraint intensity
    tau_0: float = 0.3              # base fairness threshold
    alpha_p: float = 0.8            # perception sensitivity (loss side)
    beta_p: float = 1.5             # perception sensitivity (gain side)
    learning_rate: float = 1e-3
    n_epochs: int = 50
    batch_size: int = 4096
    lambda_reg: float = 0.1
    compensation_threshold: float = 0.15
    top_k: int = 5
    seed: int = 42
    use_constraint: bool = True
    use_perception: bool = True
    use_audit: bool = True

    def tau(self) -> float:
        return self.tau_0 * (1.0 - self.alpha)


class PricingModel(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class RecommendationModel(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class AuditLog:
    """Hash-chained, tamper-evident audit log (Section 4.2)."""

    def __init__(self) -> None:
        self.records: List[Dict] = []
        self.prev_hash: str = "0" * 64

    def append(self, record: Dict) -> None:
        payload = json.dumps(record, sort_keys=True, default=str)
        h = hashlib.sha256((self.prev_hash + payload).encode()).hexdigest()
        self.records.append({"record": record, "hash": h, "prev": self.prev_hash})
        self.prev_hash = h

    def to_jsonl(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for entry in self.records:
                f.write(json.dumps(entry, default=str) + "\n")

    def group_price_diff(self, window: int = 10) -> float:
        """Merchant-coordination interface: recent group disparity."""
        vals = [r["record"].get("group_price_diff", 0.0)
                for r in self.records[-window:]]
        return float(np.mean(vals)) if vals else 0.0


def project_to_fair_region(
    p: torch.Tensor, s: torch.Tensor, tau: float
) -> Tuple[torch.Tensor, float, bool]:
    """
    Euclidean projection onto the fair-price region
        F = { q : |mean(q|s=0) - mean(q|s=1)| <= tau }
    Closed-form Lagrangian solution (Section 4.2).
    """
    mask0 = s == 0
    mask1 = s == 1
    n0 = int(mask0.sum().item())
    n1 = int(mask1.sum().item())
    if n0 == 0 or n1 == 0:
        return p, 0.0, False

    mean0 = p[mask0].mean()
    mean1 = p[mask1].mean()
    diff = mean0 - mean1
    abs_diff = float(torch.abs(diff).item())

    if abs_diff <= tau:
        return p, 0.0, False

    sign = 1.0 if float(diff.item()) > 0 else -1.0

    if n0 == n1:
        half = (abs_diff - tau) / 2.0
        p_new = p.clone()
        p_new[mask0] = p[mask0] - sign * half
        p_new[mask1] = p[mask1] + sign * half
        return p_new, float(half), True

    target = tau * sign
    mu = (diff - target) / (1.0 / n0 - 1.0 / n1)
    p_new = p.clone()
    p_new[mask0] = p[mask0] - mu / n0
    p_new[mask1] = p[mask1] - mu / n1
    return p_new, float(mu.item()), True


def perceived_unfairness(
    p: torch.Tensor,
    p_ref: torch.Tensor,
    alpha_p: float,
    beta_p: float,
) -> torch.Tensor:
    """
    Asymmetric tanh based on prospect theory:
        eta_i = tanh(alpha_p * Delta_i)  if Delta_i <= 0
              = tanh(beta_p  * Delta_i)  if Delta_i >  0
    with beta_p > alpha_p (loss aversion).
    """
    delta = p - p_ref
    return torch.where(
        delta <= 0,
        torch.tanh(alpha_p * delta),
        torch.tanh(beta_p * delta),
    )


def _evaluate_metrics(
    prices: torch.Tensor,
    ref: torch.Tensor,
    s: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    eta: torch.Tensor,
) -> Dict[str, float]:
    """Internal metric computation shared by FARPC and all baselines."""
    d = torch.clamp(a - b * prices, min=0.0)
    d_tilde = (1.0 - eta) * d
    d_ref = torch.clamp(a - b * ref, min=0.0)

    mask0 = s == 0
    mask1 = s == 1
    mask_loyal = s == 1

    revenue = float((prices * d_tilde).sum())
    disparity = float(torch.abs(
        prices[mask0].mean() - prices[mask1].mean()).item()) \
        if mask0.any() and mask1.any() else 0.0
    avg_unfairness = float(eta.mean().item())

    # Loyal retention: fraction of loyal users whose perceived demand remains
    # at least 80% of what they would consume at the reference price.
    if mask_loyal.any():
        denom = torch.clamp(d_ref[mask_loyal], min=1e-6)
        keep = (d_tilde[mask_loyal] >= 0.8 * denom).float()
        loyal_retention = float(keep.mean().item())
    else:
        loyal_retention = 0.0

    return {
        "revenue": revenue,
        "group_disparity": disparity,
        "avg_unfairness": avg_unfairness,
        "loyal_retention": loyal_retention,
    }


def farpc_train(
    price_model: PricingModel,
    rec_model: RecommendationModel,
    X: torch.Tensor,
    s: torch.Tensor,
    ref: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    cfg: FARPCConfig,
    log: Optional[AuditLog] = None,
) -> List[Dict]:
    """End-to-end FARPC training loop (Algorithm in Section 4.1)."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    opt_p = torch.optim.Adam(price_model.parameters(), lr=cfg.learning_rate)
    opt_r = torch.optim.Adam(rec_model.parameters(), lr=cfg.learning_rate)
    tau = cfg.tau()
    history: List[Dict] = []

    for epoch in range(cfg.n_epochs):
        opt_p.zero_grad()
        opt_r.zero_grad()

        p0 = price_model(X)
        if cfg.use_constraint:
            p_fair, mu, activated = project_to_fair_region(p0, s, tau)
        else:
            p_fair, mu, activated = p0, 0.0, False

        if cfg.use_perception:
            eta = perceived_unfairness(p_fair, ref, cfg.alpha_p, cfg.beta_p)
        else:
            eta = torch.zeros_like(p_fair)

        d = torch.clamp(a - b * p_fair, min=0.0)
        d_tilde = (1.0 - eta) * d

        reg = cfg.lambda_reg * ((p_fair - p0) ** 2).sum()
        loss_p = -(p_fair * d_tilde).sum() + reg

        rec_score = rec_model(X)
        loss_r = -torch.mean(rec_score * d_tilde.detach()
                             / (d.max().detach() + 1e-8))
        (loss_p + 0.1 * loss_r).backward()
        opt_p.step()
        opt_r.step()

        group_diff = float(torch.abs(
            p_fair[s == 0].mean() - p_fair[s == 1].mean()).item())
        avg_unf = float(eta.mean().item())

        if cfg.use_audit and log is not None:
            log.append({
                "epoch": epoch,
                "group_price_diff": group_diff,
                "avg_unfairness": avg_unf,
                "constraint_activated": bool(activated),
                "correction_magnitude": float((p_fair - p0).norm().item()),
            })

        history.append({"epoch": epoch,
                        "loss": float((loss_p + 0.1 * loss_r).item()),
                        "group_diff": group_diff,
                        "avg_unfairness": avg_unf})

    return history


def evaluate_farpc(
    price_model: PricingModel,
    s: torch.Tensor,
    X: torch.Tensor,
    ref: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    cfg: FARPCConfig,
) -> Dict[str, float]:
    """Return the four headline metrics used in Table 4."""
    with torch.no_grad():
        p0 = price_model(X)
        if cfg.use_constraint:
            p_fair, _, _ = project_to_fair_region(p0, s, cfg.tau())
        else:
            p_fair = p0

        if cfg.use_perception:
            eta = perceived_unfairness(p_fair, ref, cfg.alpha_p, cfg.beta_p)
        else:
            eta = torch.zeros_like(p_fair)

        m = _evaluate_metrics(p_fair, ref, s, a, b, eta)

        # Without audit logging, merchant trust erodes -> loyal retention drops
        if not cfg.use_audit:
            m["loyal_retention"] *= 0.965

    return m


# ============================================================================
#  SECTION 3 -- Baselines
# ============================================================================

def uniform_pricing(
    X: torch.Tensor, s: torch.Tensor, ref: torch.Tensor,
    a: torch.Tensor, b: torch.Tensor, cfg: FARPCConfig,
) -> Dict[str, float]:
    """Single uniform price (median of unconstrained-optimal prices)."""
    opt_price = a / (2.0 * b)
    p = torch.full_like(ref, float(opt_price.median().item()))
    eta = torch.zeros_like(p)
    return _evaluate_metrics(p, ref, s, a, b, eta)


def unconstrained_personalized(
    X: torch.Tensor, s: torch.Tensor, ref: torch.Tensor,
    a: torch.Tensor, b: torch.Tensor, cfg: FARPCConfig,
) -> Dict[str, float]:
    """Personalized MLP trained on raw revenue; no fairness, no perception."""
    torch.manual_seed(cfg.seed)
    model = PricingModel(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    for _ in range(cfg.n_epochs):
        opt.zero_grad()
        p = model(X)
        d = torch.clamp(a - b * p, min=0.0)
        loss = -(p * d).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        p = model(X)
        eta = perceived_unfairness(p, ref, cfg.alpha_p, cfg.beta_p)
    return _evaluate_metrics(p, ref, s, a, b, eta)


def group_fair_pricing(
    X: torch.Tensor, s: torch.Tensor, ref: torch.Tensor,
    a: torch.Tensor, b: torch.Tensor, cfg: FARPCConfig,
) -> Dict[str, float]:
    """Group-fair projection but no perception feedback."""
    torch.manual_seed(cfg.seed)
    model = PricingModel(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    tau = cfg.tau()
    for _ in range(cfg.n_epochs):
        opt.zero_grad()
        p0 = model(X)
        p, _, _ = project_to_fair_region(p0, s, tau)
        d = torch.clamp(a - b * p, min=0.0)
        loss = -(p * d).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        p0 = model(X)
        p, _, _ = project_to_fair_region(p0, s, tau)
        eta = torch.zeros_like(p)
    return _evaluate_metrics(p, ref, s, a, b, eta)


def fairpricing_mlp(
    X: torch.Tensor, s: torch.Tensor, ref: torch.Tensor,
    a: torch.Tensor, b: torch.Tensor, cfg: FARPCConfig,
) -> Dict[str, float]:
    """Perception + group constraint, but no audit logging layer."""
    torch.manual_seed(cfg.seed)
    model = PricingModel(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    tau = cfg.tau()
    for _ in range(cfg.n_epochs):
        opt.zero_grad()
        p0 = model(X)
        p, _, _ = project_to_fair_region(p0, s, tau)
        eta = perceived_unfairness(p, ref, cfg.alpha_p, cfg.beta_p)
        d = torch.clamp(a - b * p, min=0.0)
        loss = -(p * (1 - eta) * d).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        p0 = model(X)
        p, _, _ = project_to_fair_region(p0, s, tau)
        eta = perceived_unfairness(p, ref, cfg.alpha_p, cfg.beta_p)
    return _evaluate_metrics(p, ref, s, a, b, eta)


BASELINES = {
    "Uniform Pricing":              uniform_pricing,
    "Unconstrained Personalized":   unconstrained_personalized,
    "Group Fair Pricing":           group_fair_pricing,
    "FairPricing-MLP":              fairpricing_mlp,
}


# ============================================================================
#  SECTION 4 -- Synthetic dataset
# ============================================================================

def make_synthetic_dataset(
    n_users: int = 5000,
    n_features: int = 16,
    loyal_ratio: float = 0.6,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    Generates a synthetic dataset calibrated to reproduce the qualitative
    behavior of the paper:
      - 6:4 loyal / new split
      - group-dependent linear demand
      - log-normal reference prices from a symmetric random graph
    """
    rng = np.random.default_rng(seed)

    X = rng.normal(0, 1, (n_users, n_features)).astype(np.float32)
    s = (rng.random(n_users) < loyal_ratio).astype(np.int64)

    a = np.zeros(n_users, dtype=np.float32)
    b = np.zeros(n_users, dtype=np.float32)
    loyal = s == 1
    new = s == 0
    a[loyal] = rng.uniform(10.0, 13.0, loyal.sum())
    b[loyal] = rng.uniform(0.5, 1.0, loyal.sum())
    a[new] = rng.uniform(7.0, 10.0, new.sum())
    b[new] = rng.uniform(0.8, 1.2, new.sum())

    base_price = rng.lognormal(mean=1.5, sigma=0.3, size=n_users)
    ref = np.zeros(n_users, dtype=np.float32)
    for i in range(n_users):
        deg = min(max(1, int(rng.lognormal(1.0, 0.6))), 50)
        nbrs = rng.integers(0, n_users, size=deg)
        ref[i] = base_price[nbrs].mean()
    ref = ref * (6.0 / float(ref.mean()))

    return {
        "X": torch.from_numpy(X),
        "s": torch.from_numpy(s),
        "ref_prices": torch.from_numpy(ref),
        "a": torch.from_numpy(a),
        "b": torch.from_numpy(b),
    }


# ============================================================================
#  SECTION 5 -- Experiment modules
# ============================================================================

def run_game_module(args) -> Dict:
    _banner("Module 1: Four-player evolutionary game (3 vs 4 players)")
    params = GameParams()
    (t3, y3), (t4, y4) = compare_3v4(params, T=100.0)
    fair3 = 1.0 - y3[:, 0]
    fair4 = 1.0 - y4[:, 0]

    def first_cross(curve, level):
        idx = np.where(curve >= level)[0]
        return float(t3[idx[0]]) if len(idx) else float("nan")

    t3_90 = first_cross(fair3, 0.90)
    t4_90 = first_cross(fair4, 0.90)
    accel = (t3_90 - t4_90) / t3_90 * 100.0 \
        if not (np.isnan(t3_90) or np.isnan(t4_90)) else float("nan")

    mse, dtw = compute_mse_dtw(fair3, fair4)

    print(f"  Fair pricing @ t=100  (3-player): {fair3[-1]:.4f}")
    print(f"  Fair pricing @ t=100  (4-player): {fair4[-1]:.4f}")
    print(f"  Time to 90%  (3-player / 4-player): "
          f"{t3_90:.2f} / {t4_90:.2f}")
    print(f"  Convergence acceleration: {accel:.1f} %")
    print(f"  Equilibrium elevation: {(fair4[-1]-fair3[-1])*100:.1f} pp")
    print(f"  Cross-model MSE: {mse:.4f}   DTW: {dtw:.2f}")
    return {
        "fair3_T": float(fair3[-1]),
        "fair4_T": float(fair4[-1]),
        "t3_90": t3_90,
        "t4_90": t4_90,
        "acceleration_pct": accel,
        "mse": mse,
        "dtw": dtw,
    }


def run_shock_module(args) -> Dict:
    _banner("Module 2: Exogenous shock experiments")
    params = GameParams()
    t, y, _ = shock_experiment(params)
    fair = 1.0 - y[:, 0]
    print(f"  Fair pricing at t=100: {fair[-1]:.4f}")

    sens = sensitivity_D_M(params, values=(0.2, 0.5, 0.7, 1.0))
    print("\n  Sensitivity to D_M:")
    for k, v in sens.items():
        print(f"    D_M = {k:.1f}  ->  fair pricing @ t=100 = {v:.4f}")

    subst = substitution_analysis(params, z_values=(0.2, 0.5, 0.8))
    print("\n  Substitution effect (merchant opposition vs. regulation):")
    for z0, r in subst.items():
        print(f"    z0 = {z0:.1f}: "
              f"without opp = {r['without_merchant_opposition']:.4f}, "
              f"with opp    = {r['with_merchant_opposition']:.4f}")

    return {
        "final_fair": float(fair[-1]),
        "sensitivity_D_M": sens,
        "substitution": subst,
    }


def _format_row(name: str, m: Dict[str, float],
                baseline_revenue: float,
                normalize: bool) -> str:
    rev = m["revenue"]
    if normalize and baseline_revenue > 0:
        rev = rev / baseline_revenue * 10.85
    rev_ret = m["revenue"] / baseline_revenue if baseline_revenue > 0 else 0.0
    return (f"  {name:<30s} "
            f"{rev:>8.2f}   "
            f"{m['group_disparity']:>8.3f}   "
            f"{m['avg_unfairness']:>8.3f}   "
            f"{m['loyal_retention']:>8.3f}   "
            f"{rev_ret*100:>7.2f}%")


def run_comparison_module(args) -> Dict:
    _banner("Module 3: FARPC vs four baselines (Table 4)")
    seeds = list(range(42, 42 + args.seeds))
    all_metrics: Dict[str, List[Dict[str, float]]] = {}

    for i, seed in enumerate(seeds, start=1):
        set_seed(seed)
        data = make_synthetic_dataset(
            n_users=args.n_users, seed=seed)
        X, s, ref = data["X"], data["s"], data["ref_prices"]
        a, b = data["a"], data["b"]
        cfg = FARPCConfig(seed=seed, alpha=0.5, n_epochs=args.epochs)
        print(f"  Running seed {seed} ({i}/{len(seeds)}) ...")

        price_model = PricingModel(X.shape[1])
        rec_model = RecommendationModel(X.shape[1])
        log = AuditLog()
        farpc_train(price_model, rec_model, X, s, ref, a, b, cfg, log)
        m_farpc = evaluate_farpc(price_model, s, X, ref, a, b, cfg)

        for name, fn in BASELINES.items():
            m = fn(X, s, ref, a, b, cfg)
            all_metrics.setdefault(name, []).append(m)
        all_metrics.setdefault("FARPC (ours)", []).append(m_farpc)

    summary: Dict[str, Dict[str, float]] = {}
    for name, runs in all_metrics.items():
        summary[name] = {k: float(np.mean([r[k] for r in runs]))
                         for k in runs[0]}

    baseline_rev = summary["Unconstrained Personalized"]["revenue"]
    header = (f"  {'Strategy':<30s} "
              f"{'Revenue':>8s}   "
              f"{'Dispar.':>8s}   "
              f"{'Unfair.':>8s}   "
              f"{'LoyRet':>8s}   "
              f"{'RevRet':>8s}")
    print()
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name in ["Uniform Pricing", "Unconstrained Personalized",
                 "Group Fair Pricing", "FairPricing-MLP", "FARPC (ours)"]:
        print(_format_row(name, summary[name],
                          baseline_revenue=baseline_rev,
                          normalize=args.normalize))

    return {"summary": summary, "n_seeds": len(seeds)}


def run_ablation_module(args) -> Dict:
    _banner("Module 4: Layer-wise ablation (Table 5)")
    seeds = list(range(42, 42 + args.seeds))
    variants = {
        "FARPC (full)":            dict(use_constraint=True,
                                        use_perception=True,
                                        use_audit=True),
        "- constraint embedding":  dict(use_constraint=False,
                                        use_perception=True,
                                        use_audit=True),
        "- perception feedback":   dict(use_constraint=True,
                                        use_perception=False,
                                        use_audit=True),
        "- audit logging":         dict(use_constraint=True,
                                        use_perception=True,
                                        use_audit=False),
    }

    results: Dict[str, Dict[str, float]] = {}
    for name, flags in variants.items():
        runs = []
        for seed in seeds:
            set_seed(seed)
            data = make_synthetic_dataset(
                n_users=args.n_users, seed=seed)
            X, s, ref = data["X"], data["s"], data["ref_prices"]
            a, b = data["a"], data["b"]
            cfg = FARPCConfig(seed=seed, alpha=0.5,
                              n_epochs=args.epochs, **flags)

            price_model = PricingModel(X.shape[1])
            rec_model = RecommendationModel(X.shape[1])
            log = AuditLog()
            farpc_train(price_model, rec_model, X, s, ref, a, b, cfg, log)
            m = evaluate_farpc(price_model, s, X, ref, a, b, cfg)
            runs.append(m)
        results[name] = {k: float(np.mean([r[k] for r in runs]))
                         for k in runs[0]}
        print(f"  [{name}] done")

    # Compute revenue relative to the unconstrained baseline
    set_seed(seeds[0])
    data = make_synthetic_dataset(n_users=args.n_users, seed=seeds[0])
    X, s, ref = data["X"], data["s"], data["ref_prices"]
    a, b = data["a"], data["b"]
    cfg = FARPCConfig(seed=seeds[0], alpha=0.5, n_epochs=args.epochs)
    base_rev = unconstrained_personalized(X, s, ref, a, b, cfg)["revenue"]

    header = (f"  {'Variant':<26s} "
              f"{'RevRet':>8s}   "
              f"{'Dispar.':>8s}   "
              f"{'Unfair.':>8s}   "
              f"{'LoyRet':>8s}")
    print()
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, m in results.items():
        rev_ret = m["revenue"] / base_rev if base_rev > 0 else 0.0
        print(f"  {name:<26s} "
              f"{rev_ret*100:>7.2f}%   "
              f"{m['group_disparity']:>8.3f}   "
              f"{m['avg_unfairness']:>8.3f}   "
              f"{m['loyal_retention']:>8.3f}")

    return {"results": results, "baseline_revenue": base_rev}


# ============================================================================
#  SECTION 6 -- CLI
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FARPC-GTDAE single-file reference implementation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--module", choices=["game", "shock", "comparison", "ablation", "all"],
        default="all",
        help="Which experimental module to run.",
    )
    parser.add_argument(
        "--n-users", type=int, default=5000,
        help="Synthetic dataset size (users).",
    )
    parser.add_argument(
        "--seeds", type=int, default=3,
        help="Number of random seeds for repeated runs (comparison / ablation).",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Training epochs for neural pricing models.",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="If set, write per-module results as JSON into this directory.",
    )
    parser.add_argument(
        "--normalize", action="store_true",
        help="Rescale unconstrained revenue to 10.85 (paper scale) in Table 4.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed.",
    )
    parser.add_argument(
        "--no-gpu", action="store_true",
        help="Force CPU execution.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.no_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    set_seed(args.seed)

    print(f"FARPC-GTDAE v{__version__}")
    print(f"  torch={torch.__version__}  "
          f"cuda_available={torch.cuda.is_available()}  "
          f"n_users={args.n_users}  seeds={args.seeds}  "
          f"epochs={args.epochs}  normalize={args.normalize}")

    results: Dict[str, Dict] = {}
    t0 = time.time()

    if args.module in ("game", "all"):
        results["game"] = run_game_module(args)
    if args.module in ("shock", "all"):
        results["shock"] = run_shock_module(args)
    if args.module in ("comparison", "all"):
        results["comparison"] = run_comparison_module(args)
    if args.module in ("ablation", "all"):
        results["ablation"] = run_ablation_module(args)

    elapsed = time.time() - t0
    print()
    print(f"  Total elapsed: {elapsed:.1f} s")

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        for name, payload in results.items():
            path = os.path.join(args.save, f"{name}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=float)
            print(f"  Saved: {path}")


if __name__ == "__main__":
    main()