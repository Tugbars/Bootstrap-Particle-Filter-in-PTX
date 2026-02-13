"""
Differentiable Bootstrap Particle Filter (dBPF)

PyTorch implementation for parameter learning in stochastic volatility models.
Runs on CUDA, uses reparameterized predict + stop-gradient resampling.

The key insight (Corenflos et al. 2021, Naesseth et al. 2018):
  Loss = -sum_t log( (1/N) sum_i w_t^i )
  where w_t^i = p(y_t | h_t^i) are importance weights on PREDICTED particles.
  Gradients flow through weights → particles → predict → theta.
  Resampling uses detached indices (biased but low-variance gradient).

Model:
  h_t = mu + rho * (h_{t-1} - mu) + sigma_z * eps_t     (state transition)
  y_t | h_t ~ Student-t(0, exp(h_t), nu)                 (observation)

Usage:
  python diff_bpf.py
  python diff_bpf.py --learn mu rho sigma_z --n_particles 512
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse
import time
from typing import Optional, Tuple, Dict, Set


# =============================================================================
# Utilities
# =============================================================================

def systematic_resample(log_w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Systematic resampling. Returns (indices, normalized_weights).

    Indices are detached (non-differentiable). Particle values indexed
    by these indices still carry gradients through PyTorch autograd.
    """
    N = log_w.shape[0]
    device = log_w.device

    # Normalized weights (differentiable)
    w = F.softmax(log_w, dim=0)

    # CDF + systematic sampling (non-differentiable)
    with torch.no_grad():
        cumsum = torch.cumsum(w, dim=0)
        # Fix numerical issues: last element should be exactly 1
        cumsum[-1] = 1.0
        u = torch.rand(1, device=device) / N
        positions = u + torch.arange(N, device=device, dtype=torch.float32) / N
        indices = torch.searchsorted(cumsum, positions).clamp(0, N - 1)

    return indices, w


def generate_sv_data(
    T: int, mu: float, rho: float, sigma_z: float,
    nu_obs: float = 5.0, seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate stochastic volatility data.

    h_t = mu + rho*(h_{t-1} - mu) + sigma_z*eps_t
    y_t = exp(h_t/2) * eta_t,  eta_t ~ Student-t(nu_obs)
    """
    torch.manual_seed(seed)
    h = torch.zeros(T)
    y = torch.zeros(T)

    h[0] = mu
    for t in range(1, T):
        h[t] = mu + rho * (h[t - 1] - mu) + sigma_z * torch.randn(1).item()

    chi2 = torch.distributions.Chi2(nu_obs)
    for t in range(T):
        vol = math.exp(h[t].item() / 2.0)
        eta = torch.randn(1).item() / math.sqrt(chi2.sample().item() / nu_obs)
        y[t] = vol * eta

    return y, h


# =============================================================================
# Differentiable BPF
# =============================================================================

class DiffBPF(nn.Module):
    """Differentiable Bootstrap Particle Filter.

    Parameters stored in unconstrained space:
        mu:       identity
        rho:      sigmoid(raw)      -> (0, 1)
        sigma_z:  softplus(raw)     -> (0, inf)
        nu:       2 + softplus(raw) -> (2, inf)
    """

    def __init__(
        self,
        n_particles: int = 512,
        learn: Optional[Set[str]] = None,
        init_params: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.N = n_particles

        if learn is None:
            learn = {"mu", "rho"}
        if init_params is None:
            init_params = {}

        p = {
            "mu": init_params.get("mu", -1.0),
            "rho": init_params.get("rho", 0.95),
            "sigma_z": init_params.get("sigma_z", 0.2),
            "nu": init_params.get("nu", 5.0),
        }

        # Unconstrained parameters
        def inv_sigmoid(x):
            x = max(min(x, 0.999), 0.001)
            return math.log(x / (1 - x))

        def inv_softplus(x):
            return math.log(math.exp(x) - 1.0) if x > 0.01 else x

        self.mu_raw = nn.Parameter(torch.tensor(p["mu"]))
        self.rho_raw = nn.Parameter(torch.tensor(inv_sigmoid(p["rho"])))
        self.log_sigma_z = nn.Parameter(torch.tensor(inv_softplus(p["sigma_z"])))
        self.nu_raw = nn.Parameter(torch.tensor(inv_softplus(p["nu"] - 2.0)))

        # Freeze non-learned
        self._learn = learn
        param_map = {
            "mu": self.mu_raw, "rho": self.rho_raw,
            "sigma_z": self.log_sigma_z, "nu": self.nu_raw,
        }
        for name, param in param_map.items():
            if name not in learn:
                param.requires_grad_(False)

    @property
    def mu(self):
        return self.mu_raw

    @property
    def rho(self):
        return torch.sigmoid(self.rho_raw)

    @property
    def sigma_z(self):
        return F.softplus(self.log_sigma_z)

    @property
    def nu(self):
        return 2.0 + F.softplus(self.nu_raw)

    def get_params(self) -> Dict[str, float]:
        return {
            "mu": self.mu.item(), "rho": self.rho.item(),
            "sigma_z": self.sigma_z.item(), "nu": self.nu.item(),
        }

    def predict(self, h: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """h_new = mu + rho*(h - mu) + sigma_z * eps  (reparameterized)"""
        return self.mu + self.rho * (h - self.mu) + self.sigma_z * eps

    def log_weights(self, h: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        """Log observation density: log p(y_t | h_i).

        Student-t(nu) with variance exp(h):
          log w = const - h/2 - (nu+1)/2 * log(1 + y^2 / (nu * exp(h)))
        """
        nu = self.nu
        # exp(h) = observation variance
        log_w = (
            torch.lgamma((nu + 1.0) / 2.0)
            - torch.lgamma(nu / 2.0)
            - 0.5 * torch.log(nu * math.pi)
            - 0.5 * h
            - (nu + 1.0) / 2.0 * torch.log1p(y_t * y_t / (nu * h.exp() + 1e-8))
        )
        return log_w

    def forward(
        self, y_seq: torch.Tensor,
        chunk_size: int = 50,
        h_init: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run BPF on y_seq.  Returns log_lik [T], h_mean [T], vol [T]."""
        T = y_seq.shape[0]
        N = self.N
        device = y_seq.device

        # Initialize from stationary distribution
        if h_init is None:
            with torch.no_grad():
                stat_std = self.sigma_z.detach() / (1.0 - self.rho.detach() ** 2 + 1e-8).sqrt()
                h = self.mu.detach() + stat_std * torch.randn(N, device=device)
        else:
            h = h_init.clone()

        ll_out = torch.zeros(T, device=device)
        h_mean_out = torch.zeros(T, device=device)
        vol_out = torch.zeros(T, device=device)

        for chunk_start in range(0, T, chunk_size):
            chunk_end = min(chunk_start + chunk_size, T)
            h = h.detach()  # truncated BPTT boundary

            for t in range(chunk_start, chunk_end):
                # 1. PREDICT (reparameterized)
                eps = torch.randn(N, device=device)
                h = self.predict(h, eps)

                # 2. LOG WEIGHTS: log p(y_t | h_i)
                log_w = self.log_weights(h, y_seq[t])

                # 3. MARGINAL LOG-LIKELIHOOD (the loss!)
                #    log p(y_t | y_{1:t-1}) ≈ log( (1/N) sum_i exp(log_w_i) )
                ll_out[t] = torch.logsumexp(log_w, dim=0) - math.log(N)

                # 4. WEIGHTED MEAN (for output/diagnostics)
                w_norm = F.softmax(log_w, dim=0)
                h_mean_out[t] = (w_norm * h).sum()
                vol_out[t] = (w_norm * (h / 2.0).exp()).sum()

                # 5. RESAMPLE (stop-gradient on indices, values carry grad)
                indices, _ = systematic_resample(log_w)
                h = h[indices]

        return {"log_lik": ll_out, "h_mean": h_mean_out, "vol": vol_out}


# =============================================================================
# Training
# =============================================================================

def train(
    model: DiffBPF,
    y_seq: torch.Tensor,
    h_true: Optional[torch.Tensor] = None,
    n_epochs: int = 300,
    lr: float = 0.01,
    chunk_size: int = 50,
    log_every: int = 10,
):
    device = y_seq.device
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=30, min_lr=1e-4
    )

    history = {"loss": [], "rmse": [], "params": []}
    t0 = time.time()

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        out = model(y_seq, chunk_size=chunk_size)
        loss = -out["log_lik"].mean()

        if not torch.isfinite(loss):
            print(f"  [{epoch}] NaN loss, skipping")
            continue

        loss.backward()

        # NaN grad guard
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                p.grad.zero_()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        # Gradient diagnostics
        if epoch % log_every == 0:
            for name, param in [("mu", model.mu_raw), ("rho", model.rho_raw),
                                ("sigma_z", model.log_sigma_z), ("nu", model.nu_raw)]:
                if param.requires_grad and param.grad is not None:
                    print(f"    grad_{name:8s}: {param.grad.item():+.6e}")

        optimizer.step()
        scheduler.step(loss.item())

        params = model.get_params()
        history["loss"].append(loss.item())
        history["params"].append(params)

        rmse = float("nan")
        if h_true is not None:
            with torch.no_grad():
                rmse = (out["h_mean"] - h_true.to(device)).pow(2).mean().sqrt().item()
        history["rmse"].append(rmse)

        if epoch % log_every == 0 or epoch == n_epochs - 1:
            elapsed = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"[{epoch:4d}/{n_epochs}] loss={loss.item():8.3f}  "
                f"RMSE={rmse:.4f}  "
                f"mu={params['mu']:+.4f}  rho={params['rho']:.4f}  "
                f"sig_z={params['sigma_z']:.4f}  nu={params['nu']:.2f}  "
                f"lr={lr_now:.1e}  ({elapsed:.1f}s)"
            )

    return history


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Differentiable BPF")
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument("--n_particles", type=int, default=512)
    parser.add_argument("--learn", nargs="+", default=["mu", "rho", "sigma_z"])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"Device: {device}")

    # Ground truth
    TRUE = {"mu": -1.0, "rho": 0.98, "sigma_z": 0.15, "nu": 5.0}

    # Generate data
    y, h_true = generate_sv_data(
        args.T, TRUE["mu"], TRUE["rho"], TRUE["sigma_z"],
        nu_obs=TRUE["nu"], seed=args.seed
    )
    y, h_true = y.to(device), h_true.to(device)
    print(f"Data: T={args.T}, y ∈ [{y.min():.3f}, {y.max():.3f}]")
    print(f"True: {TRUE}")

    # Wrong initial params (to test learning)
    wrong_init = {
        "mu": 0.0,         # true: -1.0
        "rho": 0.85,       # true: 0.98
        "sigma_z": 0.30,   # true: 0.15 (2x off)
        "nu": 5.0,         # frozen
    }

    model = DiffBPF(
        n_particles=args.n_particles,
        learn=set(args.learn),
        init_params=wrong_init,
    ).to(device)

    print(f"Learning: {', '.join(sorted(args.learn))}")
    print(f"Init: {model.get_params()}")

    # Baseline
    with torch.no_grad():
        out0 = model(y, chunk_size=args.chunk_size)
        rmse0 = (out0["h_mean"] - h_true).pow(2).mean().sqrt().item()
        ll0 = out0["log_lik"].mean().item()
    print(f"Baseline: RMSE={rmse0:.4f}, mean_ll={ll0:.3f}\n")

    # Train
    history = train(
        model, y, h_true,
        n_epochs=args.epochs, lr=args.lr, chunk_size=args.chunk_size,
    )

    # Final
    with torch.no_grad():
        out_f = model(y, chunk_size=args.chunk_size)
        rmse_f = (out_f["h_mean"] - h_true).pow(2).mean().sqrt().item()
        ll_f = out_f["log_lik"].mean().item()

    print(f"\n--- Results ---")
    print(f"RMSE:   {rmse0:.4f} -> {rmse_f:.4f}")
    print(f"LL:     {ll0:.3f} -> {ll_f:.3f}")
    print(f"Learned: {model.get_params()}")
    print(f"True:    {TRUE}")


if __name__ == "__main__":
    main()
