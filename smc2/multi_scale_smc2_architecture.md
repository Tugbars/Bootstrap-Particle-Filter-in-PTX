# Multi-Scale BPF with Regime-Gated SMC² Parameter Learning

## Core Idea

Parameters aren't non-stationary — they're **regime-dependent**. A single learner accumulating across regimes fits a compromise that's wrong everywhere. Instead, dedicate one SMC² learner per BPF layer. Each learner only accumulates evidence when its regime is active. Within each learner's evidence set, parameters are approximately stationary — pure batch accumulation works.

## Architecture

```
Layer 0 (calm)     BPF₀  ←→  SMC²₀   θ₀ = (ρ, σ_z, μ_base, σ_base)
Layer 1 (normal)   BPF₁  ←→  SMC²₁   θ₁ = (ρ, σ_z, μ_base, σ_base)
Layer 2 (stressed) BPF₂  ←→  SMC²₂   θ₂ = (ρ, σ_z, μ_base, σ_base)
Layer 3 (crisis)   BPF₃  ←→  SMC²₃   θ₃ = (ρ, σ_z, μ_base, σ_base)
```

## Per-Tick Logic

1. All K layers run their BPF forward step (cheap — inner RBPF propagation only)
2. Each layer produces predictive log-likelihood `log p(yₜ | y₁:ₜ₋₁, θₖ)`
3. The layer with highest predictive likelihood is the **active regime**
4. Only the active layer's SMC² updates outer weights and accumulates evidence
5. Inactive layers: inner RBPF propagates, outer weights frozen

## Regime Routing

No external classifier. The BPF layers *are* the classifier — the layer that best explains the current observation wins. This is a proper Bayesian model comparison at every tick, using the marginal likelihood each layer already computes.

## Why This Works

- **No forgetting:** Each learner is a pure batch accumulator. No drift models, no forgetting factors, no BOCPD, no hazard rates.
- **No knobs:** Data decides routing. No tuning required beyond the number of layers.
- **Regime memory:** When a regime recurs after months, its learner still holds the full posterior from every previous activation. It picks up where it left off with zero relearning.
- **Coherent estimates:** Layer 0 might see 50,000 calm ticks over weeks — razor-tight posterior. Layer 3 might see 2,000 crisis ticks total — wider posterior, but all crisis data, so estimates are coherent for that regime.
- **Proven components:** Each SMC² learner is the same batch 4-param RBPF learner already validated (4/4 within 2σ, 20% CPMMH acceptance, 300ms/1200 ticks).

## Remaining Work

- **Reparameterize (σ_z, σ_base) → (σ_total, r):** Breaks identification ridge between vol-of-vol channels. Bijective transform, no information loss. `σ_total = √(σ_z² + σ_base²)`, `r = σ_z / σ_total ∈ (0,1)`.
- **Student-t OCSN tables:** Current OCSN assumes Gaussian observations; BPF uses Student-t. Swap in the extended mixture tables from Omori et al. (2007) for the correct `log(t_ν²)` distribution. Same Kalman machinery, closes the observation model mismatch.
- **Layer differentiation:** How layers differ — separate priors, separate θ_curve shapes, or just separate evidence accumulation with shared priors. Shared priors with separate accumulation is simplest and already sufficient.
