# Two-Layer Latent Inference: What the Model Requires vs What the BPF Actually Does

## The Generative Model Has Three Layers

The regime-switching stochastic volatility model describes three layers, only one observed:

**Layer 0 — Market stress z (deepest latent):**
z̃ follows an AR(1) in unconstrained space, mapped to z ∈ (0, 3) via tanh squashing. This is a continuous market stress regime. Low z is calm, high z is crisis. Governed by ρ (persistence) and σ_z (innovation noise). Never directly observed.

**Layer 1 — Log-volatility h (shallow latent):**
h dynamics are entirely governed by the current stress level z through three regime-dependent curves:

- μ(z) = μ_base + μ_scale · (1 − e^(−μ_rate · z)) — local mean of h
- σ_h(z) = σ_base + σ_scale · (1 − e^(−σ_rate · z)) — innovation noise of h
- θ(z) = θ_base + θ_scale · (1 − e^(−θ_rate · z)) — mean-reversion speed of h

The transition: h_t = (1 − θ(z)) · h_{t-1} + θ(z) · μ(z) + σ_h(z) · ε

h doesn't have autonomous dynamics. Everything about it — level, noise, speed — is controlled by stress.

**Layer 2 — Returns y (observed):**
y_t = exp(h_t / 2) · ε_t, ε ~ N(0,1). The only thing we see.


## What I Assumed: BPF Tracks (z, h) Jointly

The natural way to do inference in this model is a particle filter over the joint state (z̃, h). Each particle carries both a stress value and a volatility value. At each tick:

1. Propagate z̃ → z (using ρ, σ_z)
2. Evaluate curves at z → get μ(z), σ_h(z), θ(z)
3. Propagate h using the curve values
4. Weight by observation likelihood p(y | h)

The RBPF variant replaces step 3-4 with exact Kalman conditioning — particle-approximate z, analytically integrate h.


## What the Production BPF Actually Does: Single-Layer OU on h

The production vol tracker (14 hand-written PTX kernels, `bpf_kernels.ptx`) does NOT implement the two-layer model. Kernel 3 (`bpf_propagate_weight`), the hot kernel called every tick, implements:

```
h' = μ + ρ · (h − μ) + σ_z · ε
```

This is a single-layer OU process. The parameters `rho`, `sigma_z`, `mu` are scalars passed as kernel arguments — not arrays, not z-dependent, not per-particle varying. The PTX file contains zero references to any z-like state variable. No `d_z` array, no curve evaluation, no z̃ propagation.

Evidence from the code:

- **State struct** (`GpuBpfState`): Contains `d_h` (particle states) but no `d_z`. Each particle is a single float.
- **Kernel 3 params**: `param_rho`, `param_sigma_z`, `param_mu` — all `.f32` scalars, not device pointers.
- **PTX line 500**: `OU transition: h' = mu + rho*(h - mu) + sigma_z*eps` — no curve evaluation, no z dependence.
- **grep for z-layer**: Zero hits for `z_tilde`, `d_z`, `curve`, `mu_z`, `sigma_h`, `theta_z`, `propagate_z` in the entire 1800-line PTX file.

The filter tracks volatility directly from returns through a fixed-parameter mean-reverting process. There is no stress layer.


## What the BPF Does Instead of a z Layer

The missing z layer is compensated by four host-side mechanisms bolted onto the single-layer filter:

**1. Adaptive σ_z bands (MixBandConfig):**
The host classifies market state as calm/alert/panic based on a surprise EMA. Each regime uploads different σ_z scale factors to constant memory. Particles are split into contiguous index bands with different σ_z multipliers.

This approximates σ_h(z): in the full model, high stress produces larger h noise via the curve. The bands do the same thing reactively — scale up σ_z when the host detects high surprise. But switching is reactive (based on past surprise), not predictive (based on a persistent stress process).

**2. Jump diffusion (Kernel 15, Bernoulli MIM):**
Each particle draws Bernoulli(λ). With probability λ (set by regime), it gets a large perturbation: h += σ_J · N(0,1). Handles structural breaks that the smooth OU can't capture.

This compensates for regime transitions in z. In the full model, rapid z shifts (driven by σ_z) produce sudden changes in h dynamics through the curves. Jump diffusion achieves a similar effect by directly perturbing h, bypassing the missing mechanism entirely.

**3. Online μ learning (Kernel 14):**
Natural gradient + Fisher information for μ (and optionally ρ). Updated via Robbins-Monro.

This compensates for μ(z). In the full model, h's mean level depends on stress via the μ(z) curve. When stress shifts, the target mean shifts. Without z, the BPF has a fixed μ and kernel 14 slowly chases the effective mean by gradient descent. But it can't distinguish "μ should be higher because stress increased" from "μ should be higher because fundamentals changed."

**4. CUSUM gating:**
Two CUSUM detectors on log-likelihood and gradient norms. When either crosses a threshold, jumps activate and regime may switch.

In the full model, changepoint-like behavior emerges naturally from z̃ dynamics. CUSUM detects the *symptom* (filter prediction failure) rather than modeling the *cause* (stress transition).


## Why the Compensations Work (and Why They're Fragile)

The production BPF works because markets spend most time in quasi-stationary regimes where fixed-parameter OU is adequate, the adaptive bands track slow shifts, and jump diffusion handles rare abrupt transitions.

But the compensations are fragile:

- **Reactive, not predictive** — the system detects regime change AFTER it happens
- **Independent mechanisms** — σ_z scaling, μ adaptation, and jump rates don't know about each other. In the full model, all curve changes are correlated through z
- **Coarse discretization** — calm/alert/panic is 3 states approximating a continuous process
- **Manual tuning** — CUSUM thresholds, surprise EMA decay, jump σ_J, band fractions are engineering choices substituting for model parameters


## The Inference Direction in the Full Model

The generative model goes forward: z → curves → h dynamics → returns.

Inference goes backward:

**Returns → h:** The nonlinearity exp(h/2) blocks Kalman filtering. The OCSN linearization fixes this: log(y²) = h + log(χ²(1)) is linear in h with non-Gaussian noise. The 10-component Gaussian mixture approximates that noise, restoring Kalman tractability.

**h → z:** z is never inferred from returns directly. It's inferred from the *behavior* of h over time. If h jumps to a new level, becomes noisier, or mean-reverts faster — those are signatures of z having moved, because the curves translate z into h dynamics. Each z-particle proposes a stress level, the Kalman filter computes how well that level explains the observed h sequence, and particles with better explanations survive resampling.

The production BPF collapses both layers into one. It infers h directly from returns (standard observation weighting), with no mechanism to ask "what latent stress state would explain this h behavior?"


## Why σ_z Is the Hardest Parameter

σ_z controls innovation noise of z̃ — the deepest latent layer. Its influence on observations passes through four transformations:

σ_z → z̃ dynamics → z (tanh squash) → curves μ(z), σ_h(z), θ(z) → h dynamics → y

Each transformation dilutes the signal. The tanh squashing compresses extremes. The curves are saturating exponentials. By contrast, σ_base controls h noise directly (one layer deep), and ρ has a strong autocorrelation signature.

This is why the (σ_total, r_split) reparameterization matters: it separates the well-identified total noise from the weakly-identified split between channels.


## The 8+3 Architecture: Zero Offline Calibration

**8 learned by SMC² online:** ρ, σ_total, r_split, μ_base, μ_scale, μ_rate, σ_scale, σ_rate

**3 derived from sufficient statistics:** θ_base, θ_scale, θ_rate (from binned h-autocorrelation in the RBPF's Kalman means)

**0 calibrated offline.**

This replaces the production BPF's manual tuning (band fractions, CUSUM thresholds, surprise EMA, jump parameters) with model parameters that have physical meaning and are learned from data.


## What the Production BPF Got Right

The engineering decisions in the production BPF are not wrong — they're solutions to a real problem (non-stationarity) within the constraint of a single-layer model. The key insights carry forward:

- **Adaptive σ_z bands** → The RBPF gets this for free: σ_h(z) varies continuously with z per-particle
- **Jump diffusion** → May still be needed as a robustness mechanism for model misspecification
- **Online μ learning** → Subsumed by SMC² learning μ_base (and the full μ(z) curve)
- **CUSUM gating** → Could serve as an emergency fallback trigger, but the RBPF's predictive z-tracking should handle most regime transitions before CUSUM would detect them
- **Surprise EMA** → Still useful as a diagnostic, even if no longer driving the filter's behavior

The two-layer model doesn't invalidate the production BPF — it explains why those heuristics were necessary and provides the structural alternative.
