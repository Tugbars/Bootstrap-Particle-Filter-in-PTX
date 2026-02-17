# SMC² Parameter Architecture: From 4+7 to 8+3

## The Problem We Discovered

The current SMC² implementation learns 4 parameters online and treats 7 as fixed constants calibrated offline. This split was based on a false assumption: that the RBPF inner filter somehow needed fewer parameters than a BPF would. In reality, both filters run the exact same generative model forward and require the exact same 11 parameters. The RBPF doesn't remove parameters — it replaces sampling of h with exact Kalman conditioning on z̃. Same inputs, different computation.

This means the system has a hidden dependency. The 7 "fixed" curve parameters (μ_scale, μ_rate, σ_scale, σ_rate, θ_base, θ_scale, θ_rate) must come from an offline calibration stage. If that calibration is wrong, the 4 online-learned parameters become **compensatory** — they converge, the posteriors look tight, but the values are whatever best patches over the curve misspecification rather than physically meaningful quantities.

This has a concrete downstream consequence. If we ever want to use an RBPF vol tracker in production (instead of the current BPF), the model needs to be trustworthy. The BPF tracker was chosen specifically because it's robust to model misspecification — it samples h stochastically, which provides implicit exploration when the curves are wrong. An RBPF tracker commits to the exact conditional under the given curves. If the curves are wrong, the RBPF is confidently wrong. The offline calibration dependency is what forces us into BPF for the tracker.


## What We Already Had

The old SMC² + CPMMH code (before RBPF, before reparameterization) already learned **8 parameters** successfully:

```
Parameter         True       Est       Std     Err%  z-score
─────────────────────────────────────────────────────────────
rho             0.9500    0.9499    0.0189    -0.0%     0.01
sigma_z         0.1500    0.1287    0.0788   -14.2%     0.27
mu_base        -1.0000   -0.9963    0.2464    -0.4%     0.01
mu_scale        0.5000    0.5105    0.2877    +2.1%     0.04
mu_rate         1.0000    1.0394    0.4289    +3.9%     0.09
sigma_base      0.1500    0.1692    0.0357   +12.8%     0.54
sigma_scale     0.1000    0.1167    0.0422   +16.7%     0.39
sigma_rate      1.0000    1.1238    0.4437   +12.4%     0.28
─────────────────────────────────────────────────────────────
OVERALL: 8/8 within 2σ, 7/8 within 15% relative error
```

This was with a BPF inner filter and no reparameterization. All 8 parameters recovered, though σ_z had the widest relative uncertainty (52% posterior std / true value), consistent with its position at the bottom of the signal chain.

The cut from 8 to 4 was made under the mistaken belief that the RBPF architecture eliminated the need for the curve parameters. It didn't. It just hid the dependency behind an offline calibration stage.


## The Remaining 3: θ(z) Speed Curve

The 8 learned parameters cover the z̃ dynamics (ρ, σ_z), the μ(z) curve (μ_base, μ_scale, μ_rate), and the σ_h(z) curve (σ_base, σ_scale, σ_rate). That leaves three: θ_base, θ_scale, θ_rate — the mean-reversion speed curve θ(z).

These are the hardest to identify from data. θ(z) controls how fast h_t reverts to its local mean μ(z). This is a second-order dynamic — the data mostly constrains *where* h is, not the transient rate at which it got there. At typical values (θ_base ≈ 0.005), the signal-to-noise ratio of mean-reversion is extremely low relative to σ_h(z). Trying to learn θ_base, θ_scale, and θ_rate in the CPMMH alongside the other 8 would add 3 weakly-identified dimensions to the proposal, degrading acceptance rates for minimal gain.

But these 3 don't need to be learned by CPMMH at all.


## The Key Insight: θ Is Derivable from Sufficient Statistics

Once the RBPF has estimated the h trajectory and we know μ(z) and σ_h(z) from the 8 learned parameters, the mean-reversion speed is just the AR(1) persistence of h conditional on z:

> φ(z) = Cov(h_t, h_{t-1} | z_t ∈ bin) / Var(h_{t-1} | z_t ∈ bin)
>
> θ(z) = 1 − φ(z)

This is a direct regression quantity computable from the Kalman sufficient statistics the RBPF already maintains. No CPMMH proposals needed. No offline calibration. The φ-based estimator is also structurally independent of μ — it uses the autocorrelation of h, not the gap between h and μ. This breaks the circular dependency where θ_base and μ_base are statistically coupled through the OLS estimator.


## The Plan: 8 Learned + 3 Derived = 0 Offline

### Architecture

**Outer SMC² loop learns 8 parameters via CPMMH:**
- ρ — AR(1) persistence of z̃
- σ_total — total vol-of-vol (reparameterized)
- r — vol-of-vol split ratio (reparameterized)
- μ_base — base level of mean curve
- μ_scale — amplitude of mean curve
- μ_rate — steepness of mean curve
- σ_base — folded into σ_total and r via reparameterization
- σ_scale — amplitude of σ_h curve
- σ_rate — steepness of σ_h curve

Note: with the (σ_total, r) reparameterization, σ_base is derived as √(1 − r²) · σ_total. The 8 CPMMH dimensions are (ρ, σ_total, r, μ_base, μ_scale, μ_rate, σ_scale, σ_rate).

**θ curve derived from RBPF sufficient statistics:**
- θ_base, θ_scale, θ_rate computed from binned h-trajectory autocorrelation
- Updated periodically (not every tick — perhaps every 100–500 ticks)
- Uses φ-based estimator: φ = Cov(h_t, h_{t-1}) / Var(h_{t-1}) per regime bin
- Independent of μ estimates, breaking circular dependency

**Offline calibration required: none.**

### What Changes From Current Code

1. Expand ThetaParticlesSoA to carry 8 parameters instead of 4
2. Expand SVPrior and SVBounds to cover all 8
3. Extend the CPMMH proposal covariance from 4×4 to 8×8
4. Add sufficient statistics accumulators for the φ-based θ estimator inside the RBPF kernel
5. Add a periodic θ-curve update step that reads the accumulated stats and updates the θ curve
6. Consider increasing N_θ (currently 1024) to support the 8×8 adaptive covariance
7. Consider increasing K_rejuv from 5 to 8–10 for the higher-dimensional space

### Reparameterization in 8D

The (σ_total, r) reparameterization still applies and is still valuable — it breaks the identification ridge between σ_z and σ_base. The new parameters are:

- σ_total = √(σ_z² + σ_base²) — well-identified, perpendicular to ridge
- r = σ_z / σ_total — weakly identified, along the ridge

The six other parameters (ρ, μ_base, μ_scale, μ_rate, σ_scale, σ_rate) don't have known ridges between them and can be proposed in their natural coordinates. The adaptive Cholesky proposal will discover any correlations empirically.

### Expected Improvements Over Old 8-Parameter Code

The old code proved 8 parameters are learnable. The new architecture should do strictly better because:

- RBPF inner filter gives tighter likelihood estimates → more informative outer weights
- (σ_total, r) reparameterization breaks the σ_z/σ_base ridge → higher acceptance rates for vol-of-vol parameters
- Correlated CPMMH noise (ρ_corr = 0.99) → higher acceptance rates across all dimensions
- Adaptive Cholesky proposals → automatic discovery of parameter correlations

The old code achieved 8/8 within 2σ with a BPF inner filter and no reparameterization. With all three improvements, we should see tighter posteriors and faster convergence.

### Validation Plan

1. Run the 8-parameter SMC² on the same DGP as the old test (ρ=0.95, σ_z=0.15, etc.)
2. Compare posterior widths and z-scores against the old BPF results
3. Verify the φ-based θ estimator recovers θ_base, θ_scale, θ_rate from the learned h trajectory
4. Test on harder DGPs (low ρ, high σ_z, asymmetric regimes)
5. Verify T=3000 is sufficient for 8-parameter convergence
6. Profile latency — can we still do per-tick updates under 1ms with 8D CPMMH?

### If This Works: RBPF Vol Tracker Becomes Viable

The current production vol tracker uses BPF because the model can't be trusted (offline calibration may be wrong). If all 11 parameters are determined online — 8 learned, 3 derived — the model becomes trustworthy. At that point, the RBPF's exact Kalman conditioning is pure upside with no robustness penalty, and the BPF's stochastic exploration is unnecessary hedging against a problem that no longer exists.


## Summary

| Aspect | Old (4+7) | New (8+3) |
|---|---|---|
| CPMMH dimensions | 4 | 8 |
| Offline calibration | 7 params, periodic refit | None |
| θ curve | Fixed offline | Derived from sufficient stats |
| Self-contained | No | Yes |
| Compensatory risk | High (wrong curves → biased params) | Low (all curves learned/derived) |
| RBPF vol tracker viable | No (model not trustworthy) | Yes (model fully determined online) |
