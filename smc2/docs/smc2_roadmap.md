# SMC² + RBPF + BPF 2D — Improvement Roadmap

## Current Production Architecture

```
SMC²(1024 × 512 RBPF) → amnesiac windows, W=3000, stride=2000
    ↓  θ̂_k, Σ_k every ~2 seconds
Kalman(8×8) → fuse, smooth, accumulate
    ↓  filtered θ̂
BPF 2D(5K–30K) → per-tick (h, z̃) estimation
```

**Settled decisions:**
- Amnesiac windows + Kalman stitching beats traveling cloud (41% RMSE gain)
- Ratchet phased learning beats bidirectional valve (valve re-unlock latency hurts)
- BPF 2D beats RBPF 3D for production filtering (+3.3% exact obs likelihood)
- RBPF stays inside SMC² — low-variance likelihoods required for parameter learning
- OCSN 10-component mixture: acceptable approximation cost for Rao-Blackwellization
- RAW push: Kalman x[MU_BASE] → BPF directly, no eval_curve double-counting

**Learned params (8):** ρ, σ_total, r_split, μ_base, μ_scale, μ_rate, σ_scale, σ_rate

---

## 1. Warm-Start from Kalman (High Impact)

**Problem:** Amnesiac windows reinit from prior every window. The Kalman holds a
tight posterior after a few windows, but SMC² ignores it — each window spends 2–3
windows rediscovering what the Kalman already knows.

**Fix:** Sample initial θ-particles from N(x_kalman, P_kalman) instead of the
broad prior.

```
Current:  θ_i ~ N(prior_mean, prior_std²)     // wide, uninformative
Proposed: θ_i ~ N(kalman_x, α · kalman_P)     // tight, informed
```

The inflation factor α (e.g. 1.5–3.0) prevents overconfidence — the Kalman
posterior may be slightly stale, and the SMC² cloud needs room to explore.

**Expected benefit:**
- Faster convergence per window → sharper Σ_k
- Sharper Σ_k → stronger Kalman updates (K = P̄(P̄ + Σ_k)⁻¹ — small Σ_k → large K)
- Fewer wasted early-window particles sitting in low-likelihood prior regions
- Still bounded depletion (fresh particles each window, just better-placed)

**Risk:** If the Kalman drifts (bad crisis, model misspec), warm-start propagates
the error. Mitigation: fall back to prior init when Kalman P exceeds a threshold
or when a phase transition fires (new dimensions need prior exploration).

**Implementation:**
- Add `smc2_cuda_init_from_gaussian(smc2, mean[8], cov[8×8])` kernel
- ParamTracker decides: if `kalman.n_updates > 2`, warm-start; else prior
- Phase transitions always reinit from prior (new params need fresh exploration)

---

## 2. Adaptive Window Length (Medium Impact)

**Problem:** Fixed W=3000 is a compromise. During crises, z moves fast — params
drift within the window, violating the stationarity assumption. During calm, z
barely moves — 3000 ticks of near-identical data provides diminishing returns.

**Fix:** Scale window length by z-velocity or z-range.

```
z_speed = EMA(|Δz|)

if z_speed > crisis_threshold:
    W = W_short   (e.g. 1500)  — fast-moving params, short snapshots
elif z_speed < calm_threshold:
    W = W_long    (e.g. 5000)  — slow params, accumulate more signal
else:
    W = W_default (e.g. 3000)
```

**Expected benefit:**
- Crisis: shorter windows → less within-window drift → tighter per-window Σ_k
- Calm: longer windows → more data per estimate → tighter Σ_k from averaging
- Both cases produce better Kalman measurements

**Risk:** The Kalman's process noise Q assumes roughly uniform window spacing.
Variable W means variable Δt between measurements. Fix: scale Q proportionally
to W — `Q_eff = Q_base * (W / W_default)`.

**Dependency:** Needs BPF 2D wired in to get per-tick z_mean for z_speed.

---

## 3. Score-Driven Kalman Q (Medium Impact)

**Problem:** The Kalman predict step uses fixed process noise Q. But parameter
drift is regime-dependent — near-zero during calm, potentially large during
crisis transitions. Fixed Q either over-smooths during transitions or
under-smooths during calm.

**Fix:** Scale Q by observed z-velocity.

```
Q_t = Q_base * (1 + α * |Δz̄|)
```

Where Δz̄ is the change in window-mean z between consecutive windows.

**Expected benefit:**
- Calm periods: Q ≈ Q_base (small) → Kalman trusts accumulated history
- Regime transitions: Q inflated → Kalman adapts faster to new param regime
- Eliminates the tradeoff between responsiveness and stability

**Alternative:** Use the innovation-based adaptive Kalman approach:
```
e_k = z_k - H·x̄_k                         // innovation
S_k = H·P̄·H' + Σ_k                        // predicted innovation cov
Q_k = K·(e_k·e_k' - Σ_k)·K' + (I-K·H)·Q̂  // estimated Q from residuals
```
This is fully automatic but noisier. The z-velocity heuristic is simpler and
more interpretable.

---

## 4. Targeted CPMMH Rejuvenation (Medium Impact)

**Problem:** Standard CPMMH rejuvenation proposes new θ for all particles
uniformly. But most particles are fine — they sit near the posterior mode.
The problematic ones are on the ridge: high likelihood but extreme scale/rate
ratios. Rejuvenating good particles wastes compute.

**Fix:** Target rejuvenation at ridge-dwelling particles.

```
Ridge score: R_i = |log(scale_i * rate_i) - log(median_scale * median_rate)|
Rejuv probability: p_i ∝ softmax(β * R_i)
```

Particles far from the median scale×rate product get rejuvenated more often.
This collapses the ridge without disturbing well-placed particles.

**Expected benefit:**
- Faster ridge collapse per rejuvenation cycle
- Better ESS for the same compute budget
- More effective use of CPMMH (currently ~10% acceptance rate)

**Risk:** If the ridge is genuinely multimodal (different scale/rate combos
produce similar quality), targeted rejuvenation might prematurely collapse
to one mode. Mitigation: use soft targeting (β not too large).

---

## 5. Sufficient Statistics from RBPF (Low-Medium Impact)

**Problem:** After each window, we extract only θ-means and Σ from the SMC²
cloud. Each RBPF inner filter maintains a Kalman state for h — this contains
gradient information about how the likelihood responds to θ perturbations.
Currently discarded.

**Fix:** Extract per-θ score vectors (∂ log p(y|θ) / ∂θ) from the RBPF
sufficient statistics. Feed these to the Kalman as additional measurements
or use them to construct a Fisher information estimate.

```
Score: s_j = (1/N_inner) Σ_i ∂/∂θ log p(y_{1:T} | θ_j, h_i)
Fisher: F ≈ (1/N_theta) Σ_j w_j · s_j · s_j'
```

The Fisher tells the Kalman which parameter directions are well-identified
by this window's data. Poorly-identified directions get inflated uncertainty,
well-identified directions get tight updates.

**Expected benefit:**
- Anisotropic Kalman updates: tight along identified directions, wide along ridge
- Better handling of windows that only see one z-level (floor identified, scale not)
- More principled than the current isotropic Σ_k from particle covariance

**Risk:** Score estimation from particles is noisy. May need larger N_inner or
multiple passes. Compute cost is non-trivial.

**Dependency:** Requires access to RBPF Kalman internals per θ-particle. May
need kernel changes.

---

## 6. 7-Component OCSN Mixture (Low Impact, Easy Win)

**Problem:** The 10-component Kim-Shephard-Chib mixture evaluates 10 Gaussian
components per RBPF particle per tick. Three tail components have very low
probability and contribute mainly to extreme cases (near-zero returns).

**Fix:** Drop to 7 components, benchmark parameter recovery.

**Expected benefit:**
- 30% fewer mixture evaluations per tick across 524K particles
- Measurable kernel speedup

**Validation:** Run the parameter recovery test (stationary DGP) with both
7 and 10 components. If posterior means shift by < 0.3σ, take the speedup.

**Risk:** Near-zero returns during calm periods trigger the tail components.
If 7-component biases mu_base during calm, keep 10. Empirical question.

---

## Priority Order

| # | Improvement | Impact | Effort | Dependencies |
|---|-------------|--------|--------|-------------|
| 1 | Warm-start from Kalman | High | Medium | New CUDA kernel |
| 2 | Adaptive window length | Medium | Low | BPF 2D z_mean |
| 3 | Score-driven Q | Medium | Low | None |
| 4 | Targeted CPMMH | Medium | Medium | CPMMH already exists |
| 5 | RBPF sufficient stats | Low-Med | High | Kernel changes |
| 6 | 7-component OCSN | Low | Low | Benchmark only |

Items 1–3 are the next sprint. Item 6 is a free benchmark whenever convenient.
Items 4–5 are optimizations for after the core pipeline stabilizes.

---

## Resolved / Not Pursuing

- **Bidirectional valve:** Tested, ratchet wins. Valve re-unlock latency costs
  more than the benefit of locking during calm. Amnesiac windows + Kalman already
  handle calm-period noise gracefully (wide Σ → low Kalman gain).

- **BPF inside SMC²:** Tested, likelihood signal too noisy for parameter learning.
  RBPF required for the Rao-Blackwellization that gives clean likelihoods.

- **Traveling cloud:** Tested, degenerates after ~3K ticks regardless of particle
  count. Amnesiac + Kalman is the correct architecture.

- **eval_curve push:** Bug — double-counted curve correction. RAW push is correct.
