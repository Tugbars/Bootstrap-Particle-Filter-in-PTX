# SMC²-BPF Pipeline: What We Learned

## The Question We Set Out to Answer

Does adding a Kalman parameter tracker on top of the SMC² sliding-window estimator
improve BPF volatility estimation?

**Answer: No.** The Kalman layer is redundant at best, harmful at worst. The SMC²
particle cloud already performs multi-window fusion naturally — it doesn't need
external smoothing. What it needs is *curation*: controlling which parameters
the cloud is allowed to explore based on what the current data can identify.

This document records the full investigation: the hypothesis, the tests, the
failures, and the architecture that actually works.

---

## Background: The Pipeline

```
Observations (y_t, 1ms ticks)
    │
    ├──→ BPF (per-tick, ~600μs)
    │      Tracks h_t (log-vol) given μ, ρ
    │      Needs: correct μ at current stress level
    │
    └──→ SMC² (every stride ticks, sliding window)
           Estimates θ = (ρ, σ_total, r_split, μ_base, μ_scale, μ_rate, σ_scale, σ_rate)
           8 parameters describing the stochastic volatility model
           Particle cloud updates sequentially across windows
```

The μ(z) curve: `μ_base + μ_scale · (1 - exp(-μ_rate · z))` maps from stress
level z to mean log-volatility. At z=0 (calm), μ(0) = μ_base. At high z (crisis),
μ(z) approaches μ_base + μ_scale.

---

## The v1 Bug: Pushing μ_base

The original pipeline pushed `theta_mean[3]` = μ_base (the curve floor) to the BPF.

This was documented as a bug: during a crisis where z > 0, the true μ(z) is much
higher than μ_base. Example: μ_base = -4.5 (~10% vol) vs μ(z̄=2.25) = -2.5 (~30% vol).
The BPF was being told vol is 10% when the market was at 30%.

This motivated the Kalman tracker: evaluate the curve at z̄, smooth across windows,
push the corrected μ(z̄).

**What we didn't realize: the bug diagnosis was wrong.**

---

## The Discovery: The Traveling Cloud

The SMC² particle cloud is not reinitialized each window. It carries forward —
particles from window k become the prior for window k+1. The cloud migrates
through parameter space as new data arrives.

This changes everything:

- **μ_base is not the curve floor.** It's whatever value explains the current data
  best. When z is elevated, the cloud shifts μ_base upward to compensate. What we
  measured as μ_base = -1.78 (expected -4.50) wasn't a bug — it was the cloud
  correctly tracking the effective μ at the current stress level.

- **The cloud IS the multi-window fusion.** Each window's SMC² update incorporates
  all previous information through the particle history. There is no need for an
  external Kalman filter — the cloud already maintains the optimal (in the SMC sense)
  posterior given all data seen so far.

- **The Kalman was double-smoothing.** Applying a Kalman filter to the cloud's
  posterior mean is like averaging an average. It adds lag and bias without
  improving estimation.

---

## The Isolation Test

We built `test_pipeline_kalman.cu` to compare three push modes on identical data:

| Mode | What's pushed | Source |
|---|---|---|
| RAW | mu_base from cloud | SMC² posterior mean, no transformation |
| CURVE | eval_curve(mu_base, mu_scale, mu_rate, z̄) | Raw posterior + curve evaluation |
| KALMAN | Kalman-smoothed mu(z̄) | Kalman layer + curve evaluation |

### Results (every configuration we tried)

**Single-regime, all 8 params free:**
RAW = 0.85, CURVE = 1.48, KALMAN = 1.57. RAW wins by 75%.

**Single-regime, curve shape params fixed to truth:**
RAW = 1.05, CURVE = 1.60, KALMAN = 1.47. RAW wins by 40%.

**Multi-regime learning phase, then scored test phase:**
RAW = 0.99, CURVE = 1.98, KALMAN = 1.48. RAW wins by 50%.

**Every configuration:** RAW wins. KALMAN always worse than RAW. CURVE worse still.

---

## Why RAW Wins: The Ridge Manifold

The curve `μ(z) = μ_base + μ_scale · (1 - exp(-μ_rate · z))` has three parameters
but observations at a single z-level constrain only one degree of freedom: the
value μ(z̄).

Infinitely many (μ_base, μ_scale, μ_rate) triples produce the same μ(z̄). The
likelihood surface is a ridge — a 2D manifold where all points explain the data
equally well.

The traveling cloud handles this gracefully: it absorbs the full curve effect into
μ_base, which converges to the correct μ(z̄) regardless of what μ_scale and μ_rate
are doing. The BPF doesn't need the decomposition — it just needs the right number.

CURVE and KALMAN break because they try to decompose something that can't be
decomposed, then add the curve correction on top of a μ_base that already includes
it. Double-counting.

### The entanglement goes deeper than the curve

Even with curve shape params fixed to truth, the ridge persists between
(μ_base, r_split, σ_total). r_split controls how variance is allocated between
the z-process and h-process. A different r_split shifts the RBPF's internal z
estimates, which shifts where μ_base needs to sit. At a single z-level, these
trade off against each other.

The only way to break this ridge is to have z vary *within a single SMC² window*.
And even when we forced that with alternating z_bias segments shorter than the
window size, the cloud still preferred absorbing everything into μ_base because
ρ = 0.98 makes z transitions slow enough that within-window z-diversity was limited.

---

## Why Curve Evaluation Is Solving the Wrong Problem

The curve evaluation would beat the traveling cloud only if z changes faster than
the cloud can migrate. With stride = 1500 ticks (1.5 seconds at 1ms), that requires
a discontinuous jump in volatility regime — not a transition, an instant teleport.

With ρ = 0.98, even a crisis transition unfolds over hundreds of ticks. Market
volatility doesn't jump from 10% to 47% in 1.5 seconds. And in the scenario where
it does, your risk limits should have killed the position before the parameter
tracker matters.

The curve decomposition solves a theoretically elegant but practically nonexistent
problem, at the cost of an identifiability nightmare that makes everything worse
in the common case.

---

## What Actually Works: RAW + Phased Learning

The best results across all our testing came from the simplest architecture:
the traveling SMC² cloud with RAW mu_base push and phased parameter learning.

### Why phased learning helps (but not for the reason we thought)

The original purpose of phased learning was to enable curve identification:
fix unidentifiable params early, unlock them as regime diversity arrives, eventually
identify the full curve shape.

**The real purpose is particle health.** With all 8 params free, the cloud spreads
along the ridge manifold. Particles waste their diversity exploring (μ_base, μ_scale)
combinations that are observationally equivalent. This bleeds ESS without improving
estimation of the params that matter (ρ, σ_total, r_split, μ_base).

Phased learning keeps the cloud efficient by fixing params the data can't currently
constrain, concentrating particle diversity where it actually helps.

### The key insight: phased learning should be bidirectional

The current Phase 1→2→3 system is a one-way ratchet. Once ceiling params are
unlocked, they stay unlocked forever. But identifiability comes and goes:

```
Crisis hits → z-range widens → ceiling params identifiable → unlock ✓
Crisis ends → z narrows to z≈0 → ceiling params unidentifiable → still unlocked ✗
                                  → cloud drifts along ridge
                                  → ESS bleeds
                                  → good crisis estimates get corrupted
```

The fix: make phased learning a **valve that opens and closes** with the
regime diversity in the current data.

- Track z-range within each SMC² window
- Wide z-range → unlock curve params (cloud can identify them)
- Narrow z-range → lock them back to their last well-identified values
- Unlock again when z-range widens

This preserves knowledge gained during crises (by locking to last good estimate),
keeps the cloud healthy during calm periods (by removing degenerate dimensions),
and is ready to learn again when the next crisis provides new information.

---

## The Correct Production Architecture

```
Per-tick:
  BPF receives mu_base from SMC² cloud (RAW push)
  No curve evaluation. No Kalman smoothing.

Per-window:
  SMC² cloud updates (particles travel, not reinitialized)
  Push mu_base to BPF
  Push rho to BPF

Bidirectional phased controller:
  Monitor z-range within current window
  Wide range → unlock ceiling/rate params
  Narrow range → lock to last well-identified values
  Lightweight, no additional computation
```

### What gets removed

- Kalman parameter tracker (smc2_param_tracker.cu/cuh)
- Curve evaluation in the push path (eval_curve_host)
- ParamSnapshot machinery
- Kalman process noise tuning (DriftConfig, Q matrices)
- P_floor management

### What remains

- SMC² with traveling particle cloud
- BPF with external μ, ρ push
- Bidirectional phased controller (z-range → parameter mask)
- Simple RAW push: `gpu_bpf_set_mu(bpf, theta_mean[IDX_MU_BASE])`

---

## Summary of Hours Spent

| What we tried | What we learned |
|---|---|
| Kalman fusion of SMC² posteriors | Redundant — cloud already does multi-window fusion |
| Curve evaluation at z̄ | Unidentifiable — ridge manifold makes it double-count |
| Fixed curve params + Kalman | Still fails — (mu_base, r_split) ridge persists |
| Multi-regime learning phase | Cloud absorbs curve into mu_base regardless |
| Alternating z-bias segments | Within-window diversity helps but ρ=0.98 limits it |
| RAW push with traveling cloud | Works. The simplest approach was correct all along |

The Kalman tracker was a well-engineered solution to the wrong problem. The
traveling cloud doesn't need external fusion. It needs a valve that controls
which dimensions it explores, driven by the information content of the current data.

**Next step: implement bidirectional phased learning.**
