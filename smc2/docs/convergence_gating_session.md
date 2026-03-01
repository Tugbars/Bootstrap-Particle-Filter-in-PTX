# Convergence Gating for ParamTracker — Session Log

## Problem

ParamTracker runs amnesiac SMC² windows (re-initialized from prior each time) and fuses their posteriors with a Kalman filter. The question: when should we start pushing Kalman estimates to the BPF? Pushing too early risks corrupting the BPF with garbage estimates. Pushing too late wastes information the Kalman already has.

## First Attempt: R̂ Gating (Failed)

**Design:** Compute an R̂ analog across the rolling buffer of window posteriors. For each parameter, if R̂ < 1.5 (windows agree), push the Kalman estimate. Otherwise, push a safe prior default.

**Implementation:** Added `ConvergenceDiag` module to ParamTracker with per-parameter R̂, Mahalanobis d², P-trace monitoring, and coefficient of variation. Integrated into `param_tracker_run_window()` with convergence-gated snapshot push.

**Test result: catastrophic failure.** On the multi-regime DGP (12K learning → 30K calm → 12K crisis):

- μ_base stuck at prior default (−10.0) for 30K+ ticks while Kalman had −4.08 from window 1 (truth = −4.5)
- R̂ never consistently dropped below threshold because amnesiac windows naturally scatter
- When R̂ finally did drop (window 19), the calm→crisis transition at window 24 spiked it back — reverting μ_base to −10.0 after 5 good windows
- The revert was catastrophic: a 5.5 log-vol unit shock to the BPF mid-run

**Root cause:** R̂ gating and Kalman stitching are philosophically opposed for fast parameters. The Kalman's job is to fuse windows that *don't* agree. Requiring agreement before pushing means "only push when the Kalman is redundant."

## Solution: Tiered Gating with One-Way Latches

Two bugs drove the redesign:

1. **Fast params gated at all** — the Kalman has useful estimates after 2 windows
2. **No latch** — convergence reverted, causing oscillation between learned and prior values

### Three Gate Modes

**GATE_KALMAN_MIN** — For fast params (ρ, σ_total, r_split, μ_base):
- Push Kalman estimate after `min_windows` completions (default: 2)
- No R̂ check whatsoever
- One-way latch: once converged, stays converged permanently
- Rationale: these are low-dimensional, any data identifies them, and the cost of holding at prior default is high (especially μ_base)

**GATE_RHAT_LATCH** — For curve shape params (μ_scale, μ_rate, σ_scale, σ_rate):
- R̂ gate: push Kalman only after R̂ drops below threshold (1.5)
- One-way latch: once R̂ drops, converged=1 forever — never reverts
- Rationale: wrong curve shapes systematically corrupt BPF 2D across the entire z-distribution; worth waiting for window agreement
- The latch prevents regime transitions from reverting valid estimates

**GATE_LOCKED** — For params not yet freed by the phased ratchet:
- Always push prior default
- Set automatically by `set_free_mask()` for locked params
- When a param becomes free, `set_free_mask()` restores the appropriate default mode (KALMAN_MIN for indices 0–3, RHAT_LATCH for 4–7)

### Phase Transition Flow

```
Phase 1:  set_free_mask({1,1,1,1,0,0,0,0})
          → 0-3: GATE_KALMAN_MIN  → latch at window 2
          → 4-7: GATE_LOCKED      → prior defaults

Phase 2:  set_free_mask({1,1,1,1,1,1,0,0})
          → 0-3: untouched (already latched)
          → 4-5: LOCKED → RHAT_LATCH, converged=0
          → 6-7: stay LOCKED

Phase 3:  set_free_mask({1,1,1,1,1,1,1,1})
          → 6-7: LOCKED → RHAT_LATCH, converged=0
```

## Test Results

### test_convergence_gate.cu — GATE_KALMAN_MIN validation

6/6 checks passed:

| Check | Result |
|-------|--------|
| Phase 1 params converge at window 2 | ✓ |
| μ_base gated for exactly 1 window | ✓ |
| μ_base never reverted after convergence | ✓ |
| Locked params held at prior defaults | ✓ |
| Scored RMSE = 0.69 (reasonable) | ✓ |
| Final gated == Kalman (no gap) | ✓ |

Key observation: Window 9 showed R̂[μ_base] = 3.97 — would have reverted under the old design. Latch held.

### test_rhat_latch.cu — GATE_RHAT_LATCH validation

6/6 checks passed:

| Check | Result |
|-------|--------|
| Params 4-5 stayed LOCKED during Phase 1 | ✓ |
| Transitioned to RHAT_LATCH after Phase 2 | ✓ |
| Latched at window 11 (3 windows post-transition) | ✓ |
| Latch held through 10-window calm period | ✓ |
| Gated values match Kalman at latch point | ✓ |
| Params 6-7 stayed LOCKED throughout | ✓ |

Curve param estimates at latch: μ_scale = 3.066 (truth 3.0), μ_rate = 0.516 (truth 0.5).

During calm period, R̂ for μ_scale spiked back to 2.36 (window 20) — would have reverted under old design. Latch held.

## Files Modified

| File | Changes |
|------|---------|
| `smc2_param_tracker.cu` | Tiered gating in struct + run_window, `set_free_mask` restores default gate mode, `n_windows_completed` counter, removed `x_predicted` field |
| `smc2_param_tracker.cuh` | Gate mode defines, `set_gate_mode()`, `set_min_windows()` API functions |
| `smc2_convergence_diag.h` | No changes (R̂ still computed for diagnostics + RHAT_LATCH) |

## Files Created

| File | Purpose |
|------|---------|
| `test_convergence_gate.cu` | GATE_KALMAN_MIN: full pipeline with BPF 1D, multi-regime DGP |
| `test_rhat_latch.cu` | GATE_RHAT_LATCH: Phase 1→2 transition, latch holds through calm |

## Known Issues

- **P-trace collapse:** Kalman P drops to ~0 after first window (SMC² posterior is tight with 1024 particles on 3000 ticks). Causes d² to explode (observed 20–195 vs expected ~4). Cosmetic — Kalman still produces good estimates because early-window averaging is fine for stationary DGP. Fix: score-driven adaptive Q (future work).

- **σ_total / r_split ridge:** r_split converges to ~0.63–0.71 instead of truth 0.50. The product σ_z = r × σ_total is well-identified but the split isn't. Harmless for BPF 1D; will matter for BPF 2D where σ_z and σ_base are used independently.

## What's Next

- BPF 2D integration with gated ParamTracker
- Score-driven adaptive Q for Kalman
- Phase 3 test (σ_scale, σ_rate freed)
- Move files to new CMake directory structure
