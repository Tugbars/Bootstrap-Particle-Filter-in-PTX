# The Ridge Problem and 4-Param Fast Mode

## The Problem

The mean curve has three parameters:

```
μ(z) = μ_base + μ_scale · (1 - exp(-μ_rate · z))
```

Over one 3000-tick window, the latent process visits a narrow z-range. Within that range, these three parameters are nearly interchangeable — raising μ_base by 0.5 produces the same observations as raising μ_scale by 0.3 and μ_rate by 0.2. The likelihood surface has a long flat ridge through all three.

When μ_base shifts from -1 to +1 at t=6000, the post-shift SMC² windows see higher mean levels but can't attribute the change correctly. They report "μ_base went up a little, μ_scale went up a little, μ_rate went up a little." The 2-unit shift is diluted across three correlated parameters.

This is a **likelihood identifiability problem**, not a tracker problem. We tried:

- Overlap correction, P floors, innovation gating, Joseph form → 4/8
- Q tuning, K_rejuv increase, cpmmh_rho decrease → 1/8
- Warm prior (Kalman → SMC² prior) → 0/8 (prior penalty blocks true value)
- Uniform priors → doesn't help (ridge is in the likelihood, not the prior)

The Kalman tracker is optimal for its job. The sensor (SMC² window) is biased. No tracker tuning fixes a biased sensor.

## Why Fixed Params Pass 8/8

When truth is constant, each window lands on a random spot on the ridge. Window 1 might over-estimate μ_base and under-estimate μ_scale. Window 2 does the opposite. The errors are zero-mean across windows. The Kalman averages them out. This works — 8/8 within 2σ, 2-3× tighter than Oracle.

## Why Drift Fails

After the shift, every window's ridge is displaced in the same direction. The bias is no longer zero-mean — it's systematic. The Kalman faithfully tracks the biased measurements and converges to the wrong answer.

## The Solution: 4-Param Fast Mode

Remove the ridge by removing the redundant parameters.

**Fast windows** (every 1500 ticks): SMC² learns only [ρ, σ_total, r_split, μ_base]. The curve params [μ_scale, μ_rate, σ_scale, σ_rate] are fixed constants inside the SMC² model. With only one mean-level parameter, there is no ridge. If the data says the mean shifted, μ_base absorbs 100% of the signal. The measurement is unbiased. The Kalman tracks it correctly.

**Curve calibration** (nightly/offline): Run a long 8-param SMC² on 50,000+ ticks. With that much data, the process visits a wide z-range and the three mean-curve params have genuinely different effects on the likelihood — the ridge dissolves. Update the fixed curve values for tomorrow.

## Architecture

```
Every 1500 ticks:   [y_t] → 4-param SMC² (curves fixed) → [ρ̂, σ̂, r̂, μ̂_base] → 4×4 Kalman
Nightly:            [y_1..y_50000] → 8-param SMC² → update [μ_scale, μ_rate, σ_scale, σ_rate]
Per tick:           BPF runs with current params from Kalman + fixed curves
```

The Kalman becomes a 4-state tracker. It never sees curve params. No ridge enters the system.

## Implementation

1. Add `smc2_cuda_set_fixed_params(state, mask, values)` — marks params as fixed (not sampled), injects constant values into the particle population
2. Modify `kernel_init_from_prior` to skip fixed params
3. Modify CPMMH to not propose on fixed params
4. Shrink Kalman to 4×4 (or mask the 8×8 — simpler, wastes trivial memory)
5. Test: drift scenario should now track μ_base shift correctly

## Why This Works

It exploits timescale separation. μ_base can shift intraday — it needs fast tracking. The curve shape (μ_scale, μ_rate) changes over weeks/months — it doesn't need per-window estimation. Trying to learn both timescales in one 3000-tick window is asking for trouble. Separate them.
