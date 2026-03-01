/**
 * @file smc2_param_tracker.h
 * @brief Kalman Parameter Tracker — fuses sliding-window SMC² posteriors
 *
 * Architecture:
 *   SMC² (sensor) → θ̂_k, Σ_k every window → Kalman (tracker) → filtered θ, P
 *
 * The SMC² runs on a sliding window of T observations. Each window completion
 * produces a posterior mean θ̂ (8D) and covariance Σ (8×8). The Kalman filter
 * treats these as measurements of slowly-drifting true parameters:
 *
 *   State model:     θ_{k+1} = θ_k + w_k,       w_k ~ N(0, Q)
 *   Measurement:     z_k = θ_k + v_k,            v_k ~ N(0, Σ_k)
 *
 * H = I (identity), F = I (random walk). Plain linear Kalman — no EKF/UKF
 * needed.
 *
 * Q encodes parameter drift rates:
 *   - Fast:  μ_base                  (tracks market level, shifts weekly)
 *   - Medium: ρ, σ_total, r_split   (dynamics, shift over weeks-months)
 *   - Slow:  μ_scale, μ_rate, σ_scale, σ_rate  (curve shapes, months-years)
 *
 * The tracker also evaluates curves at the current z̄ estimate to produce
 * scalar parameters (μ, σ_h, θ) ready for the production BPF.
 *
 * Warm-start: SHELVED (+25% RMSE, feedback trap). Always cold-starts from prior.
 *
 * Convergence gating (v3):
 *   Per-parameter R̂ diagnostic tracks window agreement. The snapshot
 *   pushed to BPF uses Kalman estimates only for converged params
 *   (R̂ < threshold). Non-converged params hold at prior defaults.
 *   This prevents half-baked curve shapes from corrupting BPF 2D.
 *
 * Usage:
 *   ParamTracker* t = param_tracker_create(3000, 500, 1024, 512);
 *   // ... per tick:
 *   param_tracker_feed(t, y_obs);       // buffers observation
 *   if (param_tracker_window_ready(t))  // every `stride` ticks
 *       param_tracker_run_window(t);    // runs SMC², Kalman update
 *   // ... get filtered params for production BPF:
 *   ParamSnapshot snap;
 *   param_tracker_get_snapshot(t, &snap);
 *   // snap.mu, snap.sigma_h, snap.theta_speed — ready for BPF
 */

#ifndef SMC2_PARAM_TRACKER_CUH
#define SMC2_PARAM_TRACKER_CUH

#include "smc2_rbpf_batch.cuh"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Curve evaluation: f(z) = base + scale * (1 - exp(-rate * z))
 */
static inline float eval_curve_host(float base, float scale, float rate,
                                    float z) {
  return base + scale * (1.0f - expf(-rate * z));
}

/**
 * @brief Snapshot of filtered parameters for the production BPF
 *
 * Everything the production BPF needs, evaluated at the current z̄.
 */
typedef struct {
  /* Filtered 8D parameters (Kalman state) */
  float theta[N_PARAMS]; /**< [ρ, σ_total, r, μ_base, μ_scale, μ_rate, σ_scale,
                            σ_rate] */
  float P_diag[N_PARAMS]; /**< Diagonal of Kalman P (parameter uncertainty) */

  /* Derived physical parameters */
  float sigma_z;    /**< r · σ_total */
  float sigma_base; /**< √(1-r²) · σ_total */

  /* Curves evaluated at current z̄ */
  float z_mean;      /**< Posterior mean stress level */
  float mu;          /**< μ(z̄) — target for BPF μ parameter */
  float sigma_h;     /**< σ_h(z̄) — target for BPF σ_z parameter */
  float theta_speed; /**< θ(z̄) — mean-reversion speed at current stress */

  /* Diagnostics */
  int n_updates;          /**< Number of Kalman updates performed */
  float last_accept_rate; /**< SMC² acceptance rate from last window */
  float last_ess;         /**< SMC² outer ESS from last window */
} ParamSnapshot;

/**
 * @brief Process noise (Q) presets for different parameter drift rates
 */
typedef struct {
  float q_rho;         /**< ρ drift variance per window */
  float q_sigma_total; /**< σ_total drift variance */
  float q_r_split;     /**< r_split drift variance */
  float q_mu_base;     /**< μ_base drift variance (fastest) */
  float q_mu_scale;    /**< μ_scale drift variance (slow) */
  float q_mu_rate;     /**< μ_rate drift variance (slow) */
  float q_sigma_scale; /**< σ_scale drift variance (slow) */
  float q_sigma_rate;  /**< σ_rate drift variance (slow) */
} DriftConfig;

/**
 * @brief Opaque state for the parameter tracker
 */
typedef struct ParamTracker ParamTracker;

/* ═══════════════════════════════════════════════════════════════════════════
 * Lifecycle
 * ═══════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Create a parameter tracker
 * @param window_size   Number of observations per SMC² window (e.g., 3000)
 * @param stride        Ticks between window completions (e.g., 500)
 * @param N_theta       SMC² outer particle count
 * @param N_inner       SMC² inner (RBPF) particle count
 * @return              Tracker instance, or NULL on failure
 */
ParamTracker *param_tracker_create(int window_size, int stride, int N_theta,
                                   int N_inner);

/**
 * @brief Destroy tracker and free all resources (including SMC² instance)
 */
void param_tracker_destroy(ParamTracker *t);

/* ═══════════════════════════════════════════════════════════════════════════
 * Per-tick feed + window trigger
 * ═══════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Feed one observation into the circular buffer
 * @param y_obs  Observation (log(y²) for OCSN, or raw return — match your DGP)
 */
void param_tracker_feed(ParamTracker *t, float y_obs);

/**
 * @brief Check if enough new observations have accumulated to run a window
 * @return 1 if stride ticks have passed since last window, 0 otherwise
 */
int param_tracker_window_ready(const ParamTracker *t);

/**
 * @brief Run SMC² on the current window, then Kalman update
 *
 * Initializes SMC² from prior, runs on window_size observations,
 * extracts posterior, updates Kalman, runs convergence diagnostic,
 * and pushes convergence-gated snapshot.
 *
 * This is the expensive call — runs SMC² on `window_size` observations.
 * Call only when param_tracker_window_ready() returns 1.
 */
void param_tracker_run_window(ParamTracker *t);

/* ═══════════════════════════════════════════════════════════════════════════
 * Output
 * ═══════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Get the current filtered parameter snapshot
 * @param snap  Output snapshot with filtered params + BPF-ready values
 */
void param_tracker_get_snapshot(const ParamTracker *t, ParamSnapshot *snap);

/* ═══════════════════════════════════════════════════════════════════════════
 * Configuration
 * ═══════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Set process noise (drift rates) for the Kalman filter
 * @param drift  Per-parameter drift variance per window
 *
 * Defaults are set in param_tracker_create(). Call this to override.
 * Larger Q → faster tracking, more noise. Smaller Q → smoother, more lag.
 */
void param_tracker_set_drift(ParamTracker *t, const DriftConfig *drift);

/**
 * @brief Set the fixed θ(z) curve used for curve evaluation
 *
 * θ(z) is not learned by SMC² — it's derived from sufficient statistics.
 * Update this periodically if you compute φ-based estimates.
 */
void param_tracker_set_theta_curve(ParamTracker *t, float base, float scale,
                                   float rate);

/**
 * @brief Set minimum P diagonal values (prevents Kalman lockup)
 *
 * If a contaminated SMC² window reports tight Σ on a wrong estimate,
 * P can collapse to near-zero, making the Kalman gain zero for future
 * windows. The P floor prevents this by enforcing a minimum uncertainty.
 *
 * Default: P_floor[i] = Q[i] (one window's drift worth of uncertainty).
 *
 * @param p_floor  Array of N_PARAMS minimum P diagonal values
 */
void param_tracker_set_P_floor(ParamTracker *t, const float *p_floor);

/* ═══════════════════════════════════════════════════════════════════════════
 * Internal access + warm-start control
 * ═══════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Access the internal SMC² state (for custom prior/bounds setup)
 *
 * Call BEFORE the first param_tracker_run_window().
 * Modify priors, bounds, proposal stds, etc. directly on the returned state.
 */
SMC2StateCUDA *param_tracker_get_smc2(ParamTracker *t);

/**
 * @brief Get the full Kalman covariance P (8×8, row-major)
 */
void param_tracker_get_P(const ParamTracker *t, float *P_out);

/**
 * @brief Force next window to cold-start from prior (not warm-start)
 *
 * Call this after a phase transition unlocks new parameters. The Kalman
 * has no information about newly-freed dimensions — warm-starting would
 * place all particles at the same (fixed) value for those params, giving
 * no diversity and no exploration.
 *
 * Resets the Kalman so the next window explores from prior, then
 * subsequent windows resume warm-starting once the Kalman has
 * incorporated the new dimensions (after ≥2 updates).
 */
void param_tracker_force_cold(ParamTracker *t);

/* ═══════════════════════════════════════════════════════════════════════════
 * Diagnostics
 * ═══════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Print current tracker state (Kalman estimates + uncertainties)
 */
void param_tracker_print(const ParamTracker *t);

/* ═══════════════════════════════════════════════════════════════════════════
 * Convergence gating
 * ═══════════════════════════════════════════════════════════════════════════*/

/* Gate modes — per-parameter convergence strategy */
#define GATE_KALMAN_MIN  0   /* Push Kalman x after min_windows. No R̂. Fast params. */
#define GATE_RHAT_LATCH  1   /* R̂ gate with one-way latch. Curve params.            */
#define GATE_LOCKED      2   /* Always prior default. Param not free.               */

/* Forward-declare from smc2_convergence_diag.h — include that header
 * for the full struct definition when calling param_tracker_get_conv_report(). */
struct ConvergenceReport;

/**
 * @brief Set which parameters are free (1) vs locked (0)
 *
 * Called by the phased learning controller after phase transitions.
 * Locked params get GATE_LOCKED mode and hold at prior_default.
 * Kalman still updates internally for all params regardless.
 *
 * @param mask  Array of N_PARAMS ints: 1 = free, 0 = locked
 */
void param_tracker_set_free_mask(ParamTracker *t, const int *mask);

/**
 * @brief Set gate mode for a specific parameter
 *
 * @param param_idx  Parameter index (0-7)
 * @param mode       GATE_KALMAN_MIN, GATE_RHAT_LATCH, or GATE_LOCKED
 */
void param_tracker_set_gate_mode(ParamTracker *t, int param_idx, int mode);

/**
 * @brief Set minimum window count for GATE_KALMAN_MIN (default: 2)
 */
void param_tracker_set_min_windows(ParamTracker *t, int n);

/**
 * @brief Override prior default values used as fallback for non-converged params
 */
void param_tracker_set_prior_defaults(ParamTracker *t, const float *defaults);

/**
 * @brief Set R̂ threshold for GATE_RHAT_LATCH convergence (default: 1.5)
 */
void param_tracker_set_rhat_threshold(ParamTracker *t, float thresh);

/**
 * @brief Get the full convergence report (R̂, Mahalanobis, P-trace, CV)
 *
 * Requires #include "smc2_convergence_diag.h" for the ConvergenceReport struct.
 */
void param_tracker_get_conv_report(const ParamTracker *t,
                                    struct ConvergenceReport *rpt);

/**
 * @brief Get per-parameter convergence flags
 *
 * @param out  Array of N_PARAMS ints: 1=converged, 0=not, -1=locked
 */
void param_tracker_get_converged(const ParamTracker *t, int *out);

/**
 * @brief Get raw Kalman state (bypasses convergence gating)
 *
 * Unlike param_tracker_get_snapshot() which returns gated values,
 * this returns the actual Kalman-filtered estimates regardless of
 * convergence status. Use for diagnostics and monitoring.
 */
void param_tracker_get_kalman_x(const ParamTracker *t, float *x_out);

#ifdef __cplusplus
}
#endif

#endif /* SMC2_PARAM_TRACKER_CUH */
