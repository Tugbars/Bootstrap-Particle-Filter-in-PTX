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
 * Resets the internal SMC² instance, feeds the window of observations,
 * extracts θ̂ and Σ, runs Kalman predict+update, evaluates curves at z̄.
 *
 * This is the expensive call — runs SMC² on `window_size` observations.
 * Call only when param_tracker_window_ready() returns 1.
 */
void param_tracker_run_window(ParamTracker *t);

/**
 * @brief Get the current filtered parameter snapshot
 * @param snap  Output snapshot with filtered params + BPF-ready values
 */
void param_tracker_get_snapshot(const ParamTracker *t, ParamSnapshot *snap);

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

#ifdef __cplusplus
}
#endif

#endif /* SMC2_PARAM_TRACKER_CUH */
