/**
 * @file smc2_param_tracker.cuh
 * @brief Kalman Parameter Tracker + Phased Learning Controller — Header
 *
 * Sliding-window SMC² produces θ̂, Σ measurements per window.
 * Linear Kalman filter fuses them over time → Kalman state x[8].
 * Curves evaluated at z̄ produce BPF-ready scalar parameters.
 *
 * Integrated phased learning controls which params SMC² is allowed
 * to learn, based on observed z-range within each window:
 *
 *   Phase 1 (calm):   4 params: ρ, σ_total, r_split, μ_base
 *   Phase 2 (stress): + ceilings: μ_scale, σ_scale     (6 params)
 *   Phase 3 (full):   + rates: μ_rate, σ_rate           (8 params)
 *
 * Phases are BIDIRECTIONAL — when z-range narrows, learned values are
 * saved and params are re-locked to preserve information and ESS.
 *
 * The 8×8 Kalman math is host-side (too small for GPU), but this
 * file is .cuh because it calls smc2_cuda_* functions.
 */

#ifndef SMC2_PARAM_TRACKER_CUH
#define SMC2_PARAM_TRACKER_CUH

#include "smc2_engine.cuh"

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * Parameter index constants (must match ThetaParticlesSoA order)
 *═══════════════════════════════════════════════════════════════════════════════*/

#define P_RHO           0
#define P_SIGMA_TOTAL   1
#define P_R_SPLIT       2
#define P_MU_BASE       3
#define P_MU_SCALE      4
#define P_MU_RATE       5
#define P_SIGMA_SCALE   6
#define P_SIGMA_RATE    7

/*═══════════════════════════════════════════════════════════════════════════════
 * Learning phases
 *═══════════════════════════════════════════════════════════════════════════════*/

#define PHASE_1_FLOORS    1   /* ρ, σ_total, r_split, μ_base (floors only)     */
#define PHASE_2_CEILINGS  2   /* + μ_scale, σ_scale (rates fixed)              */
#define PHASE_3_RATES     3   /* + μ_rate, σ_rate (all 8 free)                 */

/*═══════════════════════════════════════════════════════════════════════════════
 * Gate modes — per-parameter convergence strategy
 *═══════════════════════════════════════════════════════════════════════════════*/

#define GATE_KALMAN_MIN  0   /* Push Kalman x after min_windows. No R̂.        */
#define GATE_RHAT_LATCH  1   /* R̂ gate with one-way latch. Curve params.      */
#define GATE_LOCKED      2   /* Always prior default / saved value.            */

/*═══════════════════════════════════════════════════════════════════════════════
 * Configuration structures
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Output snapshot — BPF-ready parameters after Kalman + gating + curves
 */
typedef struct {
    float theta[N_PARAMS];    /**< Gated θ (converged → Kalman, else → default) */
    float P_diag[N_PARAMS];   /**< Kalman P diagonal (raw, for diagnostics)     */
    float sigma_z;            /**< r_split * sigma_total                         */
    float sigma_base;         /**< sqrt(1-r²) * sigma_total                     */
    float z_mean;             /**< z̄ from last SMC² window                     */
    float mu;                 /**< eval_curve(μ_base, μ_scale, μ_rate, z̄)      */
    float sigma_h;            /**< eval_curve(σ_base, σ_scale, σ_rate, z̄)      */
    float theta_speed;        /**< eval_curve(θ_base, θ_scale, θ_rate, z̄)      */
    int   n_updates;          /**< Total windows processed                       */
    float last_accept_rate;   /**< CPMMH accept rate in last window              */
    float last_ess;           /**< Outer ESS after last window                   */
} ParamSnapshot;

/**
 * @brief Kalman process noise (Q diagonal) — drift rates per parameter
 */
typedef struct {
    float q_rho;
    float q_sigma_total;
    float q_r_split;
    float q_mu_base;
    float q_mu_scale;
    float q_mu_rate;
    float q_sigma_scale;
    float q_sigma_rate;
} DriftConfig;

/**
 * @brief Phased learning configuration — controls phase transition thresholds
 */
typedef struct {
    /* Phase 1 ↔ 2 triggers */
    float ceiling_z_threshold;      /**< z_max threshold for unlock/lock (def: 2.0) */
    int   ceiling_z_sustained;      /**< Consecutive windows needed (def: 3)        */

    /* Phase 2 ↔ 3 triggers */
    float rate_z_range_threshold;   /**< z_range threshold (def: 1.5)               */
    int   rate_range_sustained;     /**< Consecutive windows needed (def: 3)        */

    /* Fixed values for locked params (start as prior, update with learned) */
    float fixed_mu_scale;
    float fixed_mu_rate;
    float fixed_sigma_scale;
    float fixed_sigma_rate;

    /* Flags — have these been learned at least once? */
    int   learned_ceilings;
    int   learned_rates;

    /* EMA smoothing for z */
    float z_ema_alpha;              /**< EMA decay (default: 0.1)                   */

    /* Enable/disable phased learning (1=enabled, 0=always Phase 3) */
    int   enabled;
} PhasedConfig;

/**
 * @brief Phase transition history entry
 */
typedef struct {
    int   window;
    int   from;
    int   to;
    float z_min;
    float z_max;
    float z_range;
} PhaseTransition;

#define MAX_PHASE_TRANSITIONS 64

/*═══════════════════════════════════════════════════════════════════════════════
 * Default configurations
 *═══════════════════════════════════════════════════════════════════════════════*/

static inline PhasedConfig phased_default_config(void) {
    PhasedConfig c;
    c.ceiling_z_threshold    = 2.0f;
    c.ceiling_z_sustained    = 3;
    c.rate_z_range_threshold = 1.5f;
    c.rate_range_sustained   = 3;
    c.fixed_mu_scale         = 2.0f;
    c.fixed_mu_rate          = 0.3f;
    c.fixed_sigma_scale      = 0.5f;
    c.fixed_sigma_rate       = 0.3f;
    c.learned_ceilings       = 0;
    c.learned_rates          = 0;
    c.z_ema_alpha            = 0.1f;
    c.enabled                = 1;
    return c;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * Opaque type
 *═══════════════════════════════════════════════════════════════════════════════*/

typedef struct ParamTracker ParamTracker;

/*═══════════════════════════════════════════════════════════════════════════════
 * Lifecycle
 *═══════════════════════════════════════════════════════════════════════════════*/

ParamTracker* param_tracker_create(int window_size, int stride,
                                    int N_theta, int N_inner);
void          param_tracker_destroy(ParamTracker* t);

/*═══════════════════════════════════════════════════════════════════════════════
 * Per-tick feeding + window execution
 *═══════════════════════════════════════════════════════════════════════════════*/

void  param_tracker_feed(ParamTracker* t, float y_obs);
int   param_tracker_window_ready(const ParamTracker* t);
void  param_tracker_run_window(ParamTracker* t);

/*═══════════════════════════════════════════════════════════════════════════════
 * Output
 *═══════════════════════════════════════════════════════════════════════════════*/

void  param_tracker_get_snapshot(const ParamTracker* t, ParamSnapshot* snap);
void  param_tracker_print(const ParamTracker* t);

/*═══════════════════════════════════════════════════════════════════════════════
 * Configuration
 *═══════════════════════════════════════════════════════════════════════════════*/

void  param_tracker_set_drift(ParamTracker* t, const DriftConfig* drift);
void  param_tracker_set_theta_curve(ParamTracker* t, float base, float scale, float rate);
void  param_tracker_set_P_floor(ParamTracker* t, const float* p_floor);
void  param_tracker_set_prior_defaults(ParamTracker* t, const float* defaults);

/* ── Convergence gating ── */
void  param_tracker_set_free_mask(ParamTracker* t, const int* mask);
void  param_tracker_set_gate_mode(ParamTracker* t, int param_idx, int mode);
void  param_tracker_set_min_windows(ParamTracker* t, int n);
void  param_tracker_set_rhat_threshold(ParamTracker* t, float thresh);

/* ── Phased learning ── */
void  param_tracker_set_phased_config(ParamTracker* t, const PhasedConfig* pc);
void  param_tracker_set_phased_fixed_rates(ParamTracker* t, float mu_rate, float sigma_rate);
void  param_tracker_set_phased_fixed_ceilings(ParamTracker* t, float mu_scale, float sigma_scale);

/*═══════════════════════════════════════════════════════════════════════════════
 * Queries
 *═══════════════════════════════════════════════════════════════════════════════*/

SMC2StateCUDA* param_tracker_get_smc2(ParamTracker* t);
void  param_tracker_get_P(const ParamTracker* t, float* P_out);
void  param_tracker_get_kalman_x(const ParamTracker* t, float* x_out);
void  param_tracker_get_converged(const ParamTracker* t, int* out);
void  param_tracker_force_cold(ParamTracker* t);

/* ── Convergence report ── */
/* ConvergenceReport is defined in smc2_pipeline.h (§1) */
struct ConvergenceReport;
void  param_tracker_get_conv_report(const ParamTracker* t, ConvergenceReport* rpt);

/* ── Phased learning queries ── */
int   param_tracker_get_phase(const ParamTracker* t);
int   param_tracker_get_n_transitions(const ParamTracker* t);
void  param_tracker_get_z_range(const ParamTracker* t,
                                 float* z_mean, float* z_min, float* z_max);
void  param_tracker_print_phased_status(const ParamTracker* t);

#ifdef __cplusplus
}
#endif

#endif /* SMC2_PARAM_TRACKER_CUH */
