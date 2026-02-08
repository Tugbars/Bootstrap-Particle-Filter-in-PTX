/**
 * @file gpu_bpf_full.cuh
 * @brief GPU Bootstrap Particle Filter — Pure PTX edition
 *
 * Architecture:
 *   - BPF: 14 hand-written PTX kernels, PCG32 PRNG (16 bytes/particle)
 *   - APF: nvcc-compiled kernels, cuRAND (48 bytes/particle)
 *
 * Kernel 14 (bpf_propagate_weight_opt) implements a mode-matched Gaussian
 * optimal proposal via 3 Newton iterations on log p(y|h) + log p(h|h_{t-1}).
 * Drop-in replacement for kernel 3 (bootstrap). Toggle at runtime with
 * gpu_bpf_use_optimal_proposal(1) before calling gpu_bpf_create().
 *
 * Zero cuRAND dependency for BPF path. APF still requires cuRAND.
 *
 * @see bpf_kernels.ptx for the PTX source
 * @see gpu_bpf_ptx_full.cu for the host driver
 */

#ifndef GPU_BPF_FULL_CUH
#define GPU_BPF_FULL_CUH

#include <curand_kernel.h>   // Required for APF (curandState)
#include <stdint.h>

// =============================================================================
// Common types
// =============================================================================

/**
 * @brief Per-tick filter output.
 *
 * Returned by gpu_bpf_step() and gpu_apf_step().
 */
typedef struct {
    float h_mean;   /**< Posterior mean of log-volatility E[h_t | y_{1:t}] */
    float log_lik;  /**< Marginal log-likelihood log p(y_t | y_{1:t-1})   */
} BpfResult;

// =============================================================================
// Adaptive band configuration
// =============================================================================

/** Maximum number of sigma_z scale bands per regime. */
#define MIX_MAX_BANDS 4

/**
 * @brief Band configuration for adaptive sigma_z scaling.
 *
 * Maps directly to d_mix_config constant memory in PTX (36 bytes).
 * Particles in [0, boundary[0]) use k[0], [boundary[0], boundary[1]) use k[1], etc.
 * Contiguous ranges ensure zero warp divergence.
 */
typedef struct {
    int   n_bands;                   /**< Number of active bands (1-4)           */
    int   boundary[MIX_MAX_BANDS];   /**< Cumulative particle index boundaries   */
    float k[MIX_MAX_BANDS];          /**< sigma_z scale factor per band          */
} MixBandConfig;

/** @brief Regime labels for adaptive band switching. */
typedef enum {
    MIX_CALM  = 0,  /**< Low surprise — narrow exploration  */
    MIX_ALERT = 1,  /**< Moderate surprise — wider bands    */
    MIX_PANIC = 2   /**< High surprise — maximum dispersion */
} MixRegime;

// =============================================================================
// Bootstrap Particle Filter — Pure PTX (PCG32 RNG)
// =============================================================================

/**
 * @brief Opaque state for the GPU Bootstrap Particle Filter.
 *
 * All device arrays are managed internally. The filter supports:
 *   - Gaussian or Student-t state noise (via chi2 pre-generation)
 *   - Gaussian or Student-t observation model
 *   - Conditional resampling (ESS threshold)
 *   - Adaptive sigma_z bands (calm/alert/panic regimes)
 *   - Optional Silverman kernel jittering post-resample
 *   - Bootstrap (kernel 3) or optimal (kernel 14) proposal
 */
typedef struct {
    float*    d_h;              /**< Particle states h_t [N]                        */
    float*    d_h2;             /**< Scratch / resampled states [N]                 */
    float*    d_log_w;          /**< Log-weights [N]                                */
    float*    d_w;              /**< Normalized weights [N]                         */
    float*    d_cdf;            /**< CDF for systematic resampling [N]              */
    float*    d_wh;             /**< Weight × state products [N]                    */
    uint64_t* d_rng;            /**< PCG32 state: 2×u64 per particle [2N]           */
    float*    d_scalars;        /**< [5]: max_lw, sum_w, h_est, log_lik, sum_w_sq   */
    float*    d_noise;          /**< Standard normals for Silverman jitter [N]      */
    float*    d_var;            /**< Variance accumulator [1]                       */
    float*    d_log_w_prev;     /**< Saved log-weights for non-resample ticks [N]   */
    float*    d_chi2;           /**< Pre-generated chi2 variates [N] (or NULL)      */
    float*    d_chi2_normals;   /**< Temp normals for chi2 generation [N×nu_int]    */
    void*     curand_gen;       /**< curandGenerator_t for chi2 (void* for compat)  */
    float     C_obs;            /**< Precomputed lgamma constant for Student-t obs  */
    int       nu_int;           /**< Integer nu_state for chi2 square-sum           */
    int       n_particles;      /**< Number of particles N                          */
    int       block, grid;      /**< CUDA launch configuration                     */
    cudaStream_t stream;        /**< Dedicated CUDA stream                         */
    float     rho;              /**< OU persistence parameter                      */
    float     sigma_z;          /**< Vol-of-vol                                    */
    float     mu;               /**< Long-run mean of log-volatility               */
    float     nu_state;         /**< Student-t df for state noise (0 = Gaussian)   */
    float     nu_obs;           /**< Student-t df for observations (0 = Gaussian)  */
    float     silverman_shrink; /**< Jitter bandwidth (0.0 = off, 0.5 = typical)   */
    float     ess_threshold;    /**< Resample when ESS/N < threshold (0 = always)  */
    int       did_resample;     /**< Flag: resampled on last tick?                 */
    int       resample_count;   /**< Cumulative resample count (diagnostic)        */
    float     last_h_est;       /**< Previous tick's h estimate (for surprise)     */
    float     last_surprise;    /**< Previous tick's surprise score                */
    unsigned long long host_rng_state;  /**< Host-side PCG32 for resampling U      */
    int       timestep;         /**< Current tick index                            */
} GpuBpfState;

/**
 * @brief Create and initialize a BPF instance.
 *
 * Seeds PCG32 RNG, draws initial particles from stationary distribution
 * N(mu, sigma_z^2 / (1 - rho^2)). Uses whichever proposal kernel is
 * currently selected (bootstrap or optimal).
 *
 * @param n_particles  Number of particles (recommend power of 2)
 * @param rho          OU persistence (0 < rho < 1)
 * @param sigma_z      Vol-of-vol
 * @param mu           Long-run mean of log-volatility
 * @param nu_state     State noise df (0 = Gaussian, >0 = Student-t)
 * @param nu_obs       Observation df (0 = Gaussian, >0 = Student-t)
 * @param seed         Random seed
 * @return Allocated state (caller must gpu_bpf_destroy)
 */
GpuBpfState* gpu_bpf_create(int n_particles, float rho, float sigma_z, float mu,
                              float nu_state, float nu_obs, int seed);

/** @brief Free all device memory and destroy the CUDA stream. */
void gpu_bpf_destroy(GpuBpfState* state);

/**
 * @brief Process one observation (synchronous).
 *
 * Propagates particles, computes weights, conditionally resamples,
 * and returns the posterior mean and marginal log-likelihood.
 *
 * @param state  BPF instance
 * @param y_t    Observation (return) at time t
 * @return       BpfResult with h_mean and log_lik
 */
BpfResult gpu_bpf_step(GpuBpfState* state, float y_t);

/**
 * @brief Process one observation (asynchronous — no stream sync).
 *
 * Call gpu_bpf_get_result() after cudaStreamSynchronize() to read output.
 * Useful for overlapping multiple filters on separate streams.
 */
void gpu_bpf_step_async(GpuBpfState* state, float y_t);

/**
 * @brief Read results after an async step.
 * @pre cudaStreamSynchronize(state->stream) must have completed.
 */
BpfResult gpu_bpf_get_result(GpuBpfState* state);

// ── Adaptive band API ───────────────────────────────────────────────────────

/**
 * @brief Set static sigma_z scale bands (non-adaptive).
 *
 * Particles are partitioned into contiguous bands. Each band multiplies
 * sigma_z by its scale factor, enabling wider exploration for a subset.
 *
 * @param n_particles  Must match the filter's particle count
 * @param n_bands      Number of bands (1-4)
 * @param fracs        Fraction of particles per band (must sum to 1)
 * @param scales       sigma_z multiplier per band
 */
void gpu_bpf_set_bands(int n_particles, int n_bands,
                        const float* fracs, const float* scales);

/**
 * @brief Configure 3-regime adaptive bands (calm / alert / panic).
 *
 * Regime switching is driven by an EMA of the surprise score
 * |y_t| * exp(-h_est/2). Uploads to PTX constant memory only on
 * regime transitions (zero overhead on calm ticks).
 *
 * @param n_particles    Must match the filter's particle count
 * @param calm_fracs     Band fractions for calm regime
 * @param calm_scales    Band scales for calm regime
 * @param calm_nb        Number of bands in calm regime
 * @param alert_fracs    Band fractions for alert regime
 * @param alert_scales   Band scales for alert regime
 * @param alert_nb       Number of bands in alert regime
 * @param panic_fracs    Band fractions for panic regime
 * @param panic_scales   Band scales for panic regime
 * @param panic_nb       Number of bands in panic regime
 * @param thresh_alert   Surprise EMA threshold for calm → alert
 * @param thresh_panic   Surprise EMA threshold for alert → panic
 */
void gpu_bpf_set_adaptive_bands(int n_particles,
    const float* calm_fracs,  const float* calm_scales,  int calm_nb,
    const float* alert_fracs, const float* alert_scales, int alert_nb,
    const float* panic_fracs, const float* panic_scales, int panic_nb,
    float thresh_alert, float thresh_panic);

// ── Conditional resampling ──────────────────────────────────────────────────

/**
 * @brief Set ESS threshold for conditional resampling.
 *
 * Resampling only fires when ESS/N drops below threshold.
 * Set to 0.0 to always resample (default), 0.5 for ESS < N/2.
 */
void gpu_bpf_set_ess_threshold(GpuBpfState* state, float threshold);

/** @brief Return cumulative number of ticks where resampling fired. */
int gpu_bpf_get_resample_count(GpuBpfState* state);

// ── Optimal proposal toggle ─────────────────────────────────────────────────

/**
 * @brief Select proposal kernel for all subsequent BPF instances.
 *
 * Must be called BEFORE gpu_bpf_create(). Affects all BPF instances
 * created after this call (global setting).
 *
 *   - enable=0: kernel 3 (bootstrap) — blind OU proposal, obs-only weight
 *   - enable=1: kernel 14 (optimal)  — Newton mode-matched Gaussian proposal,
 *               importance-corrected weight
 *
 * If kernel 14 is not in the PTX file, silently falls back to bootstrap
 * and prints a warning to stderr.
 *
 * @param enable  0 = bootstrap, 1 = optimal
 */
void gpu_bpf_use_optimal_proposal(int enable);

/** @brief Query current proposal selection. @return 0 or 1. */
int gpu_bpf_get_optimal_proposal(void);

// ── Batch RMSE convenience ──────────────────────────────────────────────────

/**
 * @brief Run BPF over a full return series and compute RMSE vs ground truth.
 *
 * Creates a temporary filter, processes all ticks, destroys it.
 * Skips first 100 ticks for warmup.
 *
 * @return RMSE of h_mean vs true_h (post-warmup)
 */
double gpu_bpf_run_rmse(
    const double* returns, const double* true_h, int n_ticks,
    int n_particles, float rho, float sigma_z, float mu,
    float nu_state, float nu_obs, int seed
);

// =============================================================================
// Auxiliary Particle Filter — nvcc compiled, cuRAND
// =============================================================================

/**
 * @brief Opaque state for the GPU Auxiliary Particle Filter.
 *
 * Two-stage resampling: first-stage weights from predicted observation
 * likelihood, second-stage correction after propagation.
 * Uses curandState (48 bytes/particle) — not PTX.
 */
typedef struct {
    float*       d_h;           /**< Particle states [N]                   */
    float*       d_h2;          /**< Scratch states [N]                    */
    float*       d_mu_pred;     /**< Predicted means μ + ρ(h-μ) [N]       */
    float*       d_log_v;       /**< First-stage log-weights [N]           */
    float*       d_v;           /**< First-stage weights [N]               */
    float*       d_log_w;       /**< Second-stage log-weights [N]          */
    float*       d_w;           /**< Normalized weights [N]                */
    float*       d_cdf;         /**< CDF for resampling [N]                */
    float*       d_wh;          /**< Weight × state [N]                    */
    float*       d_mu_pred2;    /**< Resampled predicted means [N]         */
    curandState* d_rng;         /**< cuRAND states [N]                     */
    float*       d_scalars;     /**< [4]: max_lw, sum_w, h_est, log_lik    */
    int          n_particles;   /**< Number of particles                   */
    int          block, grid;   /**< CUDA launch config                    */
    cudaStream_t stream;        /**< Dedicated CUDA stream                 */
    float        rho;           /**< OU persistence                        */
    float        sigma_z;       /**< Vol-of-vol                            */
    float        mu;            /**< Long-run mean                         */
    float        nu_state;      /**< State noise df (0 = Gaussian)         */
    float        nu_obs;        /**< Observation df (0 = Gaussian)         */
    unsigned long long host_rng_state;  /**< Host PCG32 for resampling U   */
    int          timestep;      /**< Current tick index                    */
} GpuApfState;

/** @brief Create and initialize an APF instance. @see gpu_bpf_create for params. */
GpuApfState* gpu_apf_create(int n_particles, float rho, float sigma_z, float mu,
                              float nu_state, float nu_obs, int seed);

/** @brief Free all device memory and destroy the CUDA stream. */
void gpu_apf_destroy(GpuApfState* state);

/**
 * @brief Process one observation with APF (synchronous).
 * @param state  APF instance
 * @param y_t    Observation at time t
 * @return       BpfResult with h_mean and log_lik
 */
BpfResult gpu_apf_step(GpuApfState* state, float y_t);

/** @brief Run APF over a full series and compute RMSE. @see gpu_bpf_run_rmse. */
double gpu_apf_run_rmse(
    const double* returns, const double* true_h, int n_ticks,
    int n_particles, float rho, float sigma_z, float mu,
    float nu_state, float nu_obs, int seed
);

#endif // GPU_BPF_FULL_CUH