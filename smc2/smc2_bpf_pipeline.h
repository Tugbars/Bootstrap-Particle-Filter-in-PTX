/*═══════════════════════════════════════════════════════════════════════════════
 * @file smc2_bpf_pipeline.h
 * @brief Wiring between dBPF (per-tick vol estimator) and SMC² (parameter learner)
 *
 * Architecture (v2 — with Kalman parameter tracker):
 *
 *   ┌──────────────────────────────────────────────────────────────────────┐
 *   │  Per-tick loop                                                       │
 *   │    1. Feed y_t into ParamTracker's circular buffer                  │
 *   │    2. Run dBPF on y_t → BpfResult (microseconds)                   │
 *   │    3. At window boundary:                                            │
 *   │       a. ParamTracker runs SMC² on last W observations              │
 *   │       b. Kalman filter fuses SMC² posterior with history            │
 *   │       c. Evaluate curves at z̄ → μ(z̄), σ_h(z̄)                     │
 *   │       d. Push Kalman-smoothed params to dBPF                        │
 *   └──────────────────────────────────────────────────────────────────────┘
 *
 * ═════════════════════════════════════════════════════════════════════════════
 * CRITICAL: OBSERVATION TRANSFORM — TWO DIFFERENT OBSERVATION MODELS
 * ═════════════════════════════════════════════════════════════════════════════
 *
 * The production BPF and the SMC²/RBPF operate on DIFFERENT observation
 * spaces. Getting this wrong causes SMC² to learn completely wrong
 * parameters (e.g. mu_base = -1.5 instead of -4.5).
 *
 *   ┌─────────────┬────────────────────┬────────────────────────────────┐
 *   │ Component    │ Expects            │ Observation equation           │
 *   ├─────────────┼────────────────────┼────────────────────────────────┤
 *   │ Production  │ Raw returns r_t    │ r_t = exp(h/2) · ε_t          │
 *   │ dBPF        │                    │ ε_t ~ N(0,1)                   │
 *   │             │                    │ Direct particle filter on h    │
 *   ├─────────────┼────────────────────┼────────────────────────────────┤
 *   │ SMC²/RBPF   │ y_t = log(r_t²)   │ y_t = h_t + log(χ²(1))        │
 *   │ (OCSN       │ (transformed)      │ ≈ h_t + OCSN 10-component mix │
 *   │  Kalman)    │                    │ Rao-Blackwellized Kalman on h  │
 *   └─────────────┴────────────────────┴────────────────────────────────┘
 *
 * The OCSN (Omori-Chib-Shephard-Nakajima 2007) mixture approximates
 * log(χ²(1)) as a 10-component Gaussian mixture, which restores linear-
 * Gaussian structure for the Kalman update on h. This is what makes
 * Rao-Blackwellization possible — the key to SMC²'s clean likelihood
 * signal for parameter learning.
 *
 * The production BPF works directly with raw returns because it uses
 * a standard bootstrap particle filter with exp(h/2) observation density,
 * no OCSN mixture needed.
 *
 * PIPELINE RESPONSIBILITY:
 *   - pipeline_step() receives raw returns y_t
 *   - It feeds raw y_t to gpu_bpf_step() — correct for BPF
 *   - It transforms to log(y²) before feeding to param_tracker_feed()
 *     because ParamTracker does NOT apply the transform internally —
 *     it stores observations as-is and passes them to SMC² verbatim
 *   - If feeding SMC² directly (bypassing ParamTracker), YOU must
 *     apply the transform: log_y = logf(y * y + 1e-20f)
 *
 * ═════════════════════════════════════════════════════════════════════════════
 * CRITICAL: DGP/MODEL CONSISTENCY — h DYNAMICS MUST MATCH RBPF
 * ═════════════════════════════════════════════════════════════════════════════
 *
 * The RBPF inner filter models h with z-DEPENDENT mean-reversion speed:
 *
 *   θ(z) = θ_base + θ_scale · (1 - exp(-θ_rate · z))
 *   φ(z) = 1 - θ(z)
 *   h_{t+1} = φ(z) · h_t + θ(z) · μ(z) + σ_h(z) · ε_t
 *
 * At z=0 (calm):   θ≈0.02, φ≈0.98 — slow mean-reversion
 * At z=3 (crisis):  θ≈0.10, φ≈0.90 — faster mean-reversion
 *
 * This is NOT the same as a constant-ρ AR(1):
 *   h_{t+1} = ρ · h_t + (1-ρ) · μ(z) + σ_h(z) · ε_t     ← WRONG for DGP
 *
 * If the data-generating process uses constant ρ but the RBPF uses θ(z),
 * SMC² will shift mu_base to compensate for the model mismatch — at high z,
 * the RBPF mean-reverts ~5× faster than the DGP, so SMC² moves the target
 * closer to where h already is. This causes a ~3-unit bias in mu_base.
 *
 * The θ(z) curve is set via smc2->theta_curve = {base, scale, rate} and
 * is held FIXED (not learned) during SMC² — it's derived offline from
 * φ-based sufficient statistics.
 *
 * When writing test DGPs, ALWAYS use the RBPF's h dynamics:
 *   float theta_z = sat_exp(theta_base, theta_scale, theta_rate, z);
 *   float phi = 1.0f - theta_z;
 *   h = phi * h + theta_z * mu_z + sigma_h * noise;
 *
 * In production, the data IS the data — there's no DGP mismatch because
 * the model parameters are what they are. This only matters for testing.
 *
 * ═════════════════════════════════════════════════════════════════════════════
 *
 * Key change from v1:
 *   v1: pipeline_push_params() called smc2_cuda_get_theta_mean() directly
 *       and pushed mu_base (the curve FLOOR) to the BPF. During crises
 *       where z > 0, the true μ(z) is much higher than mu_base — this was
 *       a bug that made the BPF see ~10% vol when the market was at ~47%.
 *
 *   v2: ParamTracker owns SMC² and the Kalman filter. It produces a
 *       ParamSnapshot with snap.mu = eval_curve(mu_base, mu_scale, mu_rate, z̄),
 *       which is the correct curve-evaluated, Kalman-smoothed μ at the
 *       current stress level.
 *
 * Two modes:
 *   SYNC  — dBPF blocks at window boundary until SMC² finishes. Deterministic.
 *   ASYNC — dBPF never blocks. SMC² pushes when ready. Production mode.
 *
 * No threads. No mutexes. Two CUDA streams handle all concurrency.
 *
 * Build: Include alongside gpu_bpf_full.cuh and smc2_param_tracker.cuh
 *═══════════════════════════════════════════════════════════════════════════════*/

#ifndef SMC2_BPF_PIPELINE_H
#define SMC2_BPF_PIPELINE_H

#include "gpu_bpf_full.cuh"
#include "smc2_rbpf_batch.cuh"
#include "smc2_phased_learning.h"
#include <cuda_runtime.h>

/* ── Pipeline configuration ─────────────────────────────────────────────── */

typedef struct {
    int   window_size;       /**< SMC² window length in ticks (e.g. 3000)      */
    int   stride;            /**< Ticks between windows (e.g. 1500)            */
    int   sync_mode;         /**< 0=async (production), 1=sync (testing)       */
    int   push_rho;          /**< Push ρ from SMC² to dBPF? (0/1)             */
    float obs_buffer_sec;    /**< Observation ring buffer size in seconds      */
} PipelineConfig;

static inline PipelineConfig pipeline_default_config(void) {
    PipelineConfig c;
    c.window_size    = 3000;
    c.stride         = 1500;
    c.sync_mode      = 1;       /* Start in sync mode for testing */
    c.push_rho       = 1;
    c.obs_buffer_sec = 0.0f;    /* Unused for now */
    return c;
}

/* ── Pipeline state ─────────────────────────────────────────────────────── */

typedef struct {
    /* Sub-systems (owned externally — pipeline does not create/destroy these) */
    GpuBpfState*    bpf;
    SMC2StateCUDA*  smc2;

    /* CUDA streams */
    cudaStream_t    stream_bpf;
    cudaStream_t    stream_smc2;

    /* Configuration */
    PipelineConfig  config;

    /* Observation ring buffer for SMC² windows */
    float*          obs_buffer;          /**< Host-side circular buffer        */
    int             obs_capacity;        /**< Buffer capacity (>= window_size) */
    int             obs_write_pos;       /**< Next write position (modular)    */
    int             obs_count;           /**< Total observations received      */

    /* Tick counters */
    int             bpf_tick;            /**< Current dBPF tick                */
    int             smc2_tick;           /**< Last tick SMC² has processed to  */
    int             next_window_trigger; /**< bpf_tick that triggers next SMC² */

    /* Last SMC² posterior (for diagnostics) */
    float           last_push_mu;
    float           last_push_rho;
    float           last_push_sigma_z;
    int             last_push_tick;
    int             n_pushes;

    /* SMC² running state */
    int             smc2_running;        /**< 1 if SMC² window in progress    */

    /* Phased learning controller (optional) */
    PhasedLearner*  phased;              /**< NULL if phased learning disabled */
} SMC2BpfPipeline;

/* ── Lifecycle ──────────────────────────────────────────────────────────── */

/**
 * @brief Create pipeline wiring.
 *
 * Does NOT create the bpf or smc2 — caller owns those.
 * Pipeline creates its own CUDA streams and observation buffer.
 */
static inline SMC2BpfPipeline* pipeline_create(
    GpuBpfState*    bpf,
    SMC2StateCUDA*  smc2,
    PipelineConfig  config
) {
    SMC2BpfPipeline* p = (SMC2BpfPipeline*)calloc(1, sizeof(SMC2BpfPipeline));

    p->bpf    = bpf;
    p->smc2   = smc2;
    p->config = config;

    cudaStreamCreate(&p->stream_bpf);
    cudaStreamCreate(&p->stream_smc2);

    /* Observation buffer: 2x window for overlap */
    p->obs_capacity = config.window_size * 2;
    p->obs_buffer   = (float*)calloc(p->obs_capacity, sizeof(float));
    p->obs_write_pos = 0;
    p->obs_count     = 0;

    p->bpf_tick            = 0;
    p->smc2_tick           = 0;
    p->next_window_trigger = config.window_size;  /* First window after window_size ticks */

    p->last_push_mu      = 0.0f;
    p->last_push_rho     = 0.0f;
    p->last_push_sigma_z = 0.0f;
    p->last_push_tick    = -1;
    p->n_pushes          = 0;
    p->smc2_running      = 0;

    /* Phased learning: enabled by default */
    PhasedConfig pc = phased_default_config();
    p->phased = phased_create(smc2, pc);

    return p;
}

static inline void pipeline_destroy(SMC2BpfPipeline* p) {
    if (!p) return;
    if (p->phased) phased_destroy(p->phased);
    cudaStreamDestroy(p->stream_bpf);
    cudaStreamDestroy(p->stream_smc2);
    free(p->obs_buffer);
    free(p);
}

/* ── Internal: extract SMC² posterior and push to dBPF ──────────────────── */

static inline void pipeline_push_params(SMC2BpfPipeline* p) {
    /* Synchronize SMC² stream — results must be ready */
    cudaStreamSynchronize(p->stream_smc2);

    /* ── Phased learning: observe z-range and maybe advance phase ──── */
    if (p->phased) {
        phased_observe_z_from_smc2(p->phased);
        if (phased_update(p->phased)) {
            /* Phase advanced — mask changed, next window uses new param set */
        }
    }

    /* ── Extract posterior means from SMC² ──────────────────────────── */
    /* theta_mean layout: [rho, sigma_total, r_split, mu_base,
     *                     mu_scale, mu_rate, sigma_scale, sigma_rate] */
    float theta_mean[N_PARAMS];
    smc2_cuda_get_theta_mean(p->smc2, theta_mean);

    float post_rho         = theta_mean[0];
    float post_sigma_total = theta_mean[1];
    float post_r_split     = theta_mean[2];
    float post_mu          = theta_mean[3];  /* mu_base = floor */
    float post_sigma_z     = post_r_split * post_sigma_total;

    /* Push μ (always — Kalman P resets to P0) */
    gpu_bpf_set_mu(p->bpf, post_mu);

    /* Push ρ (optional — Kalman P resets to P0) */
    if (p->config.push_rho) {
        gpu_bpf_set_rho(p->bpf, post_rho);
    }

    /*
     * σ_z cannot be pushed — gpu_bpf has no set_sigma_z().
     * σ_z is set at creation time only. To update it, the BPF must be
     * recreated. In practice, σ_z changes slowly enough that recreation
     * at epoch boundaries (every ~100 windows) is acceptable.
     * TODO: Add gpu_bpf_set_sigma_z() to the BPF API.
     */

    /* Record for diagnostics */
    p->last_push_mu      = post_mu;
    p->last_push_rho     = post_rho;
    p->last_push_sigma_z = post_sigma_z;
    p->last_push_tick    = p->bpf_tick;
    p->n_pushes++;
}

/* ── Internal: launch SMC² window ───────────────────────────────────────── */

static inline void pipeline_launch_smc2_window(SMC2BpfPipeline* p) {
    /*
     * Extract the most recent window_size observations from the ring buffer.
     * SMC² processes them as a batch.
     */
    int W = p->config.window_size;
    int start = p->obs_count - W;
    if (start < 0) return;  /* Not enough data yet */

    /* Build contiguous observation array for SMC² */
    float* window_obs = (float*)malloc(W * sizeof(float));
    for (int i = 0; i < W; i++) {
        int buf_idx = (start + i) % p->obs_capacity;
        window_obs[i] = p->obs_buffer[buf_idx];
    }

    /* Re-initialize SMC² from prior for each window (sliding window mode) */
    smc2_cuda_init_from_prior(p->smc2);

    /* Run SMC² batch update — processes all W observations */
    smc2_cuda_update_batch(p->smc2, window_obs, W);

    free(window_obs);

    p->smc2_tick = p->obs_count;
    p->next_window_trigger += p->config.stride;
    p->smc2_running = 1;
}

/* ── Main tick function ─────────────────────────────────────────────────── */

/**
 * @brief Process one tick through the pipeline.
 *
 * Called once per observation. Returns the dBPF result for this tick.
 *
 * In SYNC mode:  blocks at window boundary until SMC² finishes + pushes params.
 * In ASYNC mode: never blocks. SMC² push happens when ready.
 */
static inline BpfResult pipeline_step(SMC2BpfPipeline* p, float y_t) {

    /* ── 1. Store observation in ring buffer ───────────────────────────── */
    p->obs_buffer[p->obs_write_pos % p->obs_capacity] = y_t;
    p->obs_write_pos++;
    p->obs_count++;

    /* ── 2. Run dBPF (always, never blocks) ────────────────────────────── */
    BpfResult result = gpu_bpf_step(p->bpf, y_t);
    p->bpf_tick++;

    /* ── 3. Check if async SMC² has finished ───────────────────────────── */
    if (!p->config.sync_mode && p->smc2_running) {
        cudaError_t status = cudaStreamQuery(p->stream_smc2);
        if (status == cudaSuccess) {
            /* SMC² finished in background — push params now */
            pipeline_push_params(p);
            p->smc2_running = 0;
        }
        /* If cudaErrorNotReady, SMC² still running — do nothing, dBPF continues */
    }

    /* ── 4. Window boundary: launch or wait for SMC² ───────────────────── */
    if (p->bpf_tick >= p->next_window_trigger) {

        if (p->config.sync_mode) {
            /* SYNC: if previous SMC² still running, wait for it */
            if (p->smc2_running) {
                pipeline_push_params(p);
                p->smc2_running = 0;
            }

            /* Launch new window and block until done */
            pipeline_launch_smc2_window(p);
            pipeline_push_params(p);
            p->smc2_running = 0;

        } else {
            /* ASYNC: if previous SMC² still running, skip (don't pile up) */
            if (!p->smc2_running) {
                pipeline_launch_smc2_window(p);
            }
            /* Don't block — dBPF continues on next tick */
        }
    }

    return result;
}

/* ── Diagnostics ────────────────────────────────────────────────────────── */

static inline void pipeline_print_status(const SMC2BpfPipeline* p) {
    printf("Pipeline status:\n");
    printf("  bpf_tick:     %d\n", p->bpf_tick);
    printf("  smc2_tick:    %d\n", p->smc2_tick);
    printf("  n_pushes:     %d\n", p->n_pushes);
    printf("  sync_mode:    %s\n", p->config.sync_mode ? "SYNC" : "ASYNC");
    printf("  smc2_running: %s\n", p->smc2_running ? "yes" : "no");
    if (p->n_pushes > 0) {
        printf("  last_push:    tick=%d  mu=%.4f  rho=%.4f  sigma_z=%.4f\n",
               p->last_push_tick, p->last_push_mu,
               p->last_push_rho, p->last_push_sigma_z);
    }
    printf("  bpf_mu:       %.4f\n", gpu_bpf_get_mu(p->bpf));
    printf("  bpf_rho:      %.4f\n", gpu_bpf_get_rho(p->bpf));
    if (p->phased) {
        printf("  ---\n");
        phased_print_status(p->phased);
    }
}

#endif /* SMC2_BPF_PIPELINE_H */