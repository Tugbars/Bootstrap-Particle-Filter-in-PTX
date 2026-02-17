/*═══════════════════════════════════════════════════════════════════════════════
 * @file smc2_bpf_pipeline.h
 * @brief Wiring between dBPF (per-tick vol estimator) and SMC² (parameter learner)
 *
 * Architecture:
 *   - dBPF runs every tick on stream_bpf (~microseconds per tick)
 *   - SMC² runs sliding windows on stream_smc2 (~2 seconds per window)
 *   - SMC² pushes posterior parameter estimates to dBPF at window boundaries
 *   - dBPF's Kalman tracker resets P→P0 on param push, re-adapts from new baseline
 *
 * Two modes:
 *   SYNC  — dBPF blocks at window boundary until SMC² finishes. Deterministic.
 *   ASYNC — dBPF never blocks. SMC² pushes when ready. Production mode.
 *
 * No threads. No mutexes. Two CUDA streams handle all concurrency.
 *
 * Build: Include alongside gpu_bpf_full.cuh and smc2_rbpf_batch.cuh
 *═══════════════════════════════════════════════════════════════════════════════*/

#ifndef SMC2_BPF_PIPELINE_H
#define SMC2_BPF_PIPELINE_H

#include "gpu_bpf_full.cuh"
#include "smc2_rbpf_batch.cuh"
#include <cuda_runtime.h>

/* ── Pipeline configuration ─────────────────────────────────────────────── */

typedef struct {
    int   window_size;       /**< SMC² window length in ticks (e.g. 3000)      */
    int   stride;            /**< Ticks between windows (e.g. 1500)            */
    int   sync_mode;         /**< 0=async (production), 1=sync (testing)       */
    int   push_rho;          /**< Push ρ from SMC² to dBPF? (0/1)             */
    int   push_sigma_z;      /**< Push σ_z from SMC² to dBPF? (0/1)           */
    float obs_buffer_sec;    /**< Observation ring buffer size in seconds      */
} PipelineConfig;

static inline PipelineConfig pipeline_default_config(void) {
    PipelineConfig c;
    c.window_size    = 3000;
    c.stride         = 1500;
    c.sync_mode      = 1;       /* Start in sync mode for testing */
    c.push_rho       = 1;
    c.push_sigma_z   = 1;
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

    return p;
}

static inline void pipeline_destroy(SMC2BpfPipeline* p) {
    if (!p) return;
    cudaStreamDestroy(p->stream_bpf);
    cudaStreamDestroy(p->stream_smc2);
    free(p->obs_buffer);
    free(p);
}

/* ── Internal: extract SMC² posterior and push to dBPF ──────────────────── */

static inline void pipeline_push_params(SMC2BpfPipeline* p) {
    /* Synchronize SMC² stream — results must be ready */
    cudaStreamSynchronize(p->stream_smc2);

    /* Extract posterior means from SMC² */
    /* TODO: Replace with actual smc2_get_posterior_mean() call.
     *       For now, this shows the interface contract. */
    float post_mu, post_rho, post_sigma_total, post_r_split;
    smc2_cuda_get_posterior_means(p->smc2,
        &post_rho, &post_sigma_total, &post_r_split, &post_mu);

    /* Derive σ_z from σ_total and r_split */
    float post_sigma_z = post_r_split * post_sigma_total;

    /* Push μ (always) */
    gpu_bpf_set_mu(p->bpf, post_mu);

    /* Push ρ (optional) */
    if (p->config.push_rho) {
        gpu_bpf_set_rho(p->bpf, post_rho);
    }

    /* Push σ_z (optional — can't be learned by gradient, SMC² is only source) */
    if (p->config.push_sigma_z) {
        gpu_bpf_set_sigma_z(p->bpf, post_sigma_z);
    }

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
    /* (In production, SMC² could read directly from ring buffer) */
    float* window_obs = (float*)malloc(W * sizeof(float));
    for (int i = 0; i < W; i++) {
        int buf_idx = (start + i) % p->obs_capacity;
        window_obs[i] = p->obs_buffer[buf_idx];
    }

    /* Run SMC² on stream_smc2 */
    /* TODO: Replace with actual smc2_cuda_run_window() call */
    smc2_cuda_process_window(p->smc2, window_obs, W, p->stream_smc2);

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
}

#endif /* SMC2_BPF_PIPELINE_H */
