/**
 * @file smc2_pipeline.h
 * @brief SMC² Integration Layer — Convergence Diagnostics + BPF Pipeline Wiring
 *
 * Everything that sits ABOVE smc2_engine and connects it to the production
 * BPF lives here. Two logical sections:
 *
 *   §1  Convergence Diagnostics    Rolling-window R̂, Mahalanobis d², P-trace
 *                                  (used by param_tracker + pipeline)
 *
 *   §2  BPF Pipeline Wiring        Per-tick orchestration between dBPF
 *                                  (vol estimator) and SMC² (param learner)
 *
 * Header-only, pure C (no .cu needed). Depends on:
 *   - smc2_engine.cuh       (types: SMC2StateCUDA, N_PARAMS, smc2_cuda_* API)
 *   - gpu_bpf_full.cuh      (types: GpuBpfState, BpfResult — only for §2)
 *   - smc2_param_tracker.cuh (phased learning is internal to param_tracker)
 *
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
 * PIPELINE RESPONSIBILITY:
 *   - pipeline_step() receives raw returns y_t
 *   - It feeds raw y_t to gpu_bpf_step() — correct for BPF
 *   - It transforms to log(y²) before feeding to param_tracker_feed()
 *     because ParamTracker does NOT apply the transform internally
 *   - If feeding SMC² directly: log_y = logf(y * y + 1e-20f)
 *
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
 * The θ(z) curve is set via smc2->theta_curve and held FIXED (not learned).
 * When writing test DGPs, ALWAYS use the RBPF's h dynamics — not constant-ρ
 * AR(1). In production, this doesn't matter (the data is the data).
 *
 *
 * Pipeline architecture (v2 — with Kalman parameter tracker):
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
 * Two modes:
 *   SYNC  — dBPF blocks at window boundary until SMC² finishes. Deterministic.
 *   ASYNC — dBPF never blocks. SMC² pushes when ready. Production mode.
 */

#ifndef SMC2_PIPELINE_H
#define SMC2_PIPELINE_H

#include "smc2_engine.cuh"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif


/*═══════════════════════════════════════════════════════════════════════════════
 * ███████████████████████████████████████████████████████████████████████████
 * §1: CONVERGENCE DIAGNOSTICS
 *
 * Rolling-window R̂ for amnesiac SMC² + Kalman parameter tracker.
 * Analogous to Gelman-Rubin R̂ for MCMC chains, using amnesiac SMC² windows
 * as independent "chains".
 *
 * Diagnostics:
 *   R̂ (per-param)    √(B/W) — between-window vs mean posterior variance
 *   Mahalanobis d²    Kalman innovation ν'S⁻¹ν. Under correct model ~ χ²(n_free)
 *   P-trace           tr(P) — should decrease monotonically then plateau
 *   CV (per-param)    √P_ii / |x_i| — relative uncertainty
 * ███████████████████████████████████████████████████████████████████████████
 *═══════════════════════════════════════════════════════════════════════════════*/

/* ── Constants ── */

#ifndef CONV_DIAG_N_PARAMS
#define CONV_DIAG_N_PARAMS  8
#endif

#define CONV_DIAG_MAX_BUF   16   /* Max rolling window of posteriors */

/* ── Structures ── */

typedef struct {
    float theta_hat[CONV_DIAG_MAX_BUF][CONV_DIAG_N_PARAMS];  /* posterior mean  */
    float sigma_diag[CONV_DIAG_MAX_BUF][CONV_DIAG_N_PARAMS]; /* diag(Σ_k)      */
    float mahal_d2[CONV_DIAG_MAX_BUF];                        /* Mahalanobis d²  */
    float p_trace[CONV_DIAG_MAX_BUF];                         /* tr(P) snapshot  */

    int count;   /* Total windows pushed (monotonically increasing) */
    int head;    /* Next write index (circular, mod M)              */
    int M;       /* Rolling buffer size                             */
} ConvergenceDiag;

typedef struct ConvergenceReport {
    /* Per-parameter diagnostics */
    float rhat[CONV_DIAG_N_PARAMS];
    float B[CONV_DIAG_N_PARAMS];          /* Between-window variance      */
    float W[CONV_DIAG_N_PARAMS];          /* Mean within-window variance  */
    float cv[CONV_DIAG_N_PARAMS];         /* √P_ii / |x_i|               */
    int   converged[CONV_DIAG_N_PARAMS];  /* 1=yes, 0=no, -1=locked      */

    /* Aggregate */
    float mahal_mean;
    float mahal_expected;   /* n_free (χ² dof) */
    float p_trace_current;
    float p_trace_prev;
    int   n_free;
    int   n_converged;
    int   all_converged;    /* 1 if all free params converged */
    int   ready;            /* 1 if buffer full (count >= M)  */
} ConvergenceReport;

/* ── Init ── */

static inline void conv_diag_init(ConvergenceDiag* d, int M) {
    memset(d, 0, sizeof(*d));
    d->M = (M > CONV_DIAG_MAX_BUF) ? CONV_DIAG_MAX_BUF : (M < 2 ? 2 : M);
}

/* ── Push one window's results ── */

static inline void conv_diag_push(ConvergenceDiag* d,
                                   const float theta_hat[],
                                   const float sigma_diag[],
                                   float mahal_d2,
                                   float p_trace)
{
    int idx = d->head;
    memcpy(d->theta_hat[idx],  theta_hat,  CONV_DIAG_N_PARAMS * sizeof(float));
    memcpy(d->sigma_diag[idx], sigma_diag, CONV_DIAG_N_PARAMS * sizeof(float));
    d->mahal_d2[idx] = mahal_d2;
    d->p_trace[idx]  = p_trace;

    d->head = (d->head + 1) % d->M;
    d->count++;
}

/* ── Compute convergence report ── */

static inline void conv_diag_report(const ConvergenceDiag* d,
                                     const float kalman_x[],
                                     const float kalman_P_diag[],
                                     const int free_mask[],
                                     float rhat_thresh,
                                     ConvergenceReport* r)
{
    memset(r, 0, sizeof(*r));

    int n = (d->count < d->M) ? d->count : d->M;
    r->ready = (n >= d->M);

    if (n < 2) return;

    /* Per-parameter: B, W, R̂ */
    for (int i = 0; i < CONV_DIAG_N_PARAMS; i++) {
        float sum = 0.0f, sum2 = 0.0f, sumW = 0.0f;
        for (int k = 0; k < n; k++) {
            float v = d->theta_hat[k][i];
            sum  += v;
            sum2 += v * v;
            sumW += d->sigma_diag[k][i];
        }

        float mean = sum / (float)n;
        float B = (sum2 / (float)n) - mean * mean;
        float W = sumW / (float)n;

        if (n > 1) B *= (float)n / (float)(n - 1);

        r->B[i] = B;
        r->W[i] = W;
        r->rhat[i] = (W > 1e-12f) ? sqrtf(B / W) : ((B > 1e-12f) ? 99.0f : 0.0f);

        /* CV from Kalman state */
        r->cv[i] = (fabsf(kalman_x[i]) > 1e-8f)
            ? sqrtf(kalman_P_diag[i]) / fabsf(kalman_x[i])
            : ((kalman_P_diag[i] > 1e-12f) ? 99.0f : 0.0f);

        /* Per-param convergence */
        if (free_mask[i]) {
            r->n_free++;
            r->converged[i] = (r->rhat[i] < rhat_thresh) ? 1 : 0;
            if (r->converged[i]) r->n_converged++;
        } else {
            r->converged[i] = -1;
        }
    }

    r->all_converged = (r->n_free > 0 && r->n_converged == r->n_free);

    /* Mahalanobis mean */
    float msum = 0.0f;
    for (int k = 0; k < n; k++) msum += d->mahal_d2[k];
    r->mahal_mean     = msum / (float)n;
    r->mahal_expected  = (float)r->n_free;

    /* P-trace */
    int last = (d->head - 1 + d->M) % d->M;
    r->p_trace_current = d->p_trace[last];
    if (n >= 2) {
        int prev = (d->head - 2 + d->M) % d->M;
        r->p_trace_prev = d->p_trace[prev];
    }
}

/* ── Diagonal Mahalanobis d² helper ── */

static inline float conv_diag_mahal_diag(const float nu[],
                                          const float S_diag[],
                                          int n_params)
{
    float d2 = 0.0f;
    for (int i = 0; i < n_params; i++)
        if (S_diag[i] > 1e-12f)
            d2 += nu[i] * nu[i] / S_diag[i];
    return d2;
}

/* ── Diagnostic printing ── */

static inline void conv_diag_print_header(const char* param_names[]) {
    printf(" Win |");
    for (int i = 0; i < CONV_DIAG_N_PARAMS; i++)
        printf(" %5s", param_names[i]);
    printf(" | Mahal  | P-trace | Conv\n");

    printf("-----|");
    for (int i = 0; i < CONV_DIAG_N_PARAMS; i++) printf("------");
    printf("-|--------|---------|------\n");
}

static inline void conv_diag_print_line(const ConvergenceReport* r, int win_idx) {
    printf(" %3d |", win_idx);

    if (!r->ready) {
        for (int i = 0; i < CONV_DIAG_N_PARAMS; i++) printf("    — ");
        printf("| d²=%5.1f | Ptr=%7.4f | —\n", r->mahal_mean, r->p_trace_current);
        return;
    }

    for (int i = 0; i < CONV_DIAG_N_PARAMS; i++) {
        if (r->converged[i] == -1)
            printf("  lock");
        else if (r->converged[i])
            printf(" %5.2f", r->rhat[i]);
        else
            printf(" %s%.2f%s", "\033[91m", r->rhat[i], "\033[0m");
    }

    printf(" | d²=%5.1f", r->mahal_mean);
    printf(" | Ptr=%7.4f", r->p_trace_current);
    printf(" | %d/%d", r->n_converged, r->n_free);
    if (r->all_converged) printf("  ✓");
    printf("\n");
}


/*═══════════════════════════════════════════════════════════════════════════════
 * ███████████████████████████████████████████████████████████████████████████
 * §2: BPF PIPELINE WIRING
 *
 * Connects the per-tick dBPF (vol estimator) with the windowed SMC²
 * (parameter learner). Handles observation transform, window triggering,
 * parameter push, and sync/async modes.
 *
 * This section requires gpu_bpf_full.cuh.
 * Phased learning is now internal to param_tracker — the pipeline
 * just orchestrates SMC2 windows and pushes params to BPF.
 * ███████████████████████████████████████████████████████████████████████████
 *═══════════════════════════════════════════════════════════════════════════════*/

#if __has_include("gpu_bpf_full.cuh")
#include "gpu_bpf_full.cuh"
#define SMC2_PIPELINE_AVAILABLE 1
#else
#define SMC2_PIPELINE_AVAILABLE 0
#endif

#if SMC2_PIPELINE_AVAILABLE

/* ── Pipeline configuration ── */

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
    c.sync_mode      = 1;
    c.push_rho       = 1;
    c.obs_buffer_sec = 0.0f;
    return c;
}

/* ── Pipeline state ── */

typedef struct {
    /* Sub-systems (owned externally — pipeline does not create/destroy) */
    GpuBpfState*    bpf;
    SMC2StateCUDA*  smc2;

    /* CUDA streams */
    cudaStream_t    stream_bpf;
    cudaStream_t    stream_smc2;

    /* Configuration */
    PipelineConfig  config;

    /* Observation ring buffer for SMC² windows */
    float*          obs_buffer;
    int             obs_capacity;
    int             obs_write_pos;
    int             obs_count;

    /* Tick counters */
    int             bpf_tick;
    int             smc2_tick;
    int             next_window_trigger;

    /* Last push diagnostics */
    float           last_push_mu;
    float           last_push_rho;
    float           last_push_sigma_z;
    int             last_push_tick;
    int             n_pushes;

    /* SMC² running state */
    int             smc2_running;
} SMC2BpfPipeline;

/* ── Lifecycle ── */

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

    p->obs_capacity  = config.window_size * 2;
    p->obs_buffer    = (float*)calloc(p->obs_capacity, sizeof(float));

    p->next_window_trigger = config.window_size;

    p->last_push_tick = -1;

    return p;
}

static inline void pipeline_destroy(SMC2BpfPipeline* p) {
    if (!p) return;
    cudaStreamDestroy(p->stream_bpf);
    cudaStreamDestroy(p->stream_smc2);
    free(p->obs_buffer);
    free(p);
}

/* ── Internal: extract posterior and push to dBPF ── */

static inline void pipeline_push_params(SMC2BpfPipeline* p) {
    cudaStreamSynchronize(p->stream_smc2);

    /* Extract posterior means via public API */
    float theta_mean[N_PARAMS];
    smc2_cuda_get_theta_mean(p->smc2, theta_mean);

    float post_rho         = theta_mean[0];
    float post_sigma_total = theta_mean[1];
    float post_r_split     = theta_mean[2];
    float post_mu          = theta_mean[3];  /* mu_base = floor */
    float post_sigma_z     = post_r_split * post_sigma_total;

    gpu_bpf_set_mu(p->bpf, post_mu);
    if (p->config.push_rho)
        gpu_bpf_set_rho(p->bpf, post_rho);

    /* σ_z cannot be pushed — gpu_bpf has no set_sigma_z().
     * TODO: Add gpu_bpf_set_sigma_z() to the BPF API. */

    p->last_push_mu      = post_mu;
    p->last_push_rho     = post_rho;
    p->last_push_sigma_z = post_sigma_z;
    p->last_push_tick    = p->bpf_tick;
    p->n_pushes++;
}

/* ── Internal: launch SMC² window ── */

static inline void pipeline_launch_smc2_window(SMC2BpfPipeline* p) {
    int W = p->config.window_size;
    int start = p->obs_count - W;
    if (start < 0) return;

    float* window_obs = (float*)malloc(W * sizeof(float));
    for (int i = 0; i < W; i++) {
        int buf_idx = (start + i) % p->obs_capacity;
        window_obs[i] = p->obs_buffer[buf_idx];
    }

    smc2_cuda_init_from_prior(p->smc2);
    smc2_cuda_update_batch(p->smc2, window_obs, W);

    free(window_obs);

    p->smc2_tick = p->obs_count;
    p->next_window_trigger += p->config.stride;
    p->smc2_running = 1;
}

/* ── Main tick function ── */

/**
 * @brief Process one tick through the pipeline.
 *
 * Called once per observation with raw return y_t.
 * Returns the dBPF result for this tick.
 *
 * SYNC mode:  blocks at window boundary until SMC² finishes + pushes params.
 * ASYNC mode: never blocks. SMC² push happens when ready.
 */
static inline BpfResult pipeline_step(SMC2BpfPipeline* p, float y_t) {

    /* 1. Store observation in ring buffer */
    p->obs_buffer[p->obs_write_pos % p->obs_capacity] = y_t;
    p->obs_write_pos++;
    p->obs_count++;

    /* 2. Run dBPF (always, never blocks) */
    BpfResult result = gpu_bpf_step(p->bpf, y_t);
    p->bpf_tick++;

    /* 3. Check if async SMC² has finished */
    if (!p->config.sync_mode && p->smc2_running) {
        cudaError_t status = cudaStreamQuery(p->stream_smc2);
        if (status == cudaSuccess) {
            pipeline_push_params(p);
            p->smc2_running = 0;
        }
    }

    /* 4. Window boundary: launch or wait for SMC² */
    if (p->bpf_tick >= p->next_window_trigger) {
        if (p->config.sync_mode) {
            if (p->smc2_running) {
                pipeline_push_params(p);
                p->smc2_running = 0;
            }
            pipeline_launch_smc2_window(p);
            pipeline_push_params(p);
            p->smc2_running = 0;
        } else {
            if (!p->smc2_running) {
                pipeline_launch_smc2_window(p);
            }
        }
    }

    return result;
}

/* ── Diagnostics ── */

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

#endif /* SMC2_PIPELINE_AVAILABLE */


#ifdef __cplusplus
}
#endif

#endif /* SMC2_PIPELINE_H */
