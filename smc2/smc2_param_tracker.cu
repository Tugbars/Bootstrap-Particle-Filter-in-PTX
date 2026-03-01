/**
 * @file smc2_param_tracker.cu
 * @brief Kalman Parameter Tracker — implementation (v2, warm-start)
 *
 * Sliding-window SMC² produces θ̂, Σ measurements.
 * Plain linear Kalman filter fuses them over time.
 * Curves evaluated at z̄ produce BPF-ready scalar parameters.
 *
 * v2 changes:
 *   - Warm-start: after first window, SMC² is initialized from
 *     N(x, P+Q) instead of flat prior. This constrains slow parameters
 *     to their accumulated estimates while letting fast params explore.
 *   - Configurable warm_start flag (default on).
 *
 * The 8×8 Kalman math is host-side (too small for GPU), but this
 * file is .cu because it calls smc2_cuda_* functions directly.
 *
 * Build:
 *   nvcc -O2 -arch=sm_120 -o tracker smc2_param_tracker.cu smc2_rbpf_cuda.cu \
 *        -lcurand --expt-relaxed-constexpr
 */

#include "smc2_param_tracker.cuh"
#include "smc2_convergence_diag.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* Gate modes — per-parameter convergence strategy */
#define GATE_KALMAN_MIN  0   /* Push Kalman x after min_windows. No R̂. Fast params. */
#define GATE_RHAT_LATCH  1   /* R̂ gate with one-way latch. Curve params.            */
#define GATE_LOCKED      2   /* Always prior default. Param not free.               */

/*═══════════════════════════════════════════════════════════════════════════════
 * Internal state
 *═══════════════════════════════════════════════════════════════════════════════*/

struct ParamTracker {
    /* ── Kalman filter state ── */
    float x[N_PARAMS];                      /* Filtered parameter estimate */
    float P[N_PARAMS * N_PARAMS];           /* Error covariance (8×8) */
    float Q[N_PARAMS * N_PARAMS];           /* Process noise (diagonal) */
    float P_floor[N_PARAMS];                /* Minimum P diagonal (prevents lockup) */
    int   initialized;                       /* 0 until first measurement */

    /* ── Observation circular buffer ── */
    float* y_buf;
    int    buf_capacity;
    int    buf_head;                         /* Next write position */
    int    buf_count;                        /* Total observations fed */

    /* ── Window configuration ── */
    int    window_size;
    int    stride;
    int    ticks_since_window;               /* Counts up to stride */

    /* ── SMC² instance ── */
    SMC2StateCUDA* smc2;
    int    N_theta;
    int    N_inner;

    /* ── θ(z) curve (fixed, updated externally) ── */
    float  theta_base;
    float  theta_scale;
    float  theta_rate;

    /* ── Convergence diagnostics + tiered gating ── */
    ConvergenceDiag conv_diag;               /* Rolling window R̂ buffer         */
    float  prior_default[N_PARAMS];          /* Fallback values for non-converged */
    int    free_mask[N_PARAMS];              /* 1 = free (check R̂), 0 = locked   */
    float  rhat_threshold;                   /* R̂ below this → converged (default 1.5) */
    int    conv_diag_M;                      /* Rolling buffer size (default 8)   */

    /* Per-param gate mode:
     *   0 = GATE_KALMAN_MIN : push Kalman after min_windows, no R̂ check
     *   1 = GATE_RHAT_LATCH : R̂ gate with one-way latch (never reverts)
     *   2 = GATE_LOCKED     : always prior default (param not free)      */
    int    gate_mode[N_PARAMS];
    int    converged[N_PARAMS];              /* Per-param: 1=converged, 0=not, -1=locked */
    int    min_windows;                      /* GATE_KALMAN_MIN fires after this (default 2) */
    int    n_windows_completed;              /* Total windows run so far */

    /* ── Latest snapshot cache ── */
    ParamSnapshot snap;
};

/*═══════════════════════════════════════════════════════════════════════════════
 * 8×8 Linear Algebra Helpers (host-side, tiny matrices)
 *═══════════════════════════════════════════════════════════════════════════════*/

#define D N_PARAMS

/** @brief C = A + B (D×D) */
static void mat_add(float* C, const float* A, const float* B) {
    for (int i = 0; i < D * D; i++) C[i] = A[i] + B[i];
}

/** @brief Cholesky decomposition: A = L·Lᵀ, returns L in lower triangle of L_out
 *  @return 1 on success, 0 if not positive definite (falls back to diagonal) */
static int cholesky(const float* A, float* L_out) {
    memset(L_out, 0, D * D * sizeof(float));
    for (int i = 0; i < D; i++) {
        for (int j = 0; j <= i; j++) {
            float sum = 0.0f;
            for (int k = 0; k < j; k++)
                sum += L_out[i * D + k] * L_out[j * D + k];
            if (i == j) {
                float val = A[i * D + i] - sum;
                if (val <= 0.0f) return 0;
                L_out[i * D + j] = sqrtf(val);
            } else {
                float diag = L_out[j * D + j];
                L_out[i * D + j] = (diag > 1e-12f) ? (A[i * D + j] - sum) / diag : 0.0f;
            }
        }
    }
    return 1;
}

/** @brief Solve L·x = b by forward substitution (L is lower triangular) */
static void solve_lower(const float* L, const float* b, float* x) {
    for (int i = 0; i < D; i++) {
        float sum = b[i];
        for (int j = 0; j < i; j++) sum -= L[i * D + j] * x[j];
        x[i] = sum / L[i * D + i];
    }
}

/** @brief Solve Lᵀ·x = b by back substitution */
static void solve_upper(const float* L, const float* b, float* x) {
    for (int i = D - 1; i >= 0; i--) {
        float sum = b[i];
        for (int j = i + 1; j < D; j++) sum -= L[j * D + i] * x[j];
        x[i] = sum / L[i * D + i];
    }
}

/**
 * @brief Solve S·X = B where S = L·Lᵀ, B is D×D → X = S⁻¹·B
 */
static void solve_spd_matrix(const float* L, const float* B, float* X) {
    float tmp[D];
    for (int j = 0; j < D; j++) {
        float col[D];
        for (int i = 0; i < D; i++) col[i] = B[i * D + j];
        solve_lower(L, col, tmp);
        float x_col[D];
        solve_upper(L, tmp, x_col);
        for (int i = 0; i < D; i++) X[i * D + j] = x_col[i];
    }
}

/** @brief C = A · B (D×D) */
static void mat_mul(float* C, const float* A, const float* B) {
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++) {
            float sum = 0.0f;
            for (int k = 0; k < D; k++) sum += A[i * D + k] * B[k * D + j];
            C[i * D + j] = sum;
        }
    }
}

/** @brief C = I - A (D×D) */
static void mat_eye_minus(float* C, const float* A) {
    for (int i = 0; i < D; i++)
        for (int j = 0; j < D; j++)
            C[i * D + j] = ((i == j) ? 1.0f : 0.0f) - A[i * D + j];
}

#undef D

/*═══════════════════════════════════════════════════════════════════════════════
 * Kalman Predict + Update
 *═══════════════════════════════════════════════════════════════════════════════*/

static void kalman_update(ParamTracker* t, const float* z_meas, const float* R) {
    float P_bar[N_PARAMS * N_PARAMS];
    float S[N_PARAMS * N_PARAMS];
    float L[N_PARAMS * N_PARAMS];
    float Kt[N_PARAMS * N_PARAMS];
    float K[N_PARAMS * N_PARAMS];
    float IminusK[N_PARAMS * N_PARAMS];
    float P_new[N_PARAMS * N_PARAMS];

    /* Predict: P̄ = P + Q */
    mat_add(P_bar, t->P, t->Q);

    /* Innovation covariance: S = P̄ + R */
    mat_add(S, P_bar, R);

    /* Regularize S diagonal */
    for (int i = 0; i < N_PARAMS; i++)
        S[i * N_PARAMS + i] += 1e-8f;

    /* Cholesky of S */
    if (!cholesky(S, L)) {
        fprintf(stderr, "param_tracker: Cholesky failed on S, using diagonal fallback\n");
        memset(L, 0, sizeof(L));
        for (int i = 0; i < N_PARAMS; i++)
            L[i * N_PARAMS + i] = sqrtf(fmaxf(S[i * N_PARAMS + i], 1e-8f));
    }

    /* K = P_bar * S^{-1}  via  solve S*Kt = P_bar then transpose */
    solve_spd_matrix(L, P_bar, Kt);
    for (int i = 0; i < N_PARAMS; i++)
        for (int j = 0; j < N_PARAMS; j++)
            K[i * N_PARAMS + j] = Kt[j * N_PARAMS + i];

    /* x = x̄ + K · (z - x̄) */
    float innov[N_PARAMS];
    for (int i = 0; i < N_PARAMS; i++)
        innov[i] = z_meas[i] - t->x[i];

    for (int i = 0; i < N_PARAMS; i++) {
        float sum = 0.0f;
        for (int j = 0; j < N_PARAMS; j++)
            sum += K[i * N_PARAMS + j] * innov[j];
        t->x[i] += sum;
    }

    /* P = (I - K) · P̄ */
    mat_eye_minus(IminusK, K);
    mat_mul(P_new, IminusK, P_bar);

    /* Symmetrize */
    for (int i = 0; i < N_PARAMS; i++)
        for (int j = 0; j < N_PARAMS; j++)
            t->P[i * N_PARAMS + j] = 0.5f * (P_new[i * N_PARAMS + j] +
                                               P_new[j * N_PARAMS + i]);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * Public API
 *═══════════════════════════════════════════════════════════════════════════════*/

ParamTracker* param_tracker_create(int window_size, int stride,
                                    int N_theta, int N_inner) {
    ParamTracker* t = (ParamTracker*)calloc(1, sizeof(ParamTracker));
    if (!t) return NULL;

    t->window_size = window_size;
    t->stride = stride;
    t->N_theta = N_theta;
    t->N_inner = N_inner;

    /* Circular buffer: capacity = window_size + some margin */
    t->buf_capacity = window_size + stride;
    t->y_buf = (float*)calloc(t->buf_capacity, sizeof(float));
    if (!t->y_buf) { free(t); return NULL; }
    t->buf_head = 0;
    t->buf_count = 0;

    /* Allocate SMC² instance */
    t->smc2 = smc2_cuda_alloc(N_theta, N_inner);
    if (!t->smc2) { free(t->y_buf); free(t); return NULL; }

    /* Default fixed-lag */
    smc2_cuda_set_fixed_lag(t->smc2, 200);

    /* Default θ(z) curve */
    t->theta_base = 0.02f;
    t->theta_scale = 0.08f;
    t->theta_rate = 1.5f;

    /* Initialize Kalman: large P (uninformative), default Q */
    t->initialized = 0;
    memset(t->x, 0, sizeof(t->x));
    memset(t->P, 0, sizeof(t->P));
    for (int i = 0; i < N_PARAMS; i++)
        t->P[i * N_PARAMS + i] = 10.0f;

    /* Default drift rates (Q diagonal) */
    memset(t->Q, 0, sizeof(t->Q));
    t->Q[0 * N_PARAMS + 0] = 1e-5f;   /* rho */
    t->Q[1 * N_PARAMS + 1] = 1e-5f;   /* sigma_total */
    t->Q[2 * N_PARAMS + 2] = 1e-5f;   /* r_split */
    t->Q[3 * N_PARAMS + 3] = 1e-3f;   /* mu_base — fastest */
    t->Q[4 * N_PARAMS + 4] = 1e-7f;   /* mu_scale */
    t->Q[5 * N_PARAMS + 5] = 1e-7f;   /* mu_rate */
    t->Q[6 * N_PARAMS + 6] = 1e-7f;   /* sigma_scale */
    t->Q[7 * N_PARAMS + 7] = 1e-7f;   /* sigma_rate */

    /* Default P floor */
    t->P_floor[0] = 1e-4f;   /* rho */
    t->P_floor[1] = 1e-4f;   /* sigma_total */
    t->P_floor[2] = 1e-3f;   /* r_split */
    t->P_floor[3] = 1e-2f;   /* mu_base */
    t->P_floor[4] = 1e-3f;   /* mu_scale */
    t->P_floor[5] = 1e-3f;   /* mu_rate */
    t->P_floor[6] = 1e-4f;   /* sigma_scale */
    t->P_floor[7] = 1e-3f;   /* sigma_rate */

    t->ticks_since_window = 0;
    memset(&t->snap, 0, sizeof(t->snap));

    /* ── Convergence diagnostics ── */
    t->conv_diag_M = 8;
    conv_diag_init(&t->conv_diag, t->conv_diag_M);
    t->rhat_threshold = 1.5f;
    t->min_windows = 2;
    t->n_windows_completed = 0;

    /* Prior defaults — BPF uses these until param converges.
     * Match prior means from smc2_cuda_set_default_priors(). */
    t->prior_default[0] = 0.85f;   /* ρ         */
    t->prior_default[1] = 0.30f;   /* σ_total   */
    t->prior_default[2] = 0.50f;   /* r_split   */
    t->prior_default[3] = -10.0f;  /* μ_base    */
    t->prior_default[4] = 3.00f;   /* μ_scale   */
    t->prior_default[5] = 1.00f;   /* μ_rate    */
    t->prior_default[6] = 0.50f;   /* σ_scale   */
    t->prior_default[7] = 0.80f;   /* σ_rate    */

    /* Default gate modes:
     *   Phase 1 (ρ, σ_total, r, μ_base): GATE_KALMAN_MIN — push after 2 windows
     *   Phase 2/3 (curve shapes):         GATE_RHAT_LATCH — R̂ gate with latch
     * Caller overrides via set_free_mask / set_gate_mode. */
    t->gate_mode[0] = GATE_KALMAN_MIN;   /* ρ         */
    t->gate_mode[1] = GATE_KALMAN_MIN;   /* σ_total   */
    t->gate_mode[2] = GATE_KALMAN_MIN;   /* r_split   */
    t->gate_mode[3] = GATE_KALMAN_MIN;   /* μ_base    */
    t->gate_mode[4] = GATE_RHAT_LATCH;   /* μ_scale   */
    t->gate_mode[5] = GATE_RHAT_LATCH;   /* μ_rate    */
    t->gate_mode[6] = GATE_RHAT_LATCH;   /* σ_scale   */
    t->gate_mode[7] = GATE_RHAT_LATCH;   /* σ_rate    */

    /* Default: all params free (caller adjusts via set_free_mask) */
    for (int i = 0; i < N_PARAMS; i++)
        t->free_mask[i] = 1;

    /* All non-converged initially */
    for (int i = 0; i < N_PARAMS; i++)
        t->converged[i] = 0;

    return t;
}

void param_tracker_destroy(ParamTracker* t) {
    if (!t) return;
    if (t->smc2) smc2_cuda_free(t->smc2);
    free(t->y_buf);
    free(t);
}

void param_tracker_feed(ParamTracker* t, float y_obs) {
    t->y_buf[t->buf_head] = y_obs;
    t->buf_head = (t->buf_head + 1) % t->buf_capacity;
    t->buf_count++;
    t->ticks_since_window++;
}

int param_tracker_window_ready(const ParamTracker* t) {
    return (t->ticks_since_window >= t->stride &&
            t->buf_count >= t->window_size);
}

void param_tracker_run_window(ParamTracker* t) {
    int W = t->window_size;

    /* Extract window from circular buffer: last W observations */
    float* window = (float*)malloc(W * sizeof(float));
    int start = (t->buf_head - W + t->buf_capacity) % t->buf_capacity;
    for (int i = 0; i < W; i++)
        window[i] = t->y_buf[(start + i) % t->buf_capacity];

    /* ── Initialize SMC² particles ──────────────────────────────────── *
     * Always cold-start from prior. Warm-start was tested and shelved   *
     * (+25% RMSE from feedback trap). See design decisions §13.         *
     * ──────────────────────────────────────────────────────────────── */
    smc2_cuda_init_from_prior(t->smc2);

    /* Run SMC² on this window */
    smc2_cuda_update_batch(t->smc2, window, W);

    /* Extract posterior mean and covariance */
    float z_meas[N_PARAMS];
    float R[N_PARAMS * N_PARAMS];
    smc2_cuda_get_theta_cov(t->smc2, z_meas, R);

    /* Extract z̄ */
    float z_mean = smc2_cuda_get_z_mean(t->smc2);

    /* Diagnostics */
    float ess = smc2_cuda_get_outer_ess(t->smc2);
    float accept_rate = (t->smc2->n_rejuv_total > 0)
        ? (float)t->smc2->n_rejuv_accepts / t->smc2->n_rejuv_total
        : 0.0f;

    /* First measurement: initialize Kalman state directly */
    if (!t->initialized) {
        memcpy(t->x, z_meas, N_PARAMS * sizeof(float));
        memcpy(t->P, R, N_PARAMS * N_PARAMS * sizeof(float));
        for (int i = 0; i < N_PARAMS; i++)
            t->P[i * N_PARAMS + i] += 1e-6f;
        t->initialized = 1;

        /* First window — no innovation yet, push with d²=0 */
        float sigma_diag[N_PARAMS];
        for (int i = 0; i < N_PARAMS; i++)
            sigma_diag[i] = R[i * N_PARAMS + i];
        float p_tr = 0.0f;
        for (int i = 0; i < N_PARAMS; i++)
            p_tr += t->P[i * N_PARAMS + i];
        conv_diag_push(&t->conv_diag, z_meas, sigma_diag, 0.0f, p_tr);
    } else {
        /* Save predicted state for innovation (before Kalman update) */
        float x_pre[N_PARAMS];
        memcpy(x_pre, t->x, N_PARAMS * sizeof(float));
        /* P_bar = P + Q (predicted covariance) */
        float P_bar_diag[N_PARAMS];
        for (int i = 0; i < N_PARAMS; i++)
            P_bar_diag[i] = t->P[i * N_PARAMS + i] + t->Q[i * N_PARAMS + i];

        kalman_update(t, z_meas, R);

        /* ── Convergence diagnostic ──────────────────────────────────── */
        float sigma_diag[N_PARAMS];
        for (int i = 0; i < N_PARAMS; i++)
            sigma_diag[i] = R[i * N_PARAMS + i];

        /* Innovation: ν = θ̂_k − x_predicted */
        float nu[N_PARAMS], S_diag[N_PARAMS];
        for (int i = 0; i < N_PARAMS; i++) {
            nu[i] = z_meas[i] - x_pre[i];
            S_diag[i] = P_bar_diag[i] + sigma_diag[i];
        }
        float d2 = conv_diag_mahal_diag(nu, S_diag, N_PARAMS);

        /* P-trace */
        float p_tr = 0.0f;
        for (int i = 0; i < N_PARAMS; i++)
            p_tr += t->P[i * N_PARAMS + i];

        conv_diag_push(&t->conv_diag, z_meas, sigma_diag, d2, p_tr);

        /* Compute R̂ report (for diagnostics — not used for KALMAN_MIN params) */
        ConvergenceReport rpt;
        float cur_P_diag[N_PARAMS];
        for (int i = 0; i < N_PARAMS; i++)
            cur_P_diag[i] = t->P[i * N_PARAMS + i];
        conv_diag_report(&t->conv_diag, t->x, cur_P_diag,
                         t->free_mask, t->rhat_threshold, &rpt);
    }

    t->n_windows_completed++;

    /* Enforce P floor */
    for (int i = 0; i < N_PARAMS; i++) {
        if (t->P[i * N_PARAMS + i] < t->P_floor[i])
            t->P[i * N_PARAMS + i] = t->P_floor[i];
    }

    /* ── Tiered convergence gating ──────────────────────────────────────
     *
     * GATE_KALMAN_MIN (fast params: ρ, σ_total, r, μ_base):
     *   Push Kalman estimate after min_windows completed. No R̂ check.
     *   One-way: once converged, stays converged.
     *   Rationale: Kalman is designed to fuse noisy amnesiac windows.
     *   Waiting for them to agree defeats the purpose.
     *
     * GATE_RHAT_LATCH (curve params: μ_scale, μ_rate, σ_scale, σ_rate):
     *   R̂ gate with one-way latch. Once R̂ drops below threshold,
     *   converged latches to 1 and never reverts. Regime transitions
     *   cause natural R̂ spikes — reverting would slam BPF 2D back
     *   to prior defaults mid-trade.
     *
     * GATE_LOCKED:
     *   Always prior default. Param is not free (phased learning).
     * ──────────────────────────────────────────────────────────────── */

    /* Get latest R̂ report for RHAT_LATCH params */
    ConvergenceReport rpt_latest;
    {
        float cur_P_diag[N_PARAMS];
        for (int i = 0; i < N_PARAMS; i++)
            cur_P_diag[i] = t->P[i * N_PARAMS + i];
        conv_diag_report(&t->conv_diag, t->x, cur_P_diag,
                         t->free_mask, t->rhat_threshold, &rpt_latest);
    }

    for (int i = 0; i < N_PARAMS; i++) {
        switch (t->gate_mode[i]) {
        case GATE_KALMAN_MIN:
            /* One-way latch: converge after min_windows, never revert */
            if (t->converged[i] != 1 &&
                t->n_windows_completed >= t->min_windows) {
                t->converged[i] = 1;
            }
            break;

        case GATE_RHAT_LATCH:
            /* One-way latch: converge when R̂ drops, never revert */
            if (t->converged[i] != 1 &&
                rpt_latest.ready &&
                rpt_latest.converged[i] == 1) {
                t->converged[i] = 1;
            }
            break;

        case GATE_LOCKED:
        default:
            t->converged[i] = -1;
            break;
        }
    }

    /* Build gated parameter vector */
    float gated[N_PARAMS];
    for (int i = 0; i < N_PARAMS; i++) {
        gated[i] = (t->converged[i] == 1) ? t->x[i] : t->prior_default[i];
    }

    /* Derive physical parameters from GATED state */
    float sigma_total = gated[1];
    float r_split = gated[2];
    float mu_base = gated[3];
    float mu_scale = gated[4];
    float mu_rate = gated[5];
    float sigma_scale = gated[6];
    float sigma_rate = gated[7];

    float sigma_z = r_split * sigma_total;
    float sigma_base = sqrtf(fmaxf(1.0f - r_split * r_split, 1e-6f)) * sigma_total;

    /* Evaluate curves at z̄ using gated values */
    float mu_at_z = eval_curve_host(mu_base, mu_scale, mu_rate, z_mean);
    float sigma_h_at_z = eval_curve_host(sigma_base, sigma_scale, sigma_rate, z_mean);
    float theta_at_z = eval_curve_host(t->theta_base, t->theta_scale, t->theta_rate, z_mean);

    /* Update snapshot — gated theta + raw Kalman P (for diagnostics) */
    memcpy(t->snap.theta, gated, N_PARAMS * sizeof(float));
    for (int i = 0; i < N_PARAMS; i++)
        t->snap.P_diag[i] = t->P[i * N_PARAMS + i];
    t->snap.sigma_z = sigma_z;
    t->snap.sigma_base = sigma_base;
    t->snap.z_mean = z_mean;
    t->snap.mu = mu_at_z;
    t->snap.sigma_h = sigma_h_at_z;
    t->snap.theta_speed = theta_at_z;
    t->snap.n_updates++;
    t->snap.last_accept_rate = accept_rate;
    t->snap.last_ess = ess;

    t->ticks_since_window = 0;

    free(window);
}

void param_tracker_get_snapshot(const ParamTracker* t, ParamSnapshot* snap) {
    *snap = t->snap;
}

void param_tracker_set_drift(ParamTracker* t, const DriftConfig* drift) {
    memset(t->Q, 0, N_PARAMS * N_PARAMS * sizeof(float));
    t->Q[0 * N_PARAMS + 0] = drift->q_rho;
    t->Q[1 * N_PARAMS + 1] = drift->q_sigma_total;
    t->Q[2 * N_PARAMS + 2] = drift->q_r_split;
    t->Q[3 * N_PARAMS + 3] = drift->q_mu_base;
    t->Q[4 * N_PARAMS + 4] = drift->q_mu_scale;
    t->Q[5 * N_PARAMS + 5] = drift->q_mu_rate;
    t->Q[6 * N_PARAMS + 6] = drift->q_sigma_scale;
    t->Q[7 * N_PARAMS + 7] = drift->q_sigma_rate;
}

void param_tracker_set_theta_curve(ParamTracker* t, float base, float scale, float rate) {
    t->theta_base = base;
    t->theta_scale = scale;
    t->theta_rate = rate;
}

void param_tracker_set_P_floor(ParamTracker* t, const float* p_floor) {
    memcpy(t->P_floor, p_floor, N_PARAMS * sizeof(float));
}

SMC2StateCUDA* param_tracker_get_smc2(ParamTracker* t) {
    return t->smc2;
}

void param_tracker_get_P(const ParamTracker* t, float* P_out) {
    memcpy(P_out, t->P, N_PARAMS * N_PARAMS * sizeof(float));
}

void param_tracker_force_cold(ParamTracker* t) {
    /* Force next window to cold-start from prior instead of warm-start.
     *
     * Call this after a phase transition unlocks new parameters.
     * The Kalman has no information about the newly-freed dimensions,
     * so warm-starting would place all particles at the same (fixed)
     * value for those params — no diversity, no exploration.
     *
     * This resets the Kalman so the next window explores from prior,
     * then subsequent windows resume warm-starting once the Kalman
     * has incorporated the new dimensions. */
    t->initialized = 0;
    t->snap.n_updates = 0;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * Diagnostic: Print current tracker state
 *═══════════════════════════════════════════════════════════════════════════════*/

void param_tracker_print(const ParamTracker* t) {
    static const char* names[N_PARAMS] = {
        "rho", "sigma_total", "r_split", "mu_base",
        "mu_scale", "mu_rate", "sigma_scale", "sigma_rate"
    };

    printf("\n  ── Kalman-Filtered Parameters (update #%d) ──\n\n", t->snap.n_updates);
    printf("  %-14s  %8s  %8s  %6s  %s\n", "Parameter", "Value", "±√P", "Gated", "Conv");
    printf("  ────────────────────────────────────────────────────\n");
    for (int i = 0; i < N_PARAMS; i++) {
        float std = sqrtf(fmaxf(t->P[i * N_PARAMS + i], 0.0f));
        const char* status = (t->converged[i] == 1) ? "✓" :
                             (t->converged[i] == -1) ? "lock" : "—";
        printf("  %-14s  %8.4f  %8.4f  %6.4f  %s\n",
               names[i], t->x[i], std, t->snap.theta[i], status);
    }
    printf("  ────────────────────────────────────────────────────\n");
    printf("  Derived:  σ_z=%.4f  σ_base=%.4f\n", t->snap.sigma_z, t->snap.sigma_base);
    printf("  At z̄=%.3f: μ=%.4f  σ_h=%.4f  θ=%.4f\n",
           t->snap.z_mean, t->snap.mu, t->snap.sigma_h, t->snap.theta_speed);
    printf("  SMC² last window: ESS=%.1f  accept=%.1f%%\n",
           t->snap.last_ess, t->snap.last_accept_rate * 100.0f);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * Convergence gating API
 *═══════════════════════════════════════════════════════════════════════════════*/

void param_tracker_set_free_mask(ParamTracker* t, const int* mask) {
    memcpy(t->free_mask, mask, N_PARAMS * sizeof(int));
    for (int i = 0; i < N_PARAMS; i++) {
        if (!mask[i]) {
            /* Locked: hold at prior default */
            t->gate_mode[i] = GATE_LOCKED;
            t->converged[i] = -1;
        } else if (t->gate_mode[i] == GATE_LOCKED) {
            /* Was locked, now free: restore appropriate default mode.
             * Phase 1 params (0-3) get KALMAN_MIN, Phase 2/3 (4-7) get RHAT_LATCH.
             * Caller can override with set_gate_mode() after this call. */
            t->gate_mode[i] = (i < 4) ? GATE_KALMAN_MIN : GATE_RHAT_LATCH;
            t->converged[i] = 0;  /* not yet converged in new mode */
        }
        /* If already free and converged (latch=1), don't touch it */
    }
}

void param_tracker_set_gate_mode(ParamTracker* t, int param_idx, int mode) {
    if (param_idx >= 0 && param_idx < N_PARAMS)
        t->gate_mode[param_idx] = mode;
}

void param_tracker_set_min_windows(ParamTracker* t, int n) {
    t->min_windows = n;
}

void param_tracker_set_prior_defaults(ParamTracker* t, const float* defaults) {
    memcpy(t->prior_default, defaults, N_PARAMS * sizeof(float));
}

void param_tracker_set_rhat_threshold(ParamTracker* t, float thresh) {
    t->rhat_threshold = thresh;
}

void param_tracker_get_conv_report(const ParamTracker* t,
                                    ConvergenceReport* rpt)
{
    float P_diag[N_PARAMS];
    for (int i = 0; i < N_PARAMS; i++)
        P_diag[i] = t->P[i * N_PARAMS + i];
    conv_diag_report(&t->conv_diag, t->x, P_diag,
                     t->free_mask, t->rhat_threshold, rpt);
}

void param_tracker_get_converged(const ParamTracker* t, int* out) {
    memcpy(out, t->converged, N_PARAMS * sizeof(int));
}

void param_tracker_get_kalman_x(const ParamTracker* t, float* x_out) {
    memcpy(x_out, t->x, N_PARAMS * sizeof(float));
}
