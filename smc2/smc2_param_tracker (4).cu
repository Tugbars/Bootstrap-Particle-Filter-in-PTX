/**
 * @file smc2_param_tracker.cu
 * @brief Kalman Parameter Tracker — implementation
 *
 * Sliding-window SMC² produces θ̂, Σ measurements.
 * Plain linear Kalman filter fuses them over time.
 * Curves evaluated at z̄ produce BPF-ready scalar parameters.
 *
 * The 8×8 Kalman math is host-side (too small for GPU), but this
 * file is .cu because it calls smc2_cuda_* functions directly.
 *
 * Build:
 *   nvcc -O2 -arch=sm_120 -o tracker smc2_param_tracker.cu smc2_rbpf_cuda.cu \
 *        -lcurand --expt-relaxed-constexpr
 */

#include "smc2_param_tracker.cuh"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/*═══════════════════════════════════════════════════════════════════════════════
 * Internal state
 *═══════════════════════════════════════════════════════════════════════════════*/

struct ParamTracker {
    /* ── Kalman filter state ── */
    float x[N_PARAMS];                      /* Filtered parameter estimate */
    float P[N_PARAMS * N_PARAMS];           /* Error covariance (8×8) */
    float Q[N_PARAMS * N_PARAMS];           /* Process noise (diagonal) */
    float P_floor[N_PARAMS];                /* Minimum P diagonal (prevents lockup) */
    float Q_boost[N_PARAMS];                /* Temporary Q multiplier (innovation gating) */
    int   initialized;                       /* 0 until first measurement */

    /* ── Innovation gating ── */
    float gate_threshold;                    /* σ-threshold for gating (default 3.0) */
    float gate_Q_multiplier;                 /* Q boost on gated param (default 100.0) */
    int   gate_cooldown[N_PARAMS];           /* Windows remaining with boosted Q */
    int   gate_cooldown_init;                /* Initial cooldown windows (default 3) */

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
 *
 * Solves column-by-column: L·L'·x_j = b_j for each column j of B.
 * Result stored in X (D×D, row-major).
 */
static void solve_spd_matrix(const float* L, const float* B, float* X) {
    float tmp[D];
    for (int j = 0; j < D; j++) {
        /* Extract column j of B */
        float col[D];
        for (int i = 0; i < D; i++) col[i] = B[i * D + j];
        /* Solve L·tmp = col */
        solve_lower(L, col, tmp);
        /* Solve L'·x = tmp */
        float x_col[D];
        solve_upper(L, tmp, x_col);
        /* Store column j of X */
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
 * Kalman Predict + Update (Robust version)
 *
 * Three robustness improvements over vanilla Kalman:
 *
 * 1. OVERLAP CORRECTION: When stride < window_size, consecutive Σ matrices
 *    share data and are correlated. We inflate R by (window_size / stride)
 *    to account for the reduced effective information per window.
 *
 * 2. INNOVATION GATING: If |innovation_i| > gate_threshold · √S_ii, parameter
 *    i experienced a structural break. We temporarily boost Q_i by 100× for
 *    a few windows, letting the Kalman "forget" and relearn quickly.
 *
 * 3. JOSEPH FORM: P = (I-K)·P̄·(I-K)ᵀ + K·R·Kᵀ is numerically stable,
 *    guaranteed symmetric positive semi-definite regardless of rounding.
 *
 * All matrices are 8×8. S⁻¹ computed via Cholesky.
 *═══════════════════════════════════════════════════════════════════════════════*/

static void kalman_update(ParamTracker* t, const float* z_meas, const float* R_raw) {
    float R[N_PARAMS * N_PARAMS];
    float P_bar[N_PARAMS * N_PARAMS];
    float Q_eff[N_PARAMS * N_PARAMS];
    float S[N_PARAMS * N_PARAMS];
    float L[N_PARAMS * N_PARAMS];
    float Kt[N_PARAMS * N_PARAMS];
    float K[N_PARAMS * N_PARAMS];

    /* ── Step 0: Overlap correction on R ──
     * When windows overlap, consecutive measurements share data.
     * Effective new information per window ∝ stride/window_size.
     * Inflate R to compensate for the correlation. */
    float overlap_factor = (t->stride < t->window_size)
        ? (float)t->window_size / (float)t->stride
        : 1.0f;
    for (int i = 0; i < N_PARAMS * N_PARAMS; i++)
        R[i] = R_raw[i] * overlap_factor;

    /* ── Step 1: Build effective Q with innovation gating boost ──
     * Q_eff = Q (base) + boost for gated parameters */
    memcpy(Q_eff, t->Q, N_PARAMS * N_PARAMS * sizeof(float));
    for (int i = 0; i < N_PARAMS; i++) {
        if (t->gate_cooldown[i] > 0) {
            Q_eff[i * N_PARAMS + i] *= t->gate_Q_multiplier;
            t->gate_cooldown[i]--;
        }
    }

    /* ── Step 2: Predict ──
     * P̄ = P + Q_eff */
    mat_add(P_bar, t->P, Q_eff);

    /* ── Step 3: Innovation + gating check ── */
    float innov[N_PARAMS];
    for (int i = 0; i < N_PARAMS; i++)
        innov[i] = z_meas[i] - t->x[i];

    /* S = P̄ + R (needed for gating check) */
    mat_add(S, P_bar, R);
    for (int i = 0; i < N_PARAMS; i++)
        S[i * N_PARAMS + i] += 1e-8f;

    /* Check each parameter for structural break */
    int any_gated = 0;
    for (int i = 0; i < N_PARAMS; i++) {
        float s_ii = S[i * N_PARAMS + i];
        float normalized = fabsf(innov[i]) / sqrtf(fmaxf(s_ii, 1e-12f));
        if (normalized > t->gate_threshold && t->gate_cooldown[i] == 0) {
            /* Structural break detected on parameter i */
            t->gate_cooldown[i] = t->gate_cooldown_init;
            any_gated = 1;
        }
    }

    /* If any parameter was just gated, re-do predict and S with boosted Q */
    if (any_gated) {
        memcpy(Q_eff, t->Q, N_PARAMS * N_PARAMS * sizeof(float));
        for (int i = 0; i < N_PARAMS; i++) {
            if (t->gate_cooldown[i] > 0)
                Q_eff[i * N_PARAMS + i] *= t->gate_Q_multiplier;
        }
        mat_add(P_bar, t->P, Q_eff);
        mat_add(S, P_bar, R);
        for (int i = 0; i < N_PARAMS; i++)
            S[i * N_PARAMS + i] += 1e-8f;
    }

    /* ── Step 4: Cholesky of S ── */
    if (!cholesky(S, L)) {
        fprintf(stderr, "param_tracker: Cholesky failed on S, using diagonal fallback\n");
        memset(L, 0, sizeof(L));
        for (int i = 0; i < N_PARAMS; i++)
            L[i * N_PARAMS + i] = sqrtf(fmaxf(S[i * N_PARAMS + i], 1e-8f));
    }

    /* ── Step 5: Kalman gain K = P̄ · S⁻¹ ── */
    solve_spd_matrix(L, P_bar, Kt);  /* Kt = S⁻¹ · P̄ */
    for (int i = 0; i < N_PARAMS; i++)
        for (int j = 0; j < N_PARAMS; j++)
            K[i * N_PARAMS + j] = Kt[j * N_PARAMS + i];

    /* ── Step 6: State update x = x̄ + K · innovation ── */
    for (int i = 0; i < N_PARAMS; i++) {
        float sum = 0.0f;
        for (int j = 0; j < N_PARAMS; j++)
            sum += K[i * N_PARAMS + j] * innov[j];
        t->x[i] += sum;
    }

    /* ── Step 7: Joseph form P update ──
     * P = (I-K)·P̄·(I-K)ᵀ + K·R·Kᵀ
     * Guaranteed symmetric positive semi-definite. */
    float IminusK[N_PARAMS * N_PARAMS];
    float tmp1[N_PARAMS * N_PARAMS];    /* (I-K)·P̄ */
    float tmp2[N_PARAMS * N_PARAMS];    /* (I-K)ᵀ */
    float term1[N_PARAMS * N_PARAMS];   /* (I-K)·P̄·(I-K)ᵀ */
    float KR[N_PARAMS * N_PARAMS];      /* K·R */
    float term2[N_PARAMS * N_PARAMS];   /* K·R·Kᵀ */

    mat_eye_minus(IminusK, K);
    mat_mul(tmp1, IminusK, P_bar);      /* (I-K)·P̄ */

    /* Transpose (I-K) */
    for (int i = 0; i < N_PARAMS; i++)
        for (int j = 0; j < N_PARAMS; j++)
            tmp2[i * N_PARAMS + j] = IminusK[j * N_PARAMS + i];

    mat_mul(term1, tmp1, tmp2);         /* (I-K)·P̄·(I-K)ᵀ */

    mat_mul(KR, K, R);                  /* K·R */
    /* Kᵀ */
    float Ktrans[N_PARAMS * N_PARAMS];
    for (int i = 0; i < N_PARAMS; i++)
        for (int j = 0; j < N_PARAMS; j++)
            Ktrans[i * N_PARAMS + j] = K[j * N_PARAMS + i];
    mat_mul(term2, KR, Ktrans);         /* K·R·Kᵀ */

    /* P = term1 + term2, symmetrize */
    for (int i = 0; i < N_PARAMS; i++)
        for (int j = 0; j < N_PARAMS; j++)
            t->P[i * N_PARAMS + j] = 0.5f * (
                term1[i * N_PARAMS + j] + term1[j * N_PARAMS + i] +
                term2[i * N_PARAMS + j] + term2[j * N_PARAMS + i]);
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
        t->P[i * N_PARAMS + i] = 10.0f;  /* Wide initial uncertainty */

    /* Default drift rates (Q diagonal):
     * These are per-window drift variances.
     * Tuned for stride=500 ticks at ~1 tick/sec ≈ 8 min between windows.
     *
     * Parameter     Timescale     Q (variance/window)
     * ─────────────────────────────────────────────
     * μ_base        days          1e-3   (fast tracking)
     * ρ             weeks         1e-5   (medium)
     * σ_total       weeks         1e-5
     * r_split       weeks         1e-5
     * μ_scale       months        1e-7   (slow, structural)
     * μ_rate        months        1e-7
     * σ_scale       months        1e-7
     * σ_rate        months        1e-7
     */
    memset(t->Q, 0, sizeof(t->Q));
    t->Q[0 * N_PARAMS + 0] = 1e-5f;   /* rho */
    t->Q[1 * N_PARAMS + 1] = 1e-5f;   /* sigma_total */
    t->Q[2 * N_PARAMS + 2] = 1e-5f;   /* r_split */
    t->Q[3 * N_PARAMS + 3] = 1e-3f;   /* mu_base — fastest */
    t->Q[4 * N_PARAMS + 4] = 1e-7f;   /* mu_scale */
    t->Q[5 * N_PARAMS + 5] = 1e-7f;   /* mu_rate */
    t->Q[6 * N_PARAMS + 6] = 1e-7f;   /* sigma_scale */
    t->Q[7 * N_PARAMS + 7] = 1e-7f;   /* sigma_rate */

    /* Default P floor: prevents lockup from confident-but-wrong measurements.
     * Floor = fraction of prior variance — ensures Kalman never becomes
     * more than ~95% confident, always accepts new evidence.
     *
     * Rule: P_floor ≈ 5-10% of typical posterior variance from one window.
     * This keeps minimum gain K ≈ P_floor/(P_floor + Σ) above ~5%.
     */
    t->P_floor[0] = 1e-4f;   /* rho:         √P_min ≈ 0.010 */
    t->P_floor[1] = 1e-4f;   /* sigma_total: √P_min ≈ 0.010 */
    t->P_floor[2] = 1e-3f;   /* r_split:     √P_min ≈ 0.032 */
    t->P_floor[3] = 1e-2f;   /* mu_base:     √P_min ≈ 0.100 — must stay very responsive */
    t->P_floor[4] = 1e-3f;   /* mu_scale:    √P_min ≈ 0.032 */
    t->P_floor[5] = 1e-3f;   /* mu_rate:     √P_min ≈ 0.032 */
    t->P_floor[6] = 1e-4f;   /* sigma_scale: √P_min ≈ 0.010 */
    t->P_floor[7] = 1e-3f;   /* sigma_rate:  √P_min ≈ 0.032 */

    /* Innovation gating defaults */
    t->gate_threshold = 3.0f;       /* 3σ triggers structural break */
    t->gate_Q_multiplier = 100.0f;  /* Boost Q by 100× on break */
    t->gate_cooldown_init = 3;      /* Boosted for 3 windows after detection */
    memset(t->gate_cooldown, 0, sizeof(t->gate_cooldown));
    memset(t->Q_boost, 0, sizeof(t->Q_boost));

    t->ticks_since_window = 0;
    memset(&t->snap, 0, sizeof(t->snap));

    return t;
}

void param_tracker_destroy(ParamTracker* t) {
    if (!t) return;
    if (t->smc2) smc2_cuda_free(t->smc2);
    free(t->y_buf);
    free(t);
}

void param_tracker_feed(ParamTracker* t, float y_obs) {
    /* Write to circular buffer */
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

    /* Reset and run SMC² on this window */
    smc2_cuda_init_from_prior(t->smc2);

    for (int i = 0; i < W; i++)
        smc2_cuda_update(t->smc2, window[i]);

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
        /* Add small regularization to initial P */
        for (int i = 0; i < N_PARAMS; i++)
            t->P[i * N_PARAMS + i] += 1e-6f;
        t->initialized = 1;
    } else {
        kalman_update(t, z_meas, R);
    }

    /* Capture gating state for diagnostics */
    float overlap_factor = (t->stride < t->window_size)
        ? (float)t->window_size / (float)t->stride : 1.0f;
    for (int i = 0; i < N_PARAMS; i++)
        t->snap.gated[i] = (t->gate_cooldown[i] > 0) ? 1 : 0;
    t->snap.overlap_factor = overlap_factor;

    /* Enforce P floor: prevent lockup from confident-but-wrong measurements */
    for (int i = 0; i < N_PARAMS; i++) {
        if (t->P[i * N_PARAMS + i] < t->P_floor[i])
            t->P[i * N_PARAMS + i] = t->P_floor[i];
    }

    /* Derive physical parameters from filtered state */
    float sigma_total = t->x[1];
    float r_split = t->x[2];
    float mu_base = t->x[3];
    float mu_scale = t->x[4];
    float mu_rate = t->x[5];
    float sigma_scale = t->x[6];
    float sigma_rate = t->x[7];

    float sigma_z = r_split * sigma_total;
    float sigma_base = sqrtf(fmaxf(1.0f - r_split * r_split, 1e-6f)) * sigma_total;

    /* Evaluate curves at z̄ */
    float mu_at_z = eval_curve_host(mu_base, mu_scale, mu_rate, z_mean);
    float sigma_h_at_z = eval_curve_host(sigma_base, sigma_scale, sigma_rate, z_mean);
    float theta_at_z = eval_curve_host(t->theta_base, t->theta_scale, t->theta_rate, z_mean);

    /* Update snapshot */
    memcpy(t->snap.theta, t->x, N_PARAMS * sizeof(float));
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

void param_tracker_set_gating(ParamTracker* t, float threshold,
                               float q_mult, int cooldown) {
    t->gate_threshold = threshold;
    t->gate_Q_multiplier = q_mult;
    t->gate_cooldown_init = cooldown;
}

SMC2StateCUDA* param_tracker_get_smc2(ParamTracker* t) {
    return t->smc2;
}

void param_tracker_get_P(const ParamTracker* t, float* P_out) {
    memcpy(P_out, t->P, N_PARAMS * N_PARAMS * sizeof(float));
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
    printf("  %-14s  %8s  %8s  %4s\n", "Parameter", "Value", "±√P", "Gate");
    printf("  ─────────────────────────────────────────────\n");
    for (int i = 0; i < N_PARAMS; i++) {
        float std = sqrtf(fmaxf(t->P[i * N_PARAMS + i], 0.0f));
        const char* gate = (t->gate_cooldown[i] > 0) ? " ⚡" : "";
        printf("  %-14s  %8.4f  %8.4f  %s\n", names[i], t->x[i], std, gate);
    }
    printf("  ─────────────────────────────────────────────\n");
    printf("  Derived:  σ_z=%.4f  σ_base=%.4f\n", t->snap.sigma_z, t->snap.sigma_base);
    printf("  At z̄=%.3f: μ=%.4f  σ_h=%.4f  θ=%.4f\n",
           t->snap.z_mean, t->snap.mu, t->snap.sigma_h, t->snap.theta_speed);
    printf("  SMC² last window: ESS=%.1f  accept=%.1f%%  overlap=%.1fx\n",
           t->snap.last_ess, t->snap.last_accept_rate * 100.0f, t->snap.overlap_factor);
}
