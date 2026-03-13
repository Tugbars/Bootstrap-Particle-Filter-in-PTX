/**
 * @file smc2_param_tracker.cu
 * @brief Kalman Parameter Tracker + Phased Learning - Merged Implementation
 *
 * param_tracker_run_window() flow:
 *   1. Extract window from circular buffer
 *   2. Run SMC2 on window (mask already set for current phase)
 *   3. Extract posterior, Kalman update
 *   4. Convergence gating
 *   5. Z-range observation + phase transition check
 *   6. Build gated snapshot
 *
 * Phased learning is fully internal. Phase transitions update gate_mode[]
 * and SMC2 fixed_mask. Save-before-lock uses Kalman x[].
 */

#include "smc2_param_tracker.cuh"
#include "smc2_pipeline.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/*======================================================================
 * Z-Range Tracker (internal)
 *======================================================================*/

typedef struct {
    float z_min_window, z_max_window, z_mean_window;
    float z_min_seen, z_max_seen, z_ema;
    int   n_windows;
    int   high_z_streak, wide_range_streak;
    int   calm_streak, narrow_range_streak;
} ZRangeTracker;

/*======================================================================
 * Internal state
 *======================================================================*/

struct ParamTracker {
    float x[N_PARAMS];
    float P[N_PARAMS * N_PARAMS];
    float Q[N_PARAMS * N_PARAMS];
    float P_floor[N_PARAMS];
    int   initialized;

    float* y_buf;
    int    buf_capacity, buf_head, buf_count;

    int    window_size, stride, ticks_since_window;

    SMC2StateCUDA* smc2;
    int    N_theta, N_inner;

    float  theta_base, theta_scale, theta_rate;

    ConvergenceDiag conv_diag;
    float  prior_default[N_PARAMS];
    int    free_mask[N_PARAMS];
    float  rhat_threshold;
    int    conv_diag_M;

    int    gate_mode[N_PARAMS];
    int    converged[N_PARAMS];
    int    min_windows;
    int    n_windows_completed;

    PhasedConfig    phased_config;
    int             phase;
    ZRangeTracker   z_tracker;
    PhaseTransition phase_history[MAX_PHASE_TRANSITIONS];
    int             n_phase_transitions;
    int             phase2_entered_at, phase3_entered_at;

    ParamSnapshot snap;
};

/*======================================================================
 * 8x8 Linear Algebra (host-side)
 *======================================================================*/

#define D N_PARAMS

static void mat_add(float* C, const float* A, const float* B) {
    for (int i = 0; i < D*D; i++) C[i] = A[i] + B[i];
}

static int cholesky_decomp(const float* A, float* L) {
    memset(L, 0, D*D*sizeof(float));
    for (int i = 0; i < D; i++) {
        for (int j = 0; j <= i; j++) {
            float s = 0;
            for (int k = 0; k < j; k++) s += L[i*D+k]*L[j*D+k];
            if (i == j) {
                float v = A[i*D+i] - s;
                if (v <= 0) return 0;
                L[i*D+j] = sqrtf(v);
            } else {
                float d = L[j*D+j];
                L[i*D+j] = (d > 1e-12f) ? (A[i*D+j] - s)/d : 0;
            }
        }
    }
    return 1;
}

static void solve_lower(const float* L, const float* b, float* x) {
    for (int i = 0; i < D; i++) {
        float s = b[i];
        for (int j = 0; j < i; j++) s -= L[i*D+j]*x[j];
        x[i] = s / L[i*D+i];
    }
}

static void solve_upper(const float* L, const float* b, float* x) {
    for (int i = D-1; i >= 0; i--) {
        float s = b[i];
        for (int j = i+1; j < D; j++) s -= L[j*D+i]*x[j];
        x[i] = s / L[i*D+i];
    }
}

static void solve_spd(const float* L, const float* B, float* X) {
    float tmp[D];
    for (int j = 0; j < D; j++) {
        float col[D]; for (int i = 0; i < D; i++) col[i] = B[i*D+j];
        solve_lower(L, col, tmp);
        float xc[D]; solve_upper(L, tmp, xc);
        for (int i = 0; i < D; i++) X[i*D+j] = xc[i];
    }
}

static void mat_mul(float* C, const float* A, const float* B) {
    for (int i = 0; i < D; i++)
        for (int j = 0; j < D; j++) {
            float s = 0;
            for (int k = 0; k < D; k++) s += A[i*D+k]*B[k*D+j];
            C[i*D+j] = s;
        }
}

static void mat_eye_minus(float* C, const float* A) {
    for (int i = 0; i < D; i++)
        for (int j = 0; j < D; j++)
            C[i*D+j] = ((i==j)?1.0f:0.0f) - A[i*D+j];
}

#undef D

/*======================================================================
 * Kalman Update
 *======================================================================*/

static void kalman_update(ParamTracker* t, const float* z_meas, const float* R) {
    float Pb[N_PARAMS*N_PARAMS], S[N_PARAMS*N_PARAMS], L[N_PARAMS*N_PARAMS];
    float Kt[N_PARAMS*N_PARAMS], K[N_PARAMS*N_PARAMS];
    float IK[N_PARAMS*N_PARAMS], Pn[N_PARAMS*N_PARAMS];

    mat_add(Pb, t->P, t->Q);
    mat_add(S, Pb, R);
    for (int i = 0; i < N_PARAMS; i++) S[i*N_PARAMS+i] += 1e-8f;

    if (!cholesky_decomp(S, L)) {
        fprintf(stderr, "param_tracker: Cholesky failed, diagonal fallback\n");
        memset(L, 0, sizeof(L));
        for (int i = 0; i < N_PARAMS; i++)
            L[i*N_PARAMS+i] = sqrtf(fmaxf(S[i*N_PARAMS+i], 1e-8f));
    }

    solve_spd(L, Pb, Kt);
    for (int i = 0; i < N_PARAMS; i++)
        for (int j = 0; j < N_PARAMS; j++)
            K[i*N_PARAMS+j] = Kt[j*N_PARAMS+i];

    float innov[N_PARAMS];
    for (int i = 0; i < N_PARAMS; i++) innov[i] = z_meas[i] - t->x[i];
    for (int i = 0; i < N_PARAMS; i++) {
        float s = 0;
        for (int j = 0; j < N_PARAMS; j++) s += K[i*N_PARAMS+j]*innov[j];
        t->x[i] += s;
    }

    mat_eye_minus(IK, K);
    mat_mul(Pn, IK, Pb);
    for (int i = 0; i < N_PARAMS; i++)
        for (int j = 0; j < N_PARAMS; j++)
            t->P[i*N_PARAMS+j] = 0.5f*(Pn[i*N_PARAMS+j]+Pn[j*N_PARAMS+i]);
}

/*======================================================================
 * Phased Learning - Internal Helpers
 *======================================================================*/

static void phased_apply_mask(ParamTracker* t) {
    uint8_t mask[N_PARAMS];
    float values[N_PARAMS];
    memset(mask, 0, sizeof(mask));
    memset(values, 0, sizeof(values));
    PhasedConfig* pc = &t->phased_config;

    switch (t->phase) {
    case PHASE_1_FLOORS:
        mask[P_MU_SCALE]=1; values[P_MU_SCALE]=pc->fixed_mu_scale;
        mask[P_MU_RATE]=1;  values[P_MU_RATE]=pc->fixed_mu_rate;
        mask[P_SIGMA_SCALE]=1; values[P_SIGMA_SCALE]=pc->fixed_sigma_scale;
        mask[P_SIGMA_RATE]=1;  values[P_SIGMA_RATE]=pc->fixed_sigma_rate;
        break;
    case PHASE_2_CEILINGS:
        mask[P_MU_RATE]=1;  values[P_MU_RATE]=pc->fixed_mu_rate;
        mask[P_SIGMA_RATE]=1; values[P_SIGMA_RATE]=pc->fixed_sigma_rate;
        break;
    case PHASE_3_RATES:
        break;
    }

    smc2_cuda_set_fixed_params(t->smc2, mask, values);

    for (int i = 0; i < N_PARAMS; i++) {
        if (mask[i]) {
            t->gate_mode[i] = GATE_LOCKED;
            t->converged[i] = -1;
            t->free_mask[i] = 0;
        } else {
            if (t->gate_mode[i] == GATE_LOCKED) {
                t->gate_mode[i] = (i < 4) ? GATE_KALMAN_MIN : GATE_RHAT_LATCH;
                t->converged[i] = 0;
            }
            t->free_mask[i] = 1;
        }
    }
}

static void phased_record(ParamTracker* t, int from, int to) {
    ZRangeTracker* zt = &t->z_tracker;
    if (t->n_phase_transitions < MAX_PHASE_TRANSITIONS) {
        PhaseTransition* tr = &t->phase_history[t->n_phase_transitions++];
        tr->window = zt->n_windows;
        tr->from = from; tr->to = to;
        tr->z_min = zt->z_min_window;
        tr->z_max = zt->z_max_window;
        tr->z_range = zt->z_max_window - zt->z_min_window;
    }
    const char* dir = (to > from) ? "UNLOCK" : "LOCK  ";
    printf("[PHASED] %s Phase %d -> %d at window %d (z: [%.2f, %.2f] range: %.2f)\n",
           dir, from, to, zt->n_windows,
           zt->z_min_window, zt->z_max_window,
           zt->z_max_window - zt->z_min_window);
}

static void phased_save_ceilings(ParamTracker* t) {
    t->phased_config.fixed_mu_scale    = t->x[P_MU_SCALE];
    t->phased_config.fixed_sigma_scale = t->x[P_SIGMA_SCALE];
    t->phased_config.learned_ceilings  = 1;
    t->prior_default[P_MU_SCALE]    = t->x[P_MU_SCALE];
    t->prior_default[P_SIGMA_SCALE] = t->x[P_SIGMA_SCALE];
    printf("[PHASED] Saving ceilings from Kalman: mu_scale=%.4f sigma_scale=%.4f\n",
           t->x[P_MU_SCALE], t->x[P_SIGMA_SCALE]);
}

static void phased_save_rates(ParamTracker* t) {
    t->phased_config.fixed_mu_rate    = t->x[P_MU_RATE];
    t->phased_config.fixed_sigma_rate = t->x[P_SIGMA_RATE];
    t->phased_config.learned_rates    = 1;
    t->prior_default[P_MU_RATE]    = t->x[P_MU_RATE];
    t->prior_default[P_SIGMA_RATE] = t->x[P_SIGMA_RATE];
    printf("[PHASED] Saving rates from Kalman: mu_rate=%.4f sigma_rate=%.4f\n",
           t->x[P_MU_RATE], t->x[P_SIGMA_RATE]);
}

static void phased_observe_z(ParamTracker* t, float zm, float zn, float zx) {
    ZRangeTracker* zt = &t->z_tracker;
    PhasedConfig* pc = &t->phased_config;

    zt->z_min_window = zn; zt->z_max_window = zx; zt->z_mean_window = zm;
    if (zn < zt->z_min_seen) zt->z_min_seen = zn;
    if (zx > zt->z_max_seen) zt->z_max_seen = zx;
    zt->z_ema = (zt->n_windows == 0) ? zm : zt->z_ema + pc->z_ema_alpha*(zm - zt->z_ema);

    float zr = zx - zn;

    if (zx > pc->ceiling_z_threshold) { zt->high_z_streak++; zt->calm_streak = 0; }
    else { zt->high_z_streak = 0; zt->calm_streak++; }

    if (zr > pc->rate_z_range_threshold) { zt->wide_range_streak++; zt->narrow_range_streak = 0; }
    else { zt->wide_range_streak = 0; zt->narrow_range_streak++; }

    zt->n_windows++;
}

static int phased_check(ParamTracker* t) {
    if (!t->phased_config.enabled) return 0;
    int prev = t->phase;
    ZRangeTracker* zt = &t->z_tracker;
    PhasedConfig* pc = &t->phased_config;

    switch (t->phase) {
    case PHASE_1_FLOORS:
        if (zt->high_z_streak >= pc->ceiling_z_sustained) {
            t->phase = PHASE_2_CEILINGS;
            if (t->phase2_entered_at < 0) t->phase2_entered_at = zt->n_windows;
            phased_record(t, prev, t->phase);
            phased_apply_mask(t);
        }
        break;
    case PHASE_2_CEILINGS:
        if (zt->wide_range_streak >= pc->rate_range_sustained) {
            t->phase = PHASE_3_RATES;
            if (t->phase3_entered_at < 0) t->phase3_entered_at = zt->n_windows;
            phased_record(t, prev, t->phase);
            phased_apply_mask(t);
        } else if (zt->calm_streak >= pc->ceiling_z_sustained) {
            phased_save_ceilings(t);
            t->phase = PHASE_1_FLOORS;
            phased_record(t, prev, t->phase);
            phased_apply_mask(t);
        }
        break;
    case PHASE_3_RATES:
        if (zt->narrow_range_streak >= pc->rate_range_sustained) {
            phased_save_rates(t);
            t->phase = PHASE_2_CEILINGS;
            phased_record(t, prev, t->phase);
            phased_apply_mask(t);
            if (zt->calm_streak >= pc->ceiling_z_sustained) {
                int mid = t->phase;
                phased_save_ceilings(t);
                t->phase = PHASE_1_FLOORS;
                phased_record(t, mid, t->phase);
                phased_apply_mask(t);
            }
        }
        break;
    }
    return (t->phase != prev);
}

/*======================================================================
 * Public API - Lifecycle
 *======================================================================*/

ParamTracker* param_tracker_create(int window_size, int stride,
                                    int N_theta, int N_inner) {
    ParamTracker* t = (ParamTracker*)calloc(1, sizeof(ParamTracker));
    if (!t) return NULL;

    t->window_size = window_size;
    t->stride = stride;
    t->N_theta = N_theta;
    t->N_inner = N_inner;

    t->buf_capacity = window_size + stride;
    t->y_buf = (float*)calloc(t->buf_capacity, sizeof(float));
    if (!t->y_buf) { free(t); return NULL; }

    t->smc2 = smc2_cuda_alloc(N_theta, N_inner);
    if (!t->smc2) { free(t->y_buf); free(t); return NULL; }
    smc2_cuda_set_fixed_lag(t->smc2, 200);

    t->theta_base = 0.02f; t->theta_scale = 0.08f; t->theta_rate = 1.5f;

    t->initialized = 0;
    memset(t->x, 0, sizeof(t->x));
    memset(t->P, 0, sizeof(t->P));
    for (int i = 0; i < N_PARAMS; i++) t->P[i*N_PARAMS+i] = 10.0f;

    memset(t->Q, 0, sizeof(t->Q));
    t->Q[0*N_PARAMS+0]=1e-5f; t->Q[1*N_PARAMS+1]=1e-5f;
    t->Q[2*N_PARAMS+2]=1e-5f; t->Q[3*N_PARAMS+3]=1e-3f;
    t->Q[4*N_PARAMS+4]=1e-7f; t->Q[5*N_PARAMS+5]=1e-7f;
    t->Q[6*N_PARAMS+6]=1e-7f; t->Q[7*N_PARAMS+7]=1e-7f;

    t->P_floor[0]=1e-4f; t->P_floor[1]=1e-4f; t->P_floor[2]=1e-3f;
    t->P_floor[3]=1e-2f; t->P_floor[4]=1e-3f; t->P_floor[5]=1e-3f;
    t->P_floor[6]=1e-4f; t->P_floor[7]=1e-3f;

    t->ticks_since_window = 0;
    memset(&t->snap, 0, sizeof(t->snap));

    t->conv_diag_M = 8;
    conv_diag_init(&t->conv_diag, t->conv_diag_M);
    t->rhat_threshold = 1.5f;
    t->min_windows = 2;
    t->n_windows_completed = 0;

    sv_default_prior_fallback(t->prior_default);

    t->gate_mode[0]=GATE_KALMAN_MIN; t->gate_mode[1]=GATE_KALMAN_MIN;
    t->gate_mode[2]=GATE_KALMAN_MIN; t->gate_mode[3]=GATE_KALMAN_MIN;
    t->gate_mode[4]=GATE_RHAT_LATCH; t->gate_mode[5]=GATE_RHAT_LATCH;
    t->gate_mode[6]=GATE_RHAT_LATCH; t->gate_mode[7]=GATE_RHAT_LATCH;

    for (int i = 0; i < N_PARAMS; i++) t->free_mask[i] = 1;
    for (int i = 0; i < N_PARAMS; i++) t->converged[i] = 0;

    t->phased_config = phased_default_config();
    t->phase = PHASE_1_FLOORS;
    memset(&t->z_tracker, 0, sizeof(t->z_tracker));
    t->z_tracker.z_min_seen = 1e6f;
    t->z_tracker.z_max_seen = -1e6f;
    t->n_phase_transitions = 0;
    t->phase2_entered_at = -1;
    t->phase3_entered_at = -1;

    phased_apply_mask(t);

    return t;
}

void param_tracker_destroy(ParamTracker* t) {
    if (!t) return;
    if (t->smc2) smc2_cuda_free(t->smc2);
    free(t->y_buf);
    free(t);
}

/*======================================================================
 * Public API - Feeding + Window
 *======================================================================*/

void param_tracker_feed(ParamTracker* t, float y_obs) {
    t->y_buf[t->buf_head] = y_obs;
    t->buf_head = (t->buf_head + 1) % t->buf_capacity;
    t->buf_count++;
    t->ticks_since_window++;
}

int param_tracker_window_ready(const ParamTracker* t) {
    return (t->ticks_since_window >= t->stride && t->buf_count >= t->window_size);
}

void param_tracker_run_window(ParamTracker* t) {
    int W = t->window_size;

    /* 1. Extract window */
    float* window = (float*)malloc(W * sizeof(float));
    int start = (t->buf_head - W + t->buf_capacity) % t->buf_capacity;
    for (int i = 0; i < W; i++)
        window[i] = t->y_buf[(start + i) % t->buf_capacity];

    /* 2. Cold-start SMC2 (mask set for current phase) */
    smc2_cuda_init_from_prior(t->smc2);

    /* 3. Run SMC2 */
    smc2_cuda_update_batch(t->smc2, window, W);

    /* 4. Extract posterior */
    float z_meas[N_PARAMS], R[N_PARAMS * N_PARAMS];
    smc2_cuda_get_theta_cov(t->smc2, z_meas, R);
    float z_mean = smc2_cuda_get_z_mean(t->smc2);
    SMC2Diagnostics diag;
    smc2_cuda_get_diagnostics(t->smc2, &diag);

    /* 5. Kalman update */
    if (!t->initialized) {
        memcpy(t->x, z_meas, N_PARAMS * sizeof(float));
        memcpy(t->P, R, N_PARAMS * N_PARAMS * sizeof(float));
        for (int i = 0; i < N_PARAMS; i++) t->P[i*N_PARAMS+i] += 1e-6f;
        t->initialized = 1;

        float sd[N_PARAMS]; for (int i=0;i<N_PARAMS;i++) sd[i]=R[i*N_PARAMS+i];
        float pt=0; for (int i=0;i<N_PARAMS;i++) pt+=t->P[i*N_PARAMS+i];
        conv_diag_push(&t->conv_diag, z_meas, sd, 0.0f, pt);
    } else {
        float xp[N_PARAMS]; memcpy(xp, t->x, sizeof(xp));
        float Pbd[N_PARAMS];
        for (int i=0;i<N_PARAMS;i++) Pbd[i]=t->P[i*N_PARAMS+i]+t->Q[i*N_PARAMS+i];

        kalman_update(t, z_meas, R);

        float sd[N_PARAMS]; for (int i=0;i<N_PARAMS;i++) sd[i]=R[i*N_PARAMS+i];
        float nu[N_PARAMS], Sd[N_PARAMS];
        for (int i=0;i<N_PARAMS;i++) { nu[i]=z_meas[i]-xp[i]; Sd[i]=Pbd[i]+sd[i]; }
        float d2 = conv_diag_mahal_diag(nu, Sd, N_PARAMS);
        float pt=0; for (int i=0;i<N_PARAMS;i++) pt+=t->P[i*N_PARAMS+i];
        conv_diag_push(&t->conv_diag, z_meas, sd, d2, pt);
    }
    t->n_windows_completed++;

    /* Enforce P floor */
    for (int i=0;i<N_PARAMS;i++)
        if (t->P[i*N_PARAMS+i] < t->P_floor[i])
            t->P[i*N_PARAMS+i] = t->P_floor[i];

    /* 6. Convergence gating */
    ConvergenceReport rpt;
    { float pd[N_PARAMS]; for(int i=0;i<N_PARAMS;i++) pd[i]=t->P[i*N_PARAMS+i];
      conv_diag_report(&t->conv_diag,t->x,pd,t->free_mask,t->rhat_threshold,&rpt); }

    for (int i=0;i<N_PARAMS;i++) {
        switch(t->gate_mode[i]) {
        case GATE_KALMAN_MIN:
            if (t->converged[i]!=1 && t->n_windows_completed>=t->min_windows)
                t->converged[i]=1;
            break;
        case GATE_RHAT_LATCH:
            if (t->converged[i]!=1 && rpt.ready && rpt.converged[i]==1)
                t->converged[i]=1;
            break;
        case GATE_LOCKED: default:
            t->converged[i]=-1;
            break;
        }
    }

    /* 7. Z-range + phase transitions */
    { float zm,zn,zx;
      smc2_cuda_get_z_range_robust(t->smc2, &zm, &zn, &zx);
      phased_observe_z(t, zm, zn, zx);
      phased_check(t); }

    /* 8. Build gated snapshot */
    float gated[N_PARAMS];
    for (int i=0;i<N_PARAMS;i++)
        gated[i] = (t->converged[i]==1) ? t->x[i] : t->prior_default[i];

    float st=gated[1], rs=gated[2];
    float sz=rs*st, sb=sqrtf(fmaxf(1-rs*rs,1e-6f))*st;

    float mu_z  = eval_curve_host(gated[3], gated[4], gated[5], z_mean);
    float sh_z  = eval_curve_host(sb, gated[6], gated[7], z_mean);
    float th_z  = eval_curve_host(t->theta_base, t->theta_scale, t->theta_rate, z_mean);

    memcpy(t->snap.theta, gated, sizeof(gated));
    for (int i=0;i<N_PARAMS;i++) t->snap.P_diag[i]=t->P[i*N_PARAMS+i];
    t->snap.sigma_z=sz; t->snap.sigma_base=sb; t->snap.z_mean=z_mean;
    t->snap.mu=mu_z; t->snap.sigma_h=sh_z; t->snap.theta_speed=th_z;
    t->snap.n_updates++;
    t->snap.last_accept_rate=diag.accept_rate;
    t->snap.last_ess=diag.outer_ess;

    t->ticks_since_window = 0;
    free(window);
}

/*======================================================================
 * Public API - Output + Config
 *======================================================================*/

void param_tracker_get_snapshot(const ParamTracker* t, ParamSnapshot* s) { *s = t->snap; }

void param_tracker_set_drift(ParamTracker* t, const DriftConfig* d) {
    memset(t->Q, 0, sizeof(t->Q));
    t->Q[0*N_PARAMS+0]=d->q_rho;          t->Q[1*N_PARAMS+1]=d->q_sigma_total;
    t->Q[2*N_PARAMS+2]=d->q_r_split;      t->Q[3*N_PARAMS+3]=d->q_mu_base;
    t->Q[4*N_PARAMS+4]=d->q_mu_scale;     t->Q[5*N_PARAMS+5]=d->q_mu_rate;
    t->Q[6*N_PARAMS+6]=d->q_sigma_scale;  t->Q[7*N_PARAMS+7]=d->q_sigma_rate;
}

void param_tracker_set_theta_curve(ParamTracker* t, float b, float s, float r) {
    t->theta_base=b; t->theta_scale=s; t->theta_rate=r;
}

void param_tracker_set_P_floor(ParamTracker* t, const float* pf) {
    memcpy(t->P_floor, pf, N_PARAMS*sizeof(float));
}

void param_tracker_set_prior_defaults(ParamTracker* t, const float* d) {
    memcpy(t->prior_default, d, N_PARAMS*sizeof(float));
}

SMC2StateCUDA* param_tracker_get_smc2(ParamTracker* t) { return t->smc2; }

void param_tracker_get_P(const ParamTracker* t, float* P) {
    memcpy(P, t->P, N_PARAMS*N_PARAMS*sizeof(float));
}

void param_tracker_get_kalman_x(const ParamTracker* t, float* x) {
    memcpy(x, t->x, N_PARAMS*sizeof(float));
}

void param_tracker_get_converged(const ParamTracker* t, int* o) {
    memcpy(o, t->converged, N_PARAMS*sizeof(int));
}

void param_tracker_force_cold(ParamTracker* t) {
    t->initialized=0; t->snap.n_updates=0;
}

void param_tracker_set_free_mask(ParamTracker* t, const int* mask) {
    memcpy(t->free_mask, mask, N_PARAMS*sizeof(int));
    for (int i=0;i<N_PARAMS;i++) {
        if (!mask[i]) { t->gate_mode[i]=GATE_LOCKED; t->converged[i]=-1; }
        else if (t->gate_mode[i]==GATE_LOCKED) {
            t->gate_mode[i]=(i<4)?GATE_KALMAN_MIN:GATE_RHAT_LATCH;
            t->converged[i]=0;
        }
    }
}

void param_tracker_set_gate_mode(ParamTracker* t, int p, int m) {
    if (p>=0&&p<N_PARAMS) t->gate_mode[p]=m;
}

void param_tracker_set_min_windows(ParamTracker* t, int n) { t->min_windows=n; }

void param_tracker_set_rhat_threshold(ParamTracker* t, float th) { t->rhat_threshold=th; }

void param_tracker_get_conv_report(const ParamTracker* t, ConvergenceReport* r) {
    float pd[N_PARAMS]; for(int i=0;i<N_PARAMS;i++) pd[i]=t->P[i*N_PARAMS+i];
    conv_diag_report(&t->conv_diag,t->x,pd,t->free_mask,t->rhat_threshold,r);
}

/* Phased learning config */

void param_tracker_set_phased_config(ParamTracker* t, const PhasedConfig* pc) {
    t->phased_config=*pc; phased_apply_mask(t);
}

void param_tracker_set_phased_fixed_rates(ParamTracker* t, float mr, float sr) {
    t->phased_config.fixed_mu_rate=mr; t->phased_config.fixed_sigma_rate=sr;
    t->phased_config.learned_rates=1;
    t->prior_default[P_MU_RATE]=mr; t->prior_default[P_SIGMA_RATE]=sr;
    if (t->phase<PHASE_3_RATES) phased_apply_mask(t);
}

void param_tracker_set_phased_fixed_ceilings(ParamTracker* t, float ms, float ss) {
    t->phased_config.fixed_mu_scale=ms; t->phased_config.fixed_sigma_scale=ss;
    t->phased_config.learned_ceilings=1;
    t->prior_default[P_MU_SCALE]=ms; t->prior_default[P_SIGMA_SCALE]=ss;
    if (t->phase<PHASE_2_CEILINGS) phased_apply_mask(t);
}

/* Phased learning queries */

int param_tracker_get_phase(const ParamTracker* t) { return t->phase; }
int param_tracker_get_n_transitions(const ParamTracker* t) { return t->n_phase_transitions; }

void param_tracker_get_z_range(const ParamTracker* t,
                                float* zm, float* zn, float* zx) {
    if (zm) *zm=t->z_tracker.z_mean_window;
    if (zn) *zn=t->z_tracker.z_min_window;
    if (zx) *zx=t->z_tracker.z_max_window;
}

/*======================================================================
 * Diagnostics Printing
 *======================================================================*/

void param_tracker_print(const ParamTracker* t) {
    static const char* names[N_PARAMS] = {
        "rho","sigma_total","r_split","mu_base",
        "mu_scale","mu_rate","sigma_scale","sigma_rate"
    };
    printf("\n  -- Kalman Parameters (update #%d, Phase %d) --\n\n",
           t->snap.n_updates, t->phase);
    printf("  %-14s  %8s  %8s  %6s  %s\n","Parameter","Value","+/-sqP","Gated","Conv");
    printf("  -----------------------------------------------\n");
    for (int i=0;i<N_PARAMS;i++) {
        float s=sqrtf(fmaxf(t->P[i*N_PARAMS+i],0));
        const char* st=(t->converged[i]==1)?"Y":(t->converged[i]==-1)?"lock":"-";
        printf("  %-14s  %8.4f  %8.4f  %6.4f  %s\n",
               names[i],t->x[i],s,t->snap.theta[i],st);
    }
    printf("  -----------------------------------------------\n");
    printf("  Derived: sz=%.4f sb=%.4f\n", t->snap.sigma_z, t->snap.sigma_base);
    printf("  At z=%.3f: mu=%.4f sh=%.4f th=%.4f\n",
           t->snap.z_mean, t->snap.mu, t->snap.sigma_h, t->snap.theta_speed);
    printf("  SMC2: ESS=%.1f accept=%.1f%%\n",
           t->snap.last_ess, t->snap.last_accept_rate*100);
}

void param_tracker_print_phased_status(const ParamTracker* t) {
    const char* pn[]={"???","FLOORS(4p)","CEILINGS(6p)","RATES(8p)"};
    const ZRangeTracker* zt=&t->z_tracker;
    const PhasedConfig* pc=&t->phased_config;
    printf("Phased Learning:\n");
    printf("  Phase: %d %s%s\n",t->phase,pn[t->phase],pc->enabled?"":" [OFF]");
    printf("  Windows: %d\n",zt->n_windows);
    printf("  z_win: [%.3f,%.3f] range=%.3f\n",
           zt->z_min_window,zt->z_max_window,zt->z_max_window-zt->z_min_window);
    printf("  z_global: [%.3f,%.3f]  ema=%.3f\n",zt->z_min_seen,zt->z_max_seen,zt->z_ema);
    printf("  Forward:  hi_z=%d/%d  wide=%d/%d\n",
           zt->high_z_streak,pc->ceiling_z_sustained,
           zt->wide_range_streak,pc->rate_range_sustained);
    printf("  Backward: calm=%d/%d  narrow=%d/%d\n",
           zt->calm_streak,pc->ceiling_z_sustained,
           zt->narrow_range_streak,pc->rate_range_sustained);
    printf("  Saved: ms=%.4f%s ss=%.4f%s mr=%.4f%s sr=%.4f%s\n",
           pc->fixed_mu_scale, pc->learned_ceilings?" (L)":" (P)",
           pc->fixed_sigma_scale, pc->learned_ceilings?" (L)":" (P)",
           pc->fixed_mu_rate, pc->learned_rates?" (L)":" (P)",
           pc->fixed_sigma_rate, pc->learned_rates?" (L)":" (P)");
    if (t->n_phase_transitions>0) {
        printf("  Transitions (%d):\n",t->n_phase_transitions);
        for (int i=0;i<t->n_phase_transitions;i++) {
            const PhaseTransition* tr=&t->phase_history[i];
            printf("    %s w=%d P%d->P%d z=[%.2f,%.2f]\n",
                   (tr->to>tr->from)?"^":"v",tr->window,tr->from,tr->to,tr->z_min,tr->z_max);
        }
    }
}
