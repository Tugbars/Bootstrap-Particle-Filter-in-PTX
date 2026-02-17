/**
 * @file test_param_tracker.cu
 * @brief Test suite for the Kalman Parameter Tracker
 *
 * Three tests:
 *   1. Fixed params: Kalman vs Oracle (single long SMC²)
 *   2. Fixed params: Kalman convergence across 4 windows
 *   3. Drifting μ_base: Kalman tracks drift, Oracle averages it out
 *
 * Build:
 *   nvcc -O2 -arch=sm_120 -o test_param_tracker \
 *        test_param_tracker.cu smc2_param_tracker.cu smc2_rbpf_cuda.cu \
 *        -lcurand --expt-relaxed-constexpr
 */

#include "smc2_param_tracker.cuh"
#include "smc2_rbpf_batch.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

/*═══════════════════════════════════════════════════════════════════════════
 * Host RNG (xoshiro256**)
 *═══════════════════════════════════════════════════════════════════════════*/

static uint64_t s_rng[4];
static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static uint64_t xoshiro_next(void) {
    uint64_t r = rotl(s_rng[1] * 5, 7) * 9;
    uint64_t t = s_rng[1] << 17;
    s_rng[2] ^= s_rng[0]; s_rng[3] ^= s_rng[1];
    s_rng[1] ^= s_rng[2]; s_rng[0] ^= s_rng[3];
    s_rng[2] ^= t; s_rng[3] = rotl(s_rng[3], 45);
    return r;
}

static void seed_rng(uint64_t seed) {
    for (int i = 0; i < 4; i++) {
        seed += 0x9E3779B97F4A7C15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        s_rng[i] = z ^ (z >> 31);
    }
}

static double uniform01(void) {
    return (double)(xoshiro_next() >> 11) * (1.0 / 9007199254740992.0);
}

static double randn(void) {
    double u1 = uniform01(), u2 = uniform01();
    while (u1 < 1e-15) u1 = uniform01();
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

/*═══════════════════════════════════════════════════════════════════════════
 * DGP (same as test_smc2_rbpf.cu)
 *═══════════════════════════════════════════════════════════════════════════*/

struct RSVParams {
    double rho, sigma_z, mu_base, sigma_base;
    double mu_scale, mu_rate, sigma_scale, sigma_rate;
    double theta_base, theta_scale, theta_rate;
    double sigma_total, r_split;
};

static double eval_curve_d(double base, double scale, double rate, double z) {
    return base + scale * (1.0 - exp(-rate * z));
}

static double zt_to_z(double zt) { return 1.5 * (1.0 + tanh(zt)); }

static RSVParams make_truth(double rho, double sigma_z, double mu_base, double sigma_base) {
    RSVParams p;
    p.rho = rho; p.sigma_z = sigma_z; p.mu_base = mu_base; p.sigma_base = sigma_base;
    p.mu_scale = 0.5;    p.mu_rate = 1.0;
    p.sigma_scale = 0.1; p.sigma_rate = 1.0;
    p.theta_base = 0.02; p.theta_scale = 0.08; p.theta_rate = 1.5;
    p.sigma_total = sqrt(sigma_z * sigma_z + sigma_base * sigma_base);
    p.r_split = sigma_z / p.sigma_total;
    return p;
}

static void simulate_rsv(const RSVParams* p, float* y_obs, int T) {
    double omr2 = fmax(1.0 - p->rho * p->rho, 1e-6);
    double zt_std = p->sigma_z / sqrt(omr2);
    double zt = zt_std * randn();
    double z = zt_to_z(zt);
    double th = eval_curve_d(p->theta_base, p->theta_scale, p->theta_rate, z);
    double mu = eval_curve_d(p->mu_base, p->mu_scale, p->mu_rate, z);
    double sh = eval_curve_d(p->sigma_base, p->sigma_scale, p->sigma_rate, z);
    double phi = 1.0 - th;
    double hv = (sh * sh) / fmax(1.0 - phi * phi, 1e-6);
    double h = mu + sqrt(hv) * randn();

    for (int t = 0; t < T; t++) {
        zt = p->rho * zt + p->sigma_z * randn();
        z = zt_to_z(zt);
        th = eval_curve_d(p->theta_base, p->theta_scale, p->theta_rate, z);
        mu = eval_curve_d(p->mu_base, p->mu_scale, p->mu_rate, z);
        sh = eval_curve_d(p->sigma_base, p->sigma_scale, p->sigma_rate, z);
        phi = 1.0 - th;
        h = phi * h + th * mu + sh * randn();
        double eps = randn();
        double y_raw = exp(h / 2.0) * eps;
        double ysq = y_raw * y_raw;
        if (ysq < 1e-30) ysq = 1e-30;
        y_obs[t] = (float)log(ysq);
    }
}

/**
 * @brief Simulate with a parameter shift at a given tick
 *
 * Uses mu_base_1 for t < t_shift, mu_base_2 for t >= t_shift.
 */
static void simulate_rsv_drift(const RSVParams* p, float mu_base_2,
                                int t_shift, float* y_obs, int T) {
    double omr2 = fmax(1.0 - p->rho * p->rho, 1e-6);
    double zt_std = p->sigma_z / sqrt(omr2);
    double zt = zt_std * randn();
    double z = zt_to_z(zt);
    double mu_base_curr = p->mu_base;
    double th = eval_curve_d(p->theta_base, p->theta_scale, p->theta_rate, z);
    double mu = eval_curve_d(mu_base_curr, p->mu_scale, p->mu_rate, z);
    double sh = eval_curve_d(p->sigma_base, p->sigma_scale, p->sigma_rate, z);
    double phi = 1.0 - th;
    double hv = (sh * sh) / fmax(1.0 - phi * phi, 1e-6);
    double h = mu + sqrt(hv) * randn();

    for (int t = 0; t < T; t++) {
        mu_base_curr = (t < t_shift) ? p->mu_base : (double)mu_base_2;
        zt = p->rho * zt + p->sigma_z * randn();
        z = zt_to_z(zt);
        th = eval_curve_d(p->theta_base, p->theta_scale, p->theta_rate, z);
        mu = eval_curve_d(mu_base_curr, p->mu_scale, p->mu_rate, z);
        sh = eval_curve_d(p->sigma_base, p->sigma_scale, p->sigma_rate, z);
        phi = 1.0 - th;
        h = phi * h + th * mu + sh * randn();
        double eps = randn();
        double y_raw = exp(h / 2.0) * eps;
        double ysq = y_raw * y_raw;
        if (ysq < 1e-30) ysq = 1e-30;
        y_obs[t] = (float)log(ysq);
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * Helpers
 *═══════════════════════════════════════════════════════════════════════════*/

static const char* param_names[N_PARAMS] = {
    "rho", "sigma_total", "r_split", "mu_base",
    "mu_scale", "mu_rate", "sigma_scale", "sigma_rate"
};

static void get_true_arr(const RSVParams* p, float* out) {
    out[0] = (float)p->rho;
    out[1] = (float)p->sigma_total;
    out[2] = (float)p->r_split;
    out[3] = (float)p->mu_base;
    out[4] = (float)p->mu_scale;
    out[5] = (float)p->mu_rate;
    out[6] = (float)p->sigma_scale;
    out[7] = (float)p->sigma_rate;
}

static void print_comparison(const char* label, const float* est, const float* std_or_P,
                              const float* truth, int use_sqrt) {
    printf("\n  ── %s ──\n\n", label);
    printf("  %-14s  %8s  %8s  %8s  %7s  %7s\n",
           "Parameter", "True", "Est", "Std", "Err%", "z-score");
    printf("  ──────────────────────────────────────────────────────────────\n");
    int n_ok = 0;
    for (int i = 0; i < N_PARAMS; i++) {
        float s = use_sqrt ? sqrtf(fmaxf(std_or_P[i], 0.0f)) : std_or_P[i];
        float err = est[i] - truth[i];
        float z = fabsf(err) / fmaxf(s, 1e-6f);
        float pct = (fabsf(truth[i]) > 0.01f) ? 100.0f * err / truth[i] : err * 100.0f;
        const char* tag = (z <= 2.0f) ? "OK" : (z <= 3.0f) ? "WARN" : "MISS";
        if (z <= 2.0f) n_ok++;
        printf("  %-14s  %8.4f  %8.4f  %8.4f  %+6.1f%%  %7.2f  [%s]\n",
               param_names[i], truth[i], est[i], s, pct, z, tag);
    }
    printf("  ──────────────────────────────────────────────────────────────\n");
    printf("  %d/%d within 2σ\n", n_ok, N_PARAMS);
}

/*═══════════════════════════════════════════════════════════════════════════
 * Run oracle: single SMC² on full series
 *═══════════════════════════════════════════════════════════════════════════*/

static void run_oracle(float* y, int T, int N_theta, int N_inner,
                        float* oracle_mean, float* oracle_std) {
    SMC2StateCUDA* s = smc2_cuda_alloc(N_theta, N_inner);
    smc2_cuda_set_fixed_lag(s, 200);
    smc2_cuda_set_seed(s, 77777);
    smc2_cuda_init_from_prior(s);

    for (int t = 0; t < T; t++) {
        smc2_cuda_update(s, y[t]);
        if ((t + 1) % 3000 == 0) {
            float m[N_PARAMS];
            smc2_cuda_get_theta_mean(s, m);
            float ess = smc2_cuda_get_outer_ess(s);
            float acc = s->n_rejuv_total > 0
                ? 100.0f * s->n_rejuv_accepts / s->n_rejuv_total : 0.0f;
            printf("  Oracle t=%5d: ESS=%6.1f  accept=%5.1f%%  "
                   "ρ=%.3f σt=%.3f r=%.3f μb=%.2f\n",
                   t + 1, ess, acc, m[0], m[1], m[2], m[3]);
        }
    }

    smc2_cuda_get_theta_mean(s, oracle_mean);
    smc2_cuda_get_theta_std(s, oracle_std);

    printf("  Oracle final: resamples=%d  accept=%.1f%%\n",
           s->n_resamples,
           s->n_rejuv_total > 0
               ? 100.0f * s->n_rejuv_accepts / s->n_rejuv_total : 0.0f);

    smc2_cuda_free(s);
}

/*═══════════════════════════════════════════════════════════════════════════
 * Test 1 + 2: Fixed params — Oracle vs Kalman, convergence tracking
 *═══════════════════════════════════════════════════════════════════════════*/

#ifndef N_THETA_TEST
#define N_THETA_TEST 1024
#endif
#ifndef N_INNER_TEST
#define N_INNER_TEST 512
#endif

void test_fixed_params(void) {
    printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  Test 1+2: Fixed Params — Oracle vs Kalman Tracker          ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");

    RSVParams truth = make_truth(0.95, 0.10, -1.0, 0.15);
    float tv[N_PARAMS];
    get_true_arr(&truth, tv);

    int T_total = 12000;
    int W = 3000;
    int n_windows = T_total / W;

    float* y = (float*)malloc(T_total * sizeof(float));
    simulate_rsv(&truth, y, T_total);

    printf("\nTrue params: ρ=%.3f  σ_total=%.4f  r=%.4f  μ_base=%.3f\n",
           truth.rho, truth.sigma_total, truth.r_split, truth.mu_base);
    printf("             μ_scale=%.2f  μ_rate=%.2f  σ_scale=%.2f  σ_rate=%.2f\n",
           truth.mu_scale, truth.mu_rate, truth.sigma_scale, truth.sigma_rate);
    printf("T=%d, %d non-overlapping windows of %d\n\n", T_total, n_windows, W);

    /* ── Oracle: single long run ── */
    printf("── Running Oracle (single SMC² on T=%d) ──\n", T_total);
    float oracle_mean[N_PARAMS], oracle_std[N_PARAMS];

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    cudaEventRecord(ev0);
    run_oracle(y, T_total, N_THETA_TEST, N_INNER_TEST, oracle_mean, oracle_std);
    cudaEventRecord(ev1); cudaEventSynchronize(ev1);
    float oracle_ms;
    cudaEventElapsedTime(&oracle_ms, ev0, ev1);
    printf("  Oracle time: %.1f ms\n", oracle_ms);

    print_comparison("Oracle (T=12000, single run)", oracle_mean, oracle_std, tv, 0);

    /* ── Kalman tracker: 4 windows ── */
    printf("\n── Running Kalman Tracker (%d windows of %d) ──\n", n_windows, W);

    ParamTracker* tracker = param_tracker_create(W, W, N_THETA_TEST, N_INNER_TEST);
    smc2_cuda_set_seed(param_tracker_get_smc2(tracker), 88888);

    cudaEventRecord(ev0);
    for (int w = 0; w < n_windows; w++) {
        /* Feed one window of observations */
        int t_start = w * W;
        for (int t = 0; t < W; t++)
            param_tracker_feed(tracker, y[t_start + t]);

        /* Run window */
        param_tracker_run_window(tracker);

        /* Print convergence (Test 2) */
        ParamSnapshot snap;
        param_tracker_get_snapshot(tracker, &snap);
        printf("\n  After window %d (t=%d..%d):\n", w + 1, t_start, t_start + W - 1);
        printf("  %-14s  %8s  %8s  %8s\n", "Parameter", "Kalman", "±√P", "True");
        printf("  ───────────────────────────────────────────────────\n");
        for (int i = 0; i < N_PARAMS; i++) {
            float std_p = sqrtf(fmaxf(snap.P_diag[i], 0.0f));
            printf("  %-14s  %8.4f  %8.4f  %8.4f\n",
                   param_names[i], snap.theta[i], std_p, tv[i]);
        }
        printf("  SMC²: ESS=%.1f  accept=%.1f%%\n",
               snap.last_ess, snap.last_accept_rate * 100.0f);
        printf("  z̄=%.3f  μ(z̄)=%.4f  σ_h(z̄)=%.4f  θ(z̄)=%.4f\n",
               snap.z_mean, snap.mu, snap.sigma_h, snap.theta_speed);
    }
    cudaEventRecord(ev1); cudaEventSynchronize(ev1);
    float tracker_ms;
    cudaEventElapsedTime(&tracker_ms, ev0, ev1);
    printf("\n  Tracker total time: %.1f ms (%.1f ms/window)\n",
           tracker_ms, tracker_ms / n_windows);

    /* Final comparison */
    ParamSnapshot final_snap;
    param_tracker_get_snapshot(tracker, &final_snap);
    print_comparison("Kalman Tracker (4 × T=3000)", final_snap.theta, final_snap.P_diag, tv, 1);

    /* ── Side-by-side ── */
    printf("\n  ── Oracle vs Kalman (head-to-head) ──\n\n");
    printf("  %-14s  %8s  %8s  %8s  %8s  %8s\n",
           "Parameter", "True", "Oracle", "Kalman", "Orc.Std", "Kal.Std");
    printf("  ─────────────────────────────────────────────────────────────────\n");
    for (int i = 0; i < N_PARAMS; i++) {
        float kal_std = sqrtf(fmaxf(final_snap.P_diag[i], 0.0f));
        printf("  %-14s  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f\n",
               param_names[i], tv[i], oracle_mean[i], final_snap.theta[i],
               oracle_std[i], kal_std);
    }
    printf("  ─────────────────────────────────────────────────────────────────\n");

    param_tracker_destroy(tracker);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    free(y);
}

/*═══════════════════════════════════════════════════════════════════════════
 * Test 3: Drifting μ_base — Kalman tracks, Oracle averages
 *═══════════════════════════════════════════════════════════════════════════*/

void test_drift(void) {
    printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  Test 3: Drifting μ_base — Kalman Tracks, Oracle Averages   ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");

    RSVParams truth_phase1 = make_truth(0.95, 0.10, -1.0, 0.15);
    float mu_base_phase2 = 1.0f;

    int T_total = 12000;
    int t_shift = 6000;
    int W = 3000;
    int n_windows = T_total / W;

    float* y = (float*)malloc(T_total * sizeof(float));
    simulate_rsv_drift(&truth_phase1, mu_base_phase2, t_shift, y, T_total);

    printf("\nDGP: μ_base = -1.0 for t < %d, then μ_base = +1.0 for t >= %d\n",
           t_shift, t_shift);
    printf("Other params fixed: ρ=%.3f  σ_total=%.4f  r=%.4f\n\n",
           truth_phase1.rho, truth_phase1.sigma_total, truth_phase1.r_split);

    /* ── Oracle: sees everything, averages over the shift ── */
    printf("── Running Oracle (T=%d, sees both phases) ──\n", T_total);
    float oracle_mean[N_PARAMS], oracle_std[N_PARAMS];
    run_oracle(y, T_total, N_THETA_TEST, N_INNER_TEST, oracle_mean, oracle_std);

    /* ── Kalman tracker: should track the shift ── */
    printf("\n── Running Kalman Tracker (%d windows of %d) ──\n", n_windows, W);

    ParamTracker* tracker = param_tracker_create(W, W, N_THETA_TEST, N_INNER_TEST);
    smc2_cuda_set_seed(param_tracker_get_smc2(tracker), 99999);

    /* Increase Q for mu_base so tracker responds to shift */
    DriftConfig drift;
    drift.q_rho = 1e-5f;
    drift.q_sigma_total = 1e-5f;
    drift.q_r_split = 1e-5f;
    drift.q_mu_base = 1e-2f;    /* Very responsive to μ_base changes */
    drift.q_mu_scale = 1e-7f;
    drift.q_mu_rate = 1e-7f;
    drift.q_sigma_scale = 1e-7f;
    drift.q_sigma_rate = 1e-7f;
    param_tracker_set_drift(tracker, &drift);

    for (int w = 0; w < n_windows; w++) {
        int t_start = w * W;
        for (int t = 0; t < W; t++)
            param_tracker_feed(tracker, y[t_start + t]);

        param_tracker_run_window(tracker);

        ParamSnapshot snap;
        param_tracker_get_snapshot(tracker, &snap);

        float true_mu_base = (t_start + W - 1 < t_shift) ? -1.0f : 1.0f;
        printf("  Window %d (t=%d..%d): Kalman μ_base=%+.3f  (true=%+.1f)  √P=%.3f\n",
               w + 1, t_start, t_start + W - 1,
               snap.theta[3], true_mu_base,
               sqrtf(fmaxf(snap.P_diag[3], 0.0f)));
    }

    ParamSnapshot final_snap;
    param_tracker_get_snapshot(tracker, &final_snap);

    printf("\n  ── Drift Test Summary ──\n\n");
    printf("  True μ_base (final phase): +1.0\n");
    printf("  Oracle μ_base estimate:    %+.4f  (averages both phases → biased)\n",
           oracle_mean[3]);
    printf("  Kalman μ_base estimate:    %+.4f  (tracks the shift)\n",
           final_snap.theta[3]);
    printf("\n  Oracle error:  %+.3f  (should be ~0.0, wrong for both phases)\n",
           oracle_mean[3] - 1.0f);
    printf("  Kalman error:  %+.3f  (should be near +1.0)\n",
           final_snap.theta[3] - 1.0f);

    float oracle_off = fabsf(oracle_mean[3] - 1.0f);
    float kalman_off = fabsf(final_snap.theta[3] - 1.0f);
    printf("\n  Kalman closer to true final μ_base? %s  (%.3f vs %.3f)\n",
           kalman_off < oracle_off ? "YES" : "NO", kalman_off, oracle_off);

    param_tracker_destroy(tracker);
    free(y);
}

/*═══════════════════════════════════════════════════════════════════════════
 * Main
 *═══════════════════════════════════════════════════════════════════════════*/

int main(int argc, char** argv) {
    printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  Parameter Tracker Test Suite                                ║\n");
    printf("║  SMC² (sensor) → Kalman (tracker) → BPF (reflex)            ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");

    seed_rng(54321);

    if (argc > 1) {
        if (strcmp(argv[1], "fixed") == 0)     { test_fixed_params(); }
        else if (strcmp(argv[1], "drift") == 0) { test_drift(); }
        else {
            printf("Usage: %s [fixed|drift]\n", argv[0]);
            return 1;
        }
        return 0;
    }

    test_fixed_params();
    test_drift();

    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("All tracker tests completed.\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    return 0;
}
