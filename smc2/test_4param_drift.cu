/**
 * @file test_4param_drift.cu
 * @brief Two tests for the dual-timescale theory
 *
 * Test A: 4-param fast mode + Kalman + drift
 *   Fix curves at truth, learn [ρ, σ_total, r_split, μ_base].
 *   μ_base shifts -1 → +1 at t=6000 over 12k ticks.
 *   If ridge removal works, μ_base should track correctly.
 *
 * Test B: 8-param Kalman tracker over 50k ticks with drift
 *   Same shift at t=25000. ~33 overlapping windows.
 *   Tests whether more windows + Kalman averaging resolves curves
 *   despite systematic ridge bias during drift.
 *
 * Build:
 *   nvcc -O2 -arch=sm_120 -o test_4param_drift \
 *        test_4param_drift.cu smc2_param_tracker.cu smc2_rbpf_cuda.cu \
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
 * DGP
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

static void print_comparison(const char* label, const float* est, const float* std_arr,
                              const float* truth, int n_params) {
    printf("\n  ── %s ──\n\n", label);
    printf("  %-14s  %8s  %8s  %8s  %7s  %7s\n",
           "Parameter", "True", "Est", "Std", "Err%", "z-score");
    printf("  ──────────────────────────────────────────────────────────────\n");
    int n_ok = 0;
    for (int i = 0; i < n_params; i++) {
        float s = std_arr[i];
        float err = est[i] - truth[i];
        float z = fabsf(err) / fmaxf(s, 1e-6f);
        float pct = (fabsf(truth[i]) > 0.01f) ? 100.0f * err / truth[i] : err * 100.0f;
        const char* tag = (z <= 2.0f) ? "OK" : (z <= 3.0f) ? "WARN" : "MISS";
        if (z <= 2.0f) n_ok++;
        printf("  %-14s  %8.4f  %8.4f  %8.4f  %+6.1f%%  %7.2f  [%s]\n",
               param_names[i], truth[i], est[i], s, pct, z, tag);
    }
    printf("  ──────────────────────────────────────────────────────────────\n");
    printf("  %d/%d within 2σ\n", n_ok, n_params);
}

#ifndef N_THETA_TEST
#define N_THETA_TEST 1024
#endif
#ifndef N_INNER_TEST
#define N_INNER_TEST 512
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * Test A: 4-Param Fast Mode + Kalman + Drift
 *
 * Fix [μ_scale, μ_rate, σ_scale, σ_rate] at their true values.
 * Only learn [ρ, σ_total, r_split, μ_base].
 * μ_base shifts -1 → +1 at t=6000.
 *
 * Expected: with no ridge, μ_base gets the full shift signal.
 * The Kalman should track it within a few post-shift windows.
 *═══════════════════════════════════════════════════════════════════════════*/

void test_4param_drift(void) {
    printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  Test A: 4-Param Fast Mode + Kalman + Drift                 ║\n");
    printf("║  Curves fixed at truth. Only [ρ,σ_total,r_split,μ_base].   ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");

    RSVParams truth = make_truth(0.95, 0.10, -1.0, 0.15);
    float mu_base_phase2 = 1.0f;

    int T_total = 12000;
    int t_shift = 6000;
    int W = 3000;
    int STRIDE = 1500;

    float* y = (float*)malloc(T_total * sizeof(float));
    simulate_rsv_drift(&truth, mu_base_phase2, t_shift, y, T_total);

    printf("\nDGP: μ_base = -1.0 for t < %d, then +1.0 for t >= %d\n",
           t_shift, t_shift);
    printf("Fixed curves: μ_scale=%.2f μ_rate=%.2f σ_scale=%.2f σ_rate=%.2f\n",
           truth.mu_scale, truth.mu_rate, truth.sigma_scale, truth.sigma_rate);
    printf("W=%d, stride=%d\n\n", W, STRIDE);

    /* ── Create tracker ── */
    ParamTracker* tracker = param_tracker_create(W, STRIDE, N_THETA_TEST, N_INNER_TEST);
    SMC2StateCUDA* smc2 = param_tracker_get_smc2(tracker);
    smc2_cuda_set_seed(smc2, 11111);

    /* Set 4-param mode: fix curve params at truth */
    uint8_t mask[N_PARAMS] = {0, 0, 0, 0, 1, 1, 1, 1};
    float fixed_vals[N_PARAMS] = {
        0.0f, 0.0f, 0.0f, 0.0f,                /* ignored (learned) */
        (float)truth.mu_scale,                    /* μ_scale = 0.5 */
        (float)truth.mu_rate,                     /* μ_rate  = 1.0 */
        (float)truth.sigma_scale,                 /* σ_scale = 0.1 */
        (float)truth.sigma_rate                   /* σ_rate  = 1.0 */
    };
    smc2_cuda_set_fixed_params(smc2, mask, fixed_vals);

    /* Drift config: responsive μ_base */
    DriftConfig drift;
    drift.q_rho = 1e-5f;
    drift.q_sigma_total = 1e-5f;
    drift.q_r_split = 1e-5f;
    drift.q_mu_base = 1e-2f;
    drift.q_mu_scale = 0.0f;    /* Fixed — zero process noise */
    drift.q_mu_rate = 0.0f;
    drift.q_sigma_scale = 0.0f;
    drift.q_sigma_rate = 0.0f;
    param_tracker_set_drift(tracker, &drift);

    int n_windows = 0;
    printf("  %-4s  %-15s  %8s  %8s  %8s  %8s  %s\n",
           "Win", "Range", "μ_base", "±√P", "True", "Err", "Status");
    printf("  ───────────────────────────────────────────────────────────────\n");

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    cudaEventRecord(ev0);

    for (int t = 0; t < T_total; t++) {
        param_tracker_feed(tracker, y[t]);

        if (param_tracker_window_ready(tracker)) {
            param_tracker_run_window(tracker);
            n_windows++;

            ParamSnapshot snap;
            param_tracker_get_snapshot(tracker, &snap);
            int win_end = t;
            int win_start = t - W + 1;

            const char* status;
            float true_mu;
            if (win_end < t_shift) {
                status = "pre-shift";
                true_mu = -1.0f;
            } else if (win_start >= t_shift) {
                status = "POST ✓";
                true_mu = 1.0f;
            } else {
                int post_frac = (int)(100.0f * (win_end - t_shift + 1) / (float)W);
                static char sbuf[32];
                snprintf(sbuf, sizeof(sbuf), "MIXED (%d%%)", post_frac);
                status = sbuf;
                true_mu = 1.0f;
            }

            float mu_est = snap.theta[3];
            float mu_std = sqrtf(fmaxf(snap.P_diag[3], 0.0f));
            printf("  %-4d  t=%5d..%5d  %+8.3f  %8.4f  %+8.1f  %+8.3f  %s  ESS=%.0f acc=%.0f%%\n",
                   n_windows, win_start, win_end, mu_est, mu_std,
                   true_mu, mu_est - true_mu, status,
                   snap.last_ess, snap.last_accept_rate * 100.0f);
        }
    }

    cudaEventRecord(ev1); cudaEventSynchronize(ev1);
    float ms;
    cudaEventElapsedTime(&ms, ev0, ev1);
    printf("\n  %d windows, %.1f ms total (%.1f ms/window)\n", n_windows, ms, ms / n_windows);

    /* Final comparison — only 4 learned params */
    ParamSnapshot final_snap;
    param_tracker_get_snapshot(tracker, &final_snap);

    float tv_final[4] = {
        (float)truth.rho,
        (float)truth.sigma_total,
        (float)truth.r_split,
        mu_base_phase2   /* post-shift truth */
    };
    float est_4[4], std_4[4];
    for (int i = 0; i < 4; i++) {
        est_4[i] = final_snap.theta[i];
        std_4[i] = sqrtf(fmaxf(final_snap.P_diag[i], 0.0f));
    }
    print_comparison("4-Param Fast Mode (final, post-shift truth)", est_4, std_4, tv_final, 4);

    param_tracker_destroy(tracker);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    free(y);
}

/*═══════════════════════════════════════════════════════════════════════════
 * Test B: 8-Param Kalman Tracker over 50k Ticks with Drift
 *
 * Same overlapping window Kalman approach, but over much more data.
 * μ_base shifts -1 → +1 at t=25000.
 * ~33 windows total (stride=1500).
 *
 * Question: does Kalman averaging over 33 windows resolve the ridge
 * despite the drift-induced systematic bias?
 *
 * Expected: ridge bias remains systematic after the shift.
 * The curve params will be contaminated. This test should FAIL
 * to confirm the 4-param approach is necessary.
 *═══════════════════════════════════════════════════════════════════════════*/

void test_8param_long_drift(void) {
    printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  Test B: 8-Param Kalman Tracker — 50k Ticks + Drift         ║\n");
    printf("║  Full 8-param SMC² windows. More data = ridge resolved?     ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");

    RSVParams truth = make_truth(0.95, 0.10, -1.0, 0.15);
    float mu_base_phase2 = 1.0f;

    int T_total = 50000;
    int t_shift = 25000;
    int W = 3000;
    int STRIDE = 1500;

    float* y = (float*)malloc(T_total * sizeof(float));
    simulate_rsv_drift(&truth, mu_base_phase2, t_shift, y, T_total);

    printf("\nDGP: μ_base = -1.0 for t < %d, then +1.0 for t >= %d\n",
           t_shift, t_shift);
    printf("T=%d, W=%d, stride=%d → ~%d windows\n\n",
           T_total, W, STRIDE, (T_total - W) / STRIDE + 1);

    ParamTracker* tracker = param_tracker_create(W, STRIDE, N_THETA_TEST, N_INNER_TEST);
    smc2_cuda_set_seed(param_tracker_get_smc2(tracker), 22222);

    /* Same drift config as the regular drift test */
    DriftConfig drift;
    drift.q_rho = 1e-5f;
    drift.q_sigma_total = 1e-5f;
    drift.q_r_split = 1e-5f;
    drift.q_mu_base = 1e-2f;
    drift.q_mu_scale = 1e-7f;
    drift.q_mu_rate = 1e-7f;
    drift.q_sigma_scale = 1e-7f;
    drift.q_sigma_rate = 1e-7f;
    param_tracker_set_drift(tracker, &drift);

    int n_windows = 0;

    /* Print every 5th window to keep output manageable */
    printf("  %-4s  %-15s  %8s  %8s  %6s  %6s  %6s  %6s  %s\n",
           "Win", "Range", "μ_base", "±√P",
           "μ_scl", "μ_rt", "σ_scl", "σ_rt", "Status");
    printf("  ─────────────────────────────────────────────────────────────────────────\n");

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    cudaEventRecord(ev0);

    for (int t = 0; t < T_total; t++) {
        param_tracker_feed(tracker, y[t]);

        if (param_tracker_window_ready(tracker)) {
            param_tracker_run_window(tracker);
            n_windows++;

            if (n_windows % 5 == 1 || n_windows <= 3 || t >= T_total - STRIDE) {
                ParamSnapshot snap;
                param_tracker_get_snapshot(tracker, &snap);
                int win_end = t;
                int win_start = t - W + 1;

                const char* status;
                if (win_end < t_shift) status = "pre";
                else if (win_start >= t_shift) status = "POST";
                else status = "MIXED";

                float mu_std = sqrtf(fmaxf(snap.P_diag[3], 0.0f));
                printf("  %-4d  t=%5d..%5d  %+8.3f  %8.4f  %6.3f  %6.3f  %6.3f  %6.3f  %s\n",
                       n_windows, win_start, win_end,
                       snap.theta[3], mu_std,
                       snap.theta[4], snap.theta[5],
                       snap.theta[6], snap.theta[7], status);
            }
        }
    }

    cudaEventRecord(ev1); cudaEventSynchronize(ev1);
    float ms;
    cudaEventElapsedTime(&ms, ev0, ev1);
    printf("\n  %d windows, %.1f ms total (%.1f ms/window)\n", n_windows, ms, ms / n_windows);

    /* Final comparison — all 8 params, truth = post-shift */
    ParamSnapshot final_snap;
    param_tracker_get_snapshot(tracker, &final_snap);

    float tv_final[N_PARAMS];
    get_true_arr(&truth, tv_final);
    tv_final[3] = mu_base_phase2;  /* post-shift truth */

    float est_8[N_PARAMS], std_8[N_PARAMS];
    for (int i = 0; i < N_PARAMS; i++) {
        est_8[i] = final_snap.theta[i];
        std_8[i] = sqrtf(fmaxf(final_snap.P_diag[i], 0.0f));
    }
    print_comparison("8-Param Kalman (50k ticks, post-shift truth)", est_8, std_8, tv_final, N_PARAMS);

    param_tracker_destroy(tracker);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    free(y);
}

/*═══════════════════════════════════════════════════════════════════════════
 * Main
 *═══════════════════════════════════════════════════════════════════════════*/

int main(int argc, char** argv) {
    printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  Dual-Timescale Theory Tests                                 ║\n");
    printf("║  A: 4-param fast mode + drift (should PASS)                  ║\n");
    printf("║  B: 8-param Kalman 50k + drift (expected to FAIL)            ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");

    seed_rng(12345);

    if (argc > 1) {
        if (strcmp(argv[1], "4param") == 0)      test_4param_drift();
        else if (strcmp(argv[1], "8param") == 0)  test_8param_long_drift();
        else {
            printf("Usage: %s [4param|8param]\n", argv[0]);
            return 1;
        }
        return 0;
    }

    test_4param_drift();
    test_8param_long_drift();

    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("Theory tests completed.\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    return 0;
}
