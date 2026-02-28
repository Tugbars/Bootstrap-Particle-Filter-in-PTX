/*═══════════════════════════════════════════════════════════════════════════════
 * @file test_warm_start.cu
 * @brief Standalone test: does warm-starting SMC² from Kalman posterior help?
 *
 * Runs param_tracker on a stationary DGP twice:
 *   A. COLD — every window inits SMC² from prior (current behavior)
 *   B. WARM — after 2 Kalman updates, inits from N(x_kalman, α·(P+Q))
 *
 * Measures per-window:
 *   - ESS after SMC² completes (higher = less degenerate)
 *   - Posterior std on each param (tighter = better convergence)
 *   - RMSE of θ̂_window vs true params
 *
 * And overall:
 *   - Kalman-filtered RMSE vs true params
 *   - Kalman P diagonal (accumulated uncertainty)
 *
 * Stationary DGP means true params don't change — pure signal, no drift.
 * Any improvement must come from better per-window initialization.
 *
 * Build:
 *   nvcc -O3 test_warm_start.cu smc2_rbpf_cuda.cu smc2_param_tracker.cu \
 *        -o test_warm_start -lcuda -lcurand
 *
 * Usage:
 *   ./test_warm_start
 *═══════════════════════════════════════════════════════════════════════════════*/

#include "smc2_param_tracker.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

/* ── PRNG ────────────────────────────────────────────────────────────────── */

static unsigned int g_seed = 42;

static inline float randf() {
    g_seed = g_seed * 1103515245 + 12345;
    return (float)((g_seed >> 16) & 0x7FFF) / 32768.0f;
}

static inline float randn() {
    float u1 = randf() + 1e-10f;
    float u2 = randf();
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

/* ── Stationary DGP ──────────────────────────────────────────────────────── */

struct TrueDGP {
    float rho;
    float sigma_z;
    float sigma_base;
    float mu_base;
    float mu_scale;
    float mu_rate;
    float sigma_scale;
    float sigma_rate;
};

static TrueDGP default_dgp() {
    TrueDGP d;
    d.rho          = 0.95f;
    d.sigma_z      = 0.10f;
    d.sigma_base   = 0.15f;
    d.mu_base      = -1.0f;
    d.mu_scale     = 0.5f;
    d.mu_rate      = 1.0f;
    d.sigma_scale  = 0.1f;
    d.sigma_rate   = 1.0f;
    return d;
}

static inline float sat_exp(float base, float scale, float rate, float z) {
    return base + scale * (1.0f - expf(-rate * z));
}

struct GeneratedData {
    std::vector<float> returns;
    std::vector<float> true_h;
    std::vector<float> true_z;
};

static GeneratedData generate_stationary(const TrueDGP& dgp, int N) {
    GeneratedData gd;
    float z_tilde = 0.0f;
    float h = dgp.mu_base;

    for (int t = 0; t < N; t++) {
        z_tilde = dgp.rho * z_tilde + dgp.sigma_z * randn();
        float z = 1.5f * (1.0f + tanhf(z_tilde));

        float mu_z    = sat_exp(dgp.mu_base, dgp.mu_scale, dgp.mu_rate, z);
        float sigma_h = sat_exp(0.0f, dgp.sigma_scale, dgp.sigma_rate, z)
                         + dgp.sigma_base;

        float phi = 0.8f;
        h = mu_z + phi * (h - mu_z) + sigma_h * randn();

        float y = expf(h * 0.5f) * randn();

        gd.returns.push_back(y);
        gd.true_h.push_back(h);
        gd.true_z.push_back(z);
    }

    return gd;
}

/* ── Per-window metrics ──────────────────────────────────────────────────── */

struct WindowMetrics {
    int    window_idx;
    float  ess;
    float  accept_rate;
    float  param_rmse;      /* RMSE of SMC² posterior mean vs truth */
    float  std_mu_base;     /* Posterior std on mu_base */
    float  std_mu_scale;    /* Posterior std on mu_scale */
    float  std_mu_rate;     /* Posterior std on mu_rate */
    int    warm;            /* 1 if warm-started, 0 if cold */
};

/* ── Run one configuration ───────────────────────────────────────────────── */

enum WarmMode { MODE_COLD, MODE_WARM };

struct RunResult {
    std::vector<WindowMetrics> windows;
    float kalman_param_rmse;
    float kalman_P_trace;
};

/* True theta vector for RMSE computation */
static void get_true_theta(const TrueDGP& dgp, float* theta) {
    float sigma_total = sqrtf(dgp.sigma_z * dgp.sigma_z +
                               dgp.sigma_base * dgp.sigma_base);
    float r_split = dgp.sigma_z / sigma_total;

    theta[0] = dgp.rho;
    theta[1] = sigma_total;
    theta[2] = r_split;
    theta[3] = dgp.mu_base;
    theta[4] = dgp.mu_scale;
    theta[5] = dgp.mu_rate;
    theta[6] = dgp.sigma_scale;
    theta[7] = dgp.sigma_rate;
}

static RunResult run_test(
    const GeneratedData& gd,
    const TrueDGP& dgp,
    WarmMode mode,
    int window_size,
    int stride,
    int N_theta,
    int N_inner
) {
    RunResult result;

    float true_theta[N_PARAMS];
    get_true_theta(dgp, true_theta);

    /* Create tracker */
    ParamTracker* t = param_tracker_create(window_size, stride, N_theta, N_inner);

    /* Set priors centered on truth with moderate uncertainty */
    SMC2StateCUDA* smc2 = param_tracker_get_smc2(t);
    smc2->prior.rho_mean         = dgp.rho;          smc2->prior.rho_std         = 0.05f;
    float sigma_total = sqrtf(dgp.sigma_z * dgp.sigma_z +
                               dgp.sigma_base * dgp.sigma_base);
    smc2->prior.sigma_total_mean = sigma_total * 1.5f; smc2->prior.sigma_total_std = 0.1f;
    smc2->prior.r_split_mean     = 0.5f;               smc2->prior.r_split_std     = 0.2f;
    smc2->prior.mu_base_mean     = dgp.mu_base;        smc2->prior.mu_base_std     = 1.0f;
    smc2->prior.mu_scale_mean    = dgp.mu_scale;       smc2->prior.mu_scale_std    = 1.5f;
    smc2->prior.mu_rate_mean     = 0.5f;               smc2->prior.mu_rate_std     = 0.3f;
    smc2->prior.sigma_scale_mean = 0.5f;               smc2->prior.sigma_scale_std = 0.3f;
    smc2->prior.sigma_rate_mean  = 0.3f;               smc2->prior.sigma_rate_std  = 0.2f;

    /* If cold mode, we need to prevent warm-start.
     * The tracker warm-starts when snap.n_updates >= 2.
     * For cold mode, we'll track this externally and force cold. */

    int N = (int)gd.returns.size();
    int window_count = 0;

    for (int tick = 0; tick < N; tick++) {
        param_tracker_feed(t, gd.returns[tick]);

        if (param_tracker_window_ready(t)) {
            /* For COLD mode: force prior init by resetting the state */
            if (mode == MODE_COLD) {
                param_tracker_force_cold(t);
            }

            param_tracker_run_window(t);
            window_count++;

            /* Extract per-window metrics */
            ParamSnapshot snap;
            param_tracker_get_snapshot(t, &snap);

            WindowMetrics wm = {};
            wm.window_idx  = window_count;
            wm.ess         = snap.last_ess;
            wm.accept_rate = snap.last_accept_rate;
            wm.warm        = (mode == MODE_WARM && snap.n_updates >= 3) ? 1 : 0;

            /* SMC² posterior mean RMSE vs truth */
            double sum_sq = 0;
            for (int i = 0; i < N_PARAMS; i++) {
                double err = snap.theta[i] - true_theta[i];
                sum_sq += err * err;
            }
            wm.param_rmse = (float)sqrt(sum_sq / N_PARAMS);

            /* Posterior stds from Kalman P diagonal */
            wm.std_mu_base  = sqrtf(fmaxf(snap.P_diag[3], 0.0f));
            wm.std_mu_scale = sqrtf(fmaxf(snap.P_diag[4], 0.0f));
            wm.std_mu_rate  = sqrtf(fmaxf(snap.P_diag[5], 0.0f));

            result.windows.push_back(wm);
        }
    }

    /* Final Kalman state */
    ParamSnapshot final_snap;
    param_tracker_get_snapshot(t, &final_snap);

    double sum_sq = 0;
    float P_trace = 0;
    for (int i = 0; i < N_PARAMS; i++) {
        double err = final_snap.theta[i] - true_theta[i];
        sum_sq += err * err;
        P_trace += final_snap.P_diag[i];
    }
    result.kalman_param_rmse = (float)sqrt(sum_sq / N_PARAMS);
    result.kalman_P_trace    = P_trace;

    param_tracker_destroy(t);
    return result;
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main() {
    int N         = 30000;   /* 30K ticks — ~10 windows at stride 3000 */
    int W         = 3000;
    int stride    = 1500;
    int N_theta   = 1024;
    int N_inner   = 512;

    TrueDGP dgp = default_dgp();

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║   Warm-Start Test — Stationary DGP                              ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════╣\n");
    printf("║  Ticks: %d   Window: %d   Stride: %d                       ║\n", N, W, stride);
    printf("║  SMC² θ: %d × %d                                           ║\n", N_theta, N_inner);
    printf("║  True: ρ=%.2f σ_z=%.2f μ_base=%.1f μ_scale=%.1f μ_rate=%.1f  ║\n",
           dgp.rho, dgp.sigma_z, dgp.mu_base, dgp.mu_scale, dgp.mu_rate);
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");

    /* Generate data once, use for both runs */
    g_seed = 12345;
    GeneratedData gd = generate_stationary(dgp, N);

    /* ── Run both modes ───────────────────────────────────────────────── */
    printf("  Running COLD (prior init every window)...\n");
    g_seed = 42;  /* Reset RNG for reproducibility */
    RunResult cold = run_test(gd, dgp, MODE_COLD, W, stride, N_theta, N_inner);

    printf("  Running WARM (Kalman-informed init after 2 updates)...\n");
    g_seed = 42;
    RunResult warm = run_test(gd, dgp, MODE_WARM, W, stride, N_theta, N_inner);

    /* ── Per-window comparison ────────────────────────────────────────── */
    int n_win = (int)fmin(cold.windows.size(), warm.windows.size());

    printf("\n");
    printf("  ═══════════════════════════════════════════════════════════════════════════════\n");
    printf("  Per-window comparison\n");
    printf("  ───────────────────────────────────────────────────────────────────────────────\n");
    printf("  %4s | %8s %8s | %8s %8s | %8s %8s | %4s\n",
           "Win", "ESS_c", "ESS_w", "RMSE_c", "RMSE_w", "σ_μb_c", "σ_μb_w", "warm");
    printf("  ──── | ──────── ──────── | ──────── ──────── | ──────── ──────── | ────\n");

    for (int w = 0; w < n_win; w++) {
        WindowMetrics* c = &cold.windows[w];
        WindowMetrics* wr = &warm.windows[w];
        printf("  %4d | %8.1f %8.1f | %8.4f %8.4f | %8.4f %8.4f | %s\n",
               w + 1,
               c->ess, wr->ess,
               c->param_rmse, wr->param_rmse,
               c->std_mu_base, wr->std_mu_base,
               wr->warm ? "YES" : "no");
    }

    /* ── Averages for windows 3+ (where warm-start is active) ─────── */
    if (n_win >= 4) {
        double cold_ess_avg = 0, warm_ess_avg = 0;
        double cold_rmse_avg = 0, warm_rmse_avg = 0;
        double cold_std_avg = 0, warm_std_avg = 0;
        int count = 0;

        for (int w = 2; w < n_win; w++) {  /* Skip first 2 (both are cold) */
            cold_ess_avg  += cold.windows[w].ess;
            warm_ess_avg  += warm.windows[w].ess;
            cold_rmse_avg += cold.windows[w].param_rmse;
            warm_rmse_avg += warm.windows[w].param_rmse;
            cold_std_avg  += cold.windows[w].std_mu_base;
            warm_std_avg  += warm.windows[w].std_mu_base;
            count++;
        }

        cold_ess_avg  /= count; warm_ess_avg  /= count;
        cold_rmse_avg /= count; warm_rmse_avg /= count;
        cold_std_avg  /= count; warm_std_avg  /= count;

        printf("\n  Averages (windows 3+, where warm-start is active):\n");
        printf("  ───────────────────────────────────────────────────────────────────────────────\n");
        printf("  %-24s %12s %12s %10s\n", "", "Cold", "Warm", "Δ%");
        printf("  %-24s %12.1f %12.1f %+9.1f%%\n", "Mean ESS",
               cold_ess_avg, warm_ess_avg,
               100.0 * (warm_ess_avg / cold_ess_avg - 1.0));
        printf("  %-24s %12.4f %12.4f %+9.1f%%\n", "Mean param RMSE",
               cold_rmse_avg, warm_rmse_avg,
               100.0 * (warm_rmse_avg / cold_rmse_avg - 1.0));
        printf("  %-24s %12.4f %12.4f %+9.1f%%\n", "Mean √P[μ_base]",
               cold_std_avg, warm_std_avg,
               100.0 * (warm_std_avg / cold_std_avg - 1.0));
    }

    /* ── Grand summary ────────────────────────────────────────────────── */
    printf("\n");
    printf("  ═══════════════════════════════════════════════════════════════════════════════\n");
    printf("  FINAL KALMAN STATE\n");
    printf("  ───────────────────────────────────────────────────────────────────────────────\n");
    printf("  %-24s %12s %12s %10s\n", "", "Cold", "Warm", "Δ%");
    printf("  %-24s %12.4f %12.4f %+9.1f%%\n", "Kalman param RMSE",
           cold.kalman_param_rmse, warm.kalman_param_rmse,
           100.0 * (warm.kalman_param_rmse / cold.kalman_param_rmse - 1.0));
    printf("  %-24s %12.4f %12.4f %+9.1f%%\n", "Kalman P trace",
           cold.kalman_P_trace, warm.kalman_P_trace,
           100.0 * (warm.kalman_P_trace / cold.kalman_P_trace - 1.0));

    printf("\n  Expected behavior:\n");
    printf("    • Windows 1-2: identical (both cold-start from prior)\n");
    printf("    • Windows 3+: warm should show higher ESS (particles start in good region)\n");
    printf("    • Windows 3+: warm should show tighter Σ (faster convergence per window)\n");
    printf("    • Final Kalman: warm should have lower RMSE and tighter P\n");
    printf("    • If warm shows NO improvement: init_from_gaussian may not be linked\n");
    printf("  ═══════════════════════════════════════════════════════════════════════════════\n\n");

    return 0;
}
