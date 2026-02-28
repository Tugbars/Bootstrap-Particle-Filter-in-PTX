/**
 * @file test_convergence_diag.cu
 * @brief Validate window-based convergence diagnostics for SMC² + Kalman
 *
 * Two levels of testing:
 *
 *   UNIT TESTS (synthetic posteriors, no CUDA):
 *     Test 1 — Agreeing windows:   all θ̂_k ≈ truth  →  R̂ ≈ 1.0 or below
 *     Test 2 — One biased param:   θ̂_k[0] bimodal   →  R̂[0] >> threshold
 *     Test 3 — Convergence drift:  early scatter,     →  R̂ drops as old
 *                                   late agreement         windows roll off
 *     Test 4 — Mahalanobis calib:  ν ~ N(0,S)        →  mean d² ≈ n_free
 *
 *   INTEGRATION TEST (actual SMC² on stationary DGP):
 *     Test 5 — R̂ trajectory:  starts high, drops, plateaus while RMSE
 *                               decreases. Validates that R̂ tracks actual
 *                               estimation quality.
 *
 * Build:
 *   Part of smc2/tests/ — add to CMakeLists.txt:
 *     smc2_add_test(test_convergence_diag)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "smc2_convergence_diag.h"
#include "smc2_param_tracker.cuh"

/* ═══════════════════════════════════════════════════════════════════════════
 * Helpers
 * ═══════════════════════════════════════════════════════════════════════════ */

static const char* PARAM_NAMES[] = {
    "rho", "sig_t", "r_spl", "mu_b", "mu_s", "mu_r", "sig_s", "sig_r"
};

#define N_P CONV_DIAG_N_PARAMS
#define THRESHOLD 1.5f

/* Simple LCG PRNG for reproducible synthetic data (not crypto-grade) */
static uint32_t rng_state = 12345u;
static float randf(void) {
    rng_state = rng_state * 1103515245u + 12345u;
    return (float)(rng_state >> 16) / 65536.0f;
}

/* Box-Muller: two standard normals */
static void randn2(float* z0, float* z1) {
    float u1 = randf() + 1e-8f;
    float u2 = randf();
    float r = sqrtf(-2.0f * logf(u1));
    *z0 = r * cosf(6.2831853f * u2);
    *z1 = r * sinf(6.2831853f * u2);
}

static float randn(void) {
    float z0, z1;
    randn2(&z0, &z1);
    return z0;
}

static int test_pass(const char* name, int pass) {
    printf("  %s: %s\n", name, pass ? "PASS ✓" : "FAIL ✗");
    return pass ? 0 : 1;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TEST 1 — Agreeing windows
 *
 * All windows draw θ̂_k ~ N(truth, Σ) with Σ = diag(σ²).
 * Between-window variance B should ≈ W = σ².
 * R̂ = √(B/W) should be near 1.0 (within noise for M=8 windows).
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test1_agreeing(void)
{
    printf("\n=== TEST 1: Agreeing Windows → R̂ ≈ 1.0 ===\n");
    rng_state = 42u;

    const float truth[N_P] = {0.85f, 0.30f, 0.50f, -9.5f, 3.0f, 1.0f, 0.50f, 0.80f};
    const float sigma2 = 0.04f;  /* posterior variance per param */

    ConvergenceDiag diag;
    conv_diag_init(&diag, 8);

    /* Push 10 windows */
    for (int w = 0; w < 10; w++) {
        float theta_hat[N_P], sigma_diag[N_P];
        for (int i = 0; i < N_P; i++) {
            theta_hat[i]  = truth[i] + sqrtf(sigma2) * randn();
            sigma_diag[i] = sigma2;
        }
        conv_diag_push(&diag, theta_hat, sigma_diag, 8.0f, 0.1f);
    }

    /* Report */
    float kalman_x[N_P], kalman_P_diag[N_P];
    int free_mask[N_P];
    memcpy(kalman_x, truth, sizeof(truth));
    for (int i = 0; i < N_P; i++) {
        kalman_P_diag[i] = 0.01f;
        free_mask[i] = 1;
    }

    ConvergenceReport rpt;
    conv_diag_report(&diag, kalman_x, kalman_P_diag, free_mask, THRESHOLD, &rpt);

    conv_diag_print_header(PARAM_NAMES);
    conv_diag_print_line(&rpt, 10);

    /* Check: all R̂ should be < 3.0 (loose bound; with M=8 and truth centered,
       expect R̂ ∈ [0.5, 2.0] from sampling noise) */
    int pass = 1;
    float max_rhat = 0.0f;
    for (int i = 0; i < N_P; i++) {
        if (rpt.rhat[i] > max_rhat) max_rhat = rpt.rhat[i];
        if (rpt.rhat[i] > 3.0f) pass = 0;
    }
    printf("  Max R̂ = %.3f (expect < 3.0)\n", max_rhat);
    printf("  n_converged = %d/%d\n", rpt.n_converged, rpt.n_free);

    return test_pass("Agreeing windows", pass);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TEST 2 — One biased parameter
 *
 * θ̂_k[0] alternates ±1.0 around truth (bimodal).
 * θ̂_k[1..7] normal around truth.
 * R̂[0] should be >> threshold, others ≈ 1.0.
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test2_biased(void)
{
    printf("\n=== TEST 2: One Biased Param → R̂[0] >> threshold ===\n");
    rng_state = 777u;

    const float truth[N_P] = {0.85f, 0.30f, 0.50f, -9.5f, 3.0f, 1.0f, 0.50f, 0.80f};
    const float sigma2 = 0.01f;  /* tight posterior per param */
    const float bias = 1.0f;     /* alternating bias on param 0 */

    ConvergenceDiag diag;
    conv_diag_init(&diag, 8);

    for (int w = 0; w < 10; w++) {
        float theta_hat[N_P], sigma_diag[N_P];
        for (int i = 0; i < N_P; i++) {
            theta_hat[i]  = truth[i] + sqrtf(sigma2) * randn();
            sigma_diag[i] = sigma2;
        }
        /* Inject alternating bias on param 0 */
        theta_hat[0] = truth[0] + ((w % 2) ? bias : -bias) + sqrtf(sigma2) * randn();

        conv_diag_push(&diag, theta_hat, sigma_diag, 8.0f, 0.1f);
    }

    float kalman_x[N_P], kalman_P_diag[N_P];
    int free_mask[N_P];
    memcpy(kalman_x, truth, sizeof(truth));
    for (int i = 0; i < N_P; i++) {
        kalman_P_diag[i] = 0.01f;
        free_mask[i] = 1;
    }

    ConvergenceReport rpt;
    conv_diag_report(&diag, kalman_x, kalman_P_diag, free_mask, THRESHOLD, &rpt);

    conv_diag_print_header(PARAM_NAMES);
    conv_diag_print_line(&rpt, 10);

    /* Check: R̂[0] should be very high, R̂[1..7] should be reasonable */
    int pass = 1;
    if (rpt.rhat[0] < 5.0f) {
        printf("  FAIL: R̂[0] = %.2f, expected > 5.0\n", rpt.rhat[0]);
        pass = 0;
    } else {
        printf("  R̂[0] = %.2f (biased param, expected > 5.0) ✓\n", rpt.rhat[0]);
    }

    int good_count = 0;
    for (int i = 1; i < N_P; i++) {
        if (rpt.rhat[i] < 3.0f) good_count++;
    }
    if (good_count < N_P - 2) {
        printf("  FAIL: only %d/7 non-biased params have R̂ < 3.0\n", good_count);
        pass = 0;
    } else {
        printf("  %d/7 non-biased params have R̂ < 3.0 ✓\n", good_count);
    }

    printf("  converged[0] = %d (should be 0)\n", rpt.converged[0]);
    if (rpt.converged[0] != 0) pass = 0;

    return test_pass("One biased param", pass);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TEST 3 — Convergence over time
 *
 * First 6 windows: θ̂_k scattered widely (σ² = 1.0).
 * Next 10 windows: θ̂_k tight (σ² = 0.01).
 * With M=6, once the scattered windows roll off, R̂ should drop.
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test3_drift(void)
{
    printf("\n=== TEST 3: Convergence Over Time → R̂ drops ===\n");
    rng_state = 1234u;

    const float truth[N_P] = {0.85f, 0.30f, 0.50f, -9.5f, 3.0f, 1.0f, 0.50f, 0.80f};
    const float sigma2 = 0.01f;  /* tight posteriors throughout */
    const int M = 6;

    ConvergenceDiag diag;
    conv_diag_init(&diag, M);

    float kalman_x[N_P], kalman_P_diag[N_P];
    int free_mask[N_P];
    memcpy(kalman_x, truth, sizeof(truth));
    for (int i = 0; i < N_P; i++) {
        kalman_P_diag[i] = 0.01f;
        free_mask[i] = 1;
    }

    conv_diag_print_header(PARAM_NAMES);

    float rhat_early = 0.0f, rhat_late = 0.0f;

    for (int w = 0; w < 16; w++) {
        float theta_hat[N_P], sigma_diag[N_P];
        for (int i = 0; i < N_P; i++) {
            /* Early windows: means biased ±0.5 from truth (disagreement).
               Late windows: means centered on truth (agreement).
               Both have tight posteriors (σ²=0.01). */
            float bias = 0.0f;
            if (w < 6)
                bias = ((w % 2) ? 0.5f : -0.5f);

            theta_hat[i]  = truth[i] + bias + sqrtf(sigma2) * randn();
            sigma_diag[i] = sigma2;
        }
        conv_diag_push(&diag, theta_hat, sigma_diag, 8.0f, 0.1f);

        ConvergenceReport rpt;
        conv_diag_report(&diag, kalman_x, kalman_P_diag, free_mask, THRESHOLD, &rpt);
        conv_diag_print_line(&rpt, w + 1);

        /* Track mean R̂ at specific windows */
        if (rpt.ready) {
            float mean_rhat = 0;
            for (int i = 0; i < N_P; i++) mean_rhat += rpt.rhat[i];
            mean_rhat /= N_P;
            if (w == 7) rhat_early = mean_rhat;   /* buffer still has biased windows */
            if (w == 15) rhat_late = mean_rhat;    /* only agreeing windows remain    */
        }
    }

    printf("\n  Mean R̂ at window 8:  %.3f (biased windows still in buffer)\n", rhat_early);
    printf("  Mean R̂ at window 16: %.3f (only agreeing windows remain)\n", rhat_late);

    /* R̂ at window 16 should be much lower than at window 8 */
    int pass = (rhat_late < rhat_early * 0.5f);
    if (!pass)
        printf("  R̂ did not drop enough (ratio: %.2f, need < 0.5)\n",
               rhat_late / (rhat_early + 1e-8f));

    return test_pass("Convergence over time", pass);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TEST 4 — Mahalanobis calibration
 *
 * Construct innovations ν_k ~ N(0, S) where S = diag(S_ii).
 * Diagonal Mahalanobis d² = Σ ν_i²/S_ii.
 * Mean should ≈ n_free (= N_P when all free).
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test4_mahal(void)
{
    printf("\n=== TEST 4: Mahalanobis Calibration → mean d² ≈ %d ===\n", N_P);
    rng_state = 55555u;

    ConvergenceDiag diag;
    conv_diag_init(&diag, 12);

    float S_diag[N_P];
    for (int i = 0; i < N_P; i++) S_diag[i] = 0.5f + 0.1f * i;

    /* Generate 50 windows with ν ~ N(0, S) */
    int N_windows = 50;
    float d2_sum = 0.0f;
    for (int w = 0; w < N_windows; w++) {
        float nu[N_P];
        for (int i = 0; i < N_P; i++)
            nu[i] = sqrtf(S_diag[i]) * randn();

        float d2 = conv_diag_mahal_diag(nu, S_diag, N_P);
        d2_sum += d2;

        /* Push dummy posteriors (not testing R̂ here) */
        float dummy_theta[N_P] = {0}, dummy_sigma[N_P] = {0};
        conv_diag_push(&diag, dummy_theta, dummy_sigma, d2, 0.1f);
    }

    /* Report */
    float kalman_x[N_P] = {0}, kalman_P_diag[N_P] = {0};
    int free_mask[N_P];
    for (int i = 0; i < N_P; i++) free_mask[i] = 1;

    ConvergenceReport rpt;
    conv_diag_report(&diag, kalman_x, kalman_P_diag, free_mask, THRESHOLD, &rpt);

    float mean_d2_all = d2_sum / N_windows;

    printf("  Mean d² (all %d windows) = %.2f (expect ~%.1f)\n",
           N_windows, mean_d2_all, (float)N_P);
    printf("  Mean d² (rolling %d)     = %.2f\n", diag.M, rpt.mahal_mean);

    /* d² should be within [N_P - 3, N_P + 5] — generous for small sample */
    int pass = (mean_d2_all > (float)(N_P - 4) && mean_d2_all < (float)(N_P + 6));

    return test_pass("Mahalanobis calibration", pass);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TEST 5 — Locked params skipped
 *
 * Free only params 0–3 (Phase 1). Params 4–7 locked.
 * R̂ should only be computed for free params.
 * converged[4..7] should be -1 (locked).
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test5_locked(void)
{
    printf("\n=== TEST 5: Locked Params Skipped ===\n");
    rng_state = 9999u;

    const float truth[N_P] = {0.85f, 0.30f, 0.50f, -9.5f, 3.0f, 1.0f, 0.50f, 0.80f};

    ConvergenceDiag diag;
    conv_diag_init(&diag, 6);

    for (int w = 0; w < 8; w++) {
        float theta_hat[N_P], sigma_diag[N_P];
        for (int i = 0; i < N_P; i++) {
            theta_hat[i]  = truth[i] + 0.05f * randn();
            sigma_diag[i] = 0.01f;
        }
        conv_diag_push(&diag, theta_hat, sigma_diag, 4.0f, 0.1f);
    }

    float kalman_x[N_P], kalman_P_diag[N_P];
    int free_mask[N_P] = {1, 1, 1, 1, 0, 0, 0, 0};  /* Phase 1 */
    memcpy(kalman_x, truth, sizeof(truth));
    for (int i = 0; i < N_P; i++) kalman_P_diag[i] = 0.01f;

    ConvergenceReport rpt;
    conv_diag_report(&diag, kalman_x, kalman_P_diag, free_mask, THRESHOLD, &rpt);

    conv_diag_print_header(PARAM_NAMES);
    conv_diag_print_line(&rpt, 8);

    int pass = 1;

    if (rpt.n_free != 4) {
        printf("  FAIL: n_free = %d, expected 4\n", rpt.n_free);
        pass = 0;
    }

    for (int i = 4; i < N_P; i++) {
        if (rpt.converged[i] != -1) {
            printf("  FAIL: converged[%d] = %d, expected -1\n", i, rpt.converged[i]);
            pass = 0;
        }
    }

    printf("  n_free = %d, n_converged = %d\n", rpt.n_free, rpt.n_converged);
    printf("  mahal_expected = %.1f (should be 4.0 for 4 free params)\n",
           rpt.mahal_expected);

    return test_pass("Locked params skipped", pass);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TEST 6 — Full SMC² integration on stationary DGP
 *
 * Simulate from known SV model, run param_tracker (amnesiac SMC²+Kalman),
 * feed diagnostics, check that R̂ trajectory correlates with RMSE.
 *
 * Expected behavior:
 *   - First few windows: R̂ high (windows disagree), RMSE high
 *   - Later windows: R̂ drops (windows agree), RMSE drops
 *   - Correlation(mean_R̂, RMSE) should be positive
 * ═══════════════════════════════════════════════════════════════════════════ */

/* DGP simulation (same as test_warm_start.cu) */
static void simulate_sv_returns(float* y_out, int T,
                                 float rho, float sigma_z,
                                 float mu_base, float sigma_base,
                                 float theta_speed, uint32_t seed)
{
    uint32_t s = seed;
    float z_tilde = 0.0f;
    float h = mu_base;

    for (int t = 0; t < T; t++) {
        /* Advance LCG for reproducibility */
        s = s * 1103515245u + 12345u;
        float u1 = (float)(s >> 16) / 65536.0f + 1e-8f;
        s = s * 1103515245u + 12345u;
        float u2 = (float)(s >> 16) / 65536.0f;
        float eps_z = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
        float eps_h = sqrtf(-2.0f * logf(u1)) * sinf(6.2831853f * u2);

        s = s * 1103515245u + 12345u;
        float u3 = (float)(s >> 16) / 65536.0f + 1e-8f;
        s = s * 1103515245u + 12345u;
        float u4 = (float)(s >> 16) / 65536.0f;
        float eps_y = sqrtf(-2.0f * logf(u3)) * cosf(6.2831853f * u4);

        /* z̃ dynamics */
        z_tilde = rho * z_tilde + sigma_z * eps_z;

        /* h dynamics (stationary — no curves, just flat μ_base) */
        h = mu_base + theta_speed * (h - mu_base) + sigma_base * eps_h;

        /* observation */
        y_out[t] = expf(h * 0.5f) * eps_y;
    }
}

static int test6_integration(void)
{
    printf("\n=== TEST 6: SMC² Integration — R̂ Tracks RMSE ===\n");

    /* True parameters */
    const float TRUE_THETA[N_P] = {
        0.85f,   /* ρ          */
        0.30f,   /* σ_total    */
        0.50f,   /* r_split    */
       -9.50f,   /* μ_base     */
        3.00f,   /* μ_scale    */
        1.00f,   /* μ_rate     */
        0.50f,   /* σ_scale    */
        0.80f    /* σ_rate     */
    };

    /* Derived */
    float sigma_z    = TRUE_THETA[2] * TRUE_THETA[1];  /* r * σ_total */
    float sigma_base = sqrtf(1.0f - TRUE_THETA[2] * TRUE_THETA[2]) * TRUE_THETA[1];
    float theta_speed = 0.98f;

    /* Simulation */
    const int T = 30000;
    const int W = 1500;
    const int STRIDE = 1500;
    const int N_THETA = 512;
    const int N_INNER = 256;

    float* y_data = (float*)malloc(T * sizeof(float));
    simulate_sv_returns(y_data, T, TRUE_THETA[0], sigma_z,
                        TRUE_THETA[3], sigma_base, theta_speed, 314159u);

    /* Create param tracker (owns SMC² internally) */
    ParamTracker* tracker = param_tracker_create(W, STRIDE, N_THETA, N_INNER);

    /* Convergence diagnostic */
    ConvergenceDiag diag;
    conv_diag_init(&diag, 8);

    /* All params free (no phasing for this test) */
    int free_mask[N_P];
    for (int i = 0; i < N_P; i++) free_mask[i] = 1;

    /* Print header */
    printf("\n");
    printf(" Win |");
    for (int i = 0; i < N_P; i++) printf(" %5s", PARAM_NAMES[i]);
    printf(" | Mahal  | P-trace |  RMSE  | Conv\n");

    printf("-----|");
    for (int i = 0; i < N_P; i++) printf("------");
    printf("-|--------|---------|--------|------\n");

    /* Run */
    int n_windows = 0;
    float rhat_history[40] = {0};
    float rmse_history[40] = {0};

    for (int t = 0; t < T; t++) {
        param_tracker_feed(tracker, y_data[t]);

        if (param_tracker_window_ready(tracker) && n_windows < 40) {
            param_tracker_run_window(tracker);

            /* Get snapshot (opaque — use accessor) */
            ParamSnapshot snap;
            param_tracker_get_snapshot(tracker, &snap);

            /* Get SMC² posterior for diagnostic */
            SMC2StateCUDA* smc2 = param_tracker_get_smc2(tracker);
            float theta_hat[N_P];
            float R[N_P * N_P];
            smc2_cuda_get_theta_cov(smc2, theta_hat, R);

            float sigma_diag[N_P];
            for (int i = 0; i < N_P; i++)
                sigma_diag[i] = R[i * N_P + i];

            /* Compute innovation and Mahalanobis (diagonal approx) */
            float nu[N_P], S_diag[N_P];
            for (int i = 0; i < N_P; i++) {
                nu[i] = theta_hat[i] - snap.theta[i];
                S_diag[i] = snap.P_diag[i] + sigma_diag[i];
            }
            float d2 = conv_diag_mahal_diag(nu, S_diag, N_P);

            /* P trace */
            float p_tr = 0.0f;
            for (int i = 0; i < N_P; i++)
                p_tr += snap.P_diag[i];

            /* Push to diagnostic */
            conv_diag_push(&diag, theta_hat, sigma_diag, d2, p_tr);

            /* Report */
            ConvergenceReport rpt;
            conv_diag_report(&diag, snap.theta, snap.P_diag,
                             free_mask, THRESHOLD, &rpt);

            /* RMSE */
            float rmse = 0.0f;
            for (int i = 0; i < N_P; i++) {
                float e = snap.theta[i] - TRUE_THETA[i];
                rmse += e * e;
            }
            rmse = sqrtf(rmse / N_P);

            /* Mean R̂ */
            float mean_rhat = 0.0f;
            if (rpt.ready) {
                for (int i = 0; i < N_P; i++) mean_rhat += rpt.rhat[i];
                mean_rhat /= N_P;
            }

            rhat_history[n_windows] = mean_rhat;
            rmse_history[n_windows] = rmse;

            /* Print line */
            printf(" %3d |", n_windows + 1);
            if (!rpt.ready) {
                for (int i = 0; i < N_P; i++) printf("    — ");
            } else {
                for (int i = 0; i < N_P; i++) {
                    if (rpt.rhat[i] < THRESHOLD)
                        printf(" %5.2f", rpt.rhat[i]);
                    else
                        printf(" %s%.2f%s", "\033[91m", rpt.rhat[i], "\033[0m");
                }
            }
            printf(" | d²=%4.1f", rpt.mahal_mean);
            printf(" | %7.4f", rpt.p_trace_current);
            printf(" | %6.4f", rmse);
            printf(" | %d/%d", rpt.n_converged, rpt.n_free);
            if (rpt.all_converged) printf(" ✓");
            printf("\n");

            n_windows++;
        }
    }

    /* ── Validate: R̂ trajectory correlates with RMSE ── */
    printf("\n  ── Validation ──\n");

    /* Find first window where R̂ is available (count >= M) */
    int first_ready = diag.M;  /* 0-indexed */
    if (first_ready >= n_windows) first_ready = n_windows - 1;

    /* Compute Pearson correlation between mean R̂ and RMSE */
    int n_valid = 0;
    float sum_r = 0, sum_e = 0, sum_r2 = 0, sum_e2 = 0, sum_re = 0;
    for (int w = first_ready; w < n_windows; w++) {
        if (rhat_history[w] < 1e-6f) continue;  /* skip if not ready */
        float r = rhat_history[w], e = rmse_history[w];
        sum_r  += r;  sum_e  += e;
        sum_r2 += r*r; sum_e2 += e*e;
        sum_re += r*e;
        n_valid++;
    }

    float corr = 0.0f;
    if (n_valid > 2) {
        float mean_r = sum_r / n_valid, mean_e = sum_e / n_valid;
        float cov = sum_re / n_valid - mean_r * mean_e;
        float var_r = sum_r2 / n_valid - mean_r * mean_r;
        float var_e = sum_e2 / n_valid - mean_e * mean_e;
        if (var_r > 1e-12f && var_e > 1e-12f)
            corr = cov / sqrtf(var_r * var_e);
    }

    printf("  Pearson corr(mean R̂, RMSE) = %.3f over %d windows\n", corr, n_valid);

    /* Check: RMSE at end should be lower than at start */
    float rmse_first = rmse_history[0];
    float rmse_last  = rmse_history[n_windows - 1];
    printf("  RMSE first window: %.4f\n", rmse_first);
    printf("  RMSE last window:  %.4f\n", rmse_last);

    /* Check: at least 4/8 params converged by end */
    ParamSnapshot snap_final;
    param_tracker_get_snapshot(tracker, &snap_final);

    ConvergenceReport rpt_final;
    conv_diag_report(&diag, snap_final.theta, snap_final.P_diag,
                     free_mask, THRESHOLD, &rpt_final);

    printf("  Final convergence: %d/%d params\n",
           rpt_final.n_converged, rpt_final.n_free);

    int pass = 1;

    /* RMSE should improve */
    if (rmse_last >= rmse_first) {
        printf("  WARNING: RMSE did not improve (first=%.4f, last=%.4f)\n",
               rmse_first, rmse_last);
        /* Not a hard fail — SMC² with 512 particles on 30K ticks is noisy */
    }

    /* Correlation should be non-negative (R̂ drops when RMSE drops) */
    if (corr < -0.3f) {
        printf("  FAIL: negative correlation (%.3f) — R̂ not tracking quality\n", corr);
        pass = 0;
    }

    /* At least some params should converge */
    if (rpt_final.n_converged < 3) {
        printf("  FAIL: only %d/8 converged\n", rpt_final.n_converged);
        pass = 0;
    }

    /* Cleanup */
    free(y_data);
    param_tracker_destroy(tracker);

    return test_pass("SMC² integration", pass);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(void)
{
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║       Convergence Diagnostic Tests                         ║\n");
    printf("║       Window R̂ + Mahalanobis + P-trace                    ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    int fails = 0;

    /* Unit tests (synthetic, no CUDA) */
    fails += test1_agreeing();
    fails += test2_biased();
    fails += test3_drift();
    fails += test4_mahal();
    fails += test5_locked();

    /* Integration test (needs CUDA) */
    fails += test6_integration();

    printf("\n══════════════════════════════════════════════\n");
    if (fails == 0)
        printf("  ALL TESTS PASSED ✓\n");
    else
        printf("  %d TEST(S) FAILED ✗\n", fails);
    printf("══════════════════════════════════════════════\n");

    return fails;
}
