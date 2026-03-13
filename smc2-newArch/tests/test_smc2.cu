/**
 * @file test_smc2.cu
 * @brief Smoke test for SMC² engine + param tracker
 *
 * Exercises the full stack:
 *   1. Alloc + configure SMC² engine
 *   2. Init from prior
 *   3. Feed synthetic DGP observations
 *   4. Query posterior + z-range
 *   5. Param tracker with phased learning
 *   6. Free everything
 *
 * Build: make test SM=90
 *   or:  nvcc -O2 -arch=sm_90 test/test_smc2.cu -o build/bin/test_smc2 \
 *         -Lbuild/lib -lsmc2 -lcurand -I.
 */

#include "smc2_engine.cuh"
#include "smc2_param_tracker.cuh"
#include "smc2_pipeline.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* ── Simple DGP ─────────────────────────────────────────────────────────── */

typedef struct {
    float rho;           /* z̃ persistence */
    float sigma_z;       /* z̃ innovation scale */
    float mu_base;       /* μ(z) base */
    float mu_scale;      /* μ(z) scale */
    float mu_rate;       /* μ(z) rate */
    float sigma_base;    /* σ_h(z) base */
    float sigma_scale;   /* σ_h(z) scale */
    float sigma_rate;    /* σ_h(z) rate */
    float theta_base;    /* θ(z) base */
    float theta_scale;   /* θ(z) scale */
    float theta_rate;    /* θ(z) rate */
    /* State */
    float z_tilde;
    float h;
} DGP;

static float sat_exp(float base, float scale, float rate, float z) {
    return base + scale * (1.0f - expf(-rate * z));
}

static float randn(void) {
    /* Box-Muller */
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

static void dgp_init(DGP* d) {
    d->rho          = 0.96f;
    d->sigma_z      = 0.10f;
    d->mu_base      = -4.5f;
    d->mu_scale     = 2.0f;
    d->mu_rate      = 0.5f;
    d->sigma_base   = 0.08f;
    d->sigma_scale  = 0.3f;
    d->sigma_rate   = 0.5f;
    d->theta_base   = 0.02f;
    d->theta_scale  = 0.08f;
    d->theta_rate   = 1.5f;
    d->z_tilde      = 0.0f;
    d->h            = d->mu_base;
}

/**
 * @brief Generate one observation using RBPF-consistent dynamics.
 * Returns y = log(r²) for OCSN parameterization.
 */
static float dgp_step(DGP* d) {
    /* z̃ dynamics */
    d->z_tilde = d->rho * d->z_tilde + d->sigma_z * randn();
    float z = 1.5f * (1.0f + tanhf(d->z_tilde));

    /* h dynamics — θ(z)-based mean reversion, matching RBPF */
    float theta_z = sat_exp(d->theta_base, d->theta_scale, d->theta_rate, z);
    float mu_z    = sat_exp(d->mu_base, d->mu_scale, d->mu_rate, z);
    float sigma_h = sat_exp(d->sigma_base, d->sigma_scale, d->sigma_rate, z);
    float phi = 1.0f - theta_z;

    d->h = phi * d->h + theta_z * mu_z + sigma_h * randn();

    /* Observation: r = exp(h/2) * ε, then y = log(r²) */
    float r = expf(d->h / 2.0f) * randn();
    float y = logf(r * r + 1e-20f);

    return y;
}

/* ── Test helpers ───────────────────────────────────────────────────────── */

static int g_passed = 0;
static int g_failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_passed++; printf("  [PASS] %s\n", msg); } \
    else { g_failed++; printf("  [FAIL] %s\n", msg); } \
} while(0)

#define CHECK_FINITE(val, msg) CHECK(isfinite(val), msg)

/* ══════════════════════════════════════════════════════════════════════════ */

static void test_engine_lifecycle(void) {
    printf("\n=== Test: Engine Lifecycle ===\n");

    SMC2StateCUDA* s = smc2_cuda_alloc(128, 128);
    CHECK(s != NULL, "smc2_cuda_alloc(128,128)");

    CHECK(s->N_theta == 128, "N_theta == 128");
    CHECK(s->N_inner == 128, "N_inner == 128");

    /* Verify defaults came from named functions */
    CHECK(s->prior.rho_mean == 0.95f, "Default prior rho_mean = 0.95");
    CHECK(s->bounds.rho_min == 0.8f, "Default bound rho_min = 0.8");

    smc2_cuda_init_from_prior(s);

    /* Query posterior — should be near prior */
    float theta[N_PARAMS];
    smc2_cuda_get_theta_mean(s, theta);
    CHECK_FINITE(theta[0], "theta[rho] is finite after init");
    CHECK_FINITE(theta[3], "theta[mu_base] is finite after init");

    float z = smc2_cuda_get_z_mean(s);
    CHECK_FINITE(z, "z_mean is finite after init");
    CHECK(z >= 0.0f && z <= 3.0f, "z_mean in [0,3]");

    smc2_cuda_free(s);
    printf("  Engine lifecycle OK\n");
}

static void test_engine_update(void) {
    printf("\n=== Test: Engine Update (100 ticks) ===\n");

    SMC2StateCUDA* s = smc2_cuda_alloc(64, 128);
    smc2_cuda_set_fixed_lag(s, 50);
    smc2_cuda_init_from_prior(s);

    DGP dgp;
    dgp_init(&dgp);

    float last_ess = 0;
    for (int t = 0; t < 100; t++) {
        float y = dgp_step(&dgp);
        last_ess = smc2_cuda_update(s, y);
    }

    CHECK_FINITE(last_ess, "ESS finite after 100 ticks");
    CHECK(last_ess > 0, "ESS > 0");

    float theta[N_PARAMS], theta_std[N_PARAMS];
    smc2_cuda_get_theta_mean(s, theta);
    smc2_cuda_get_theta_std(s, theta_std);

    for (int i = 0; i < N_PARAMS; i++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "theta[%d] finite after 100 ticks", i);
        CHECK_FINITE(theta[i], buf);
    }

    SMC2Diagnostics diag;
    smc2_cuda_get_diagnostics(s, &diag);
    printf("  Diagnostics: resamples=%d accept=%.1f%% ess=%.1f\n",
           diag.n_resamples, diag.accept_rate * 100.0f, diag.outer_ess);

    smc2_cuda_free(s);
}

static void test_batch_update(void) {
    printf("\n=== Test: Batch Update (500 ticks) ===\n");

    SMC2StateCUDA* s = smc2_cuda_alloc(64, 128);
    smc2_cuda_set_fixed_lag(s, 50);
    smc2_cuda_init_from_prior(s);

    DGP dgp;
    dgp_init(&dgp);

    float* batch = (float*)malloc(500 * sizeof(float));
    for (int i = 0; i < 500; i++) batch[i] = dgp_step(&dgp);

    float ess = smc2_cuda_update_batch(s, batch, 500);
    CHECK_FINITE(ess, "Batch ESS finite");
    CHECK(ess > 0, "Batch ESS > 0");

    float theta[N_PARAMS];
    smc2_cuda_get_theta_mean(s, theta);
    printf("  Posterior after 500 batch ticks:\n");
    printf("    rho=%.4f  sigma_total=%.4f  r_split=%.4f  mu_base=%.4f\n",
           theta[0], theta[1], theta[2], theta[3]);

    free(batch);
    smc2_cuda_free(s);
}

static void test_param_tracker(void) {
    printf("\n=== Test: Param Tracker + Phased Learning ===\n");

    ParamTracker* pt = param_tracker_create(500, 250, 64, 128);
    CHECK(pt != NULL, "param_tracker_create");
    CHECK(param_tracker_get_phase(pt) == PHASE_1_FLOORS, "Initial phase = 1 (FLOORS)");

    DGP dgp;
    dgp_init(&dgp);

    /* Feed 500 observations */
    for (int i = 0; i < 500; i++) {
        float y = dgp_step(&dgp);
        param_tracker_feed(pt, y);
    }

    CHECK(param_tracker_window_ready(pt), "Window ready after 500 ticks");

    /* Run first window */
    param_tracker_run_window(pt);

    ParamSnapshot snap;
    param_tracker_get_snapshot(pt, &snap);
    CHECK(snap.n_updates == 1, "n_updates == 1 after first window");
    CHECK_FINITE(snap.mu, "snap.mu finite");
    CHECK_FINITE(snap.sigma_h, "snap.sigma_h finite");
    CHECK_FINITE(snap.z_mean, "snap.z_mean finite");
    CHECK(snap.z_mean >= 0 && snap.z_mean <= 3.0f, "snap.z_mean in [0,3]");

    printf("  Snapshot: mu=%.4f sigma_h=%.4f z=%.3f phase=%d\n",
           snap.mu, snap.sigma_h, snap.z_mean, param_tracker_get_phase(pt));

    /* Check convergence state */
    int conv[N_PARAMS];
    param_tracker_get_converged(pt, conv);
    printf("  Converged: ");
    for (int i = 0; i < N_PARAMS; i++) printf("%d ", conv[i]);
    printf("\n");

    param_tracker_print(pt);

    param_tracker_destroy(pt);
}

static void test_warm_init(void) {
    printf("\n=== Test: Warm Init ===\n");

    SMC2StateCUDA* s = smc2_cuda_alloc(64, 128);

    float warm_mean[N_PARAMS] = {0.95f, 0.20f, 0.5f, -4.0f, 1.5f, 0.5f, 0.3f, 0.5f};
    float warm_cov[N_PARAMS * N_PARAMS];
    memset(warm_cov, 0, sizeof(warm_cov));
    for (int i = 0; i < N_PARAMS; i++) warm_cov[i * N_PARAMS + i] = 0.01f;

    smc2_cuda_init_warm(s, warm_mean, warm_cov);

    float theta[N_PARAMS];
    smc2_cuda_get_theta_mean(s, theta);

    /* After warm init, means should be near warm_mean (not prior default) */
    float rho_err = fabsf(theta[0] - warm_mean[0]);
    CHECK(rho_err < 0.1f, "Warm init rho near warm_mean");

    float mu_err = fabsf(theta[3] - warm_mean[3]);
    CHECK(mu_err < 1.0f, "Warm init mu_base near warm_mean");

    printf("  Warm init posterior: rho=%.4f mu_base=%.4f (err: %.4f, %.4f)\n",
           theta[0], theta[3], rho_err, mu_err);

    smc2_cuda_free(s);
}

static void test_fixed_params(void) {
    printf("\n=== Test: Fixed Params (4-param mode) ===\n");

    SMC2StateCUDA* s = smc2_cuda_alloc(64, 128);

    /* Fix curve params, learn only floors */
    uint8_t mask[N_PARAMS] = {0, 0, 0, 0, 1, 1, 1, 1};
    float values[N_PARAMS] = {0, 0, 0, 0, 2.0f, 0.5f, 0.3f, 0.5f};
    smc2_cuda_set_fixed_params(s, mask, values);

    smc2_cuda_init_from_prior(s);

    DGP dgp;
    dgp_init(&dgp);

    float batch[200];
    for (int i = 0; i < 200; i++) batch[i] = dgp_step(&dgp);
    smc2_cuda_update_batch(s, batch, 200);

    float theta[N_PARAMS];
    smc2_cuda_get_theta_mean(s, theta);

    /* Fixed params should be exactly at their fixed values */
    CHECK(fabsf(theta[4] - 2.0f) < 0.01f, "Fixed mu_scale = 2.0");
    CHECK(fabsf(theta[5] - 0.5f) < 0.01f, "Fixed mu_rate = 0.5");
    CHECK(fabsf(theta[6] - 0.3f) < 0.01f, "Fixed sigma_scale = 0.3");
    CHECK(fabsf(theta[7] - 0.5f) < 0.01f, "Fixed sigma_rate = 0.5");

    /* Free params should have moved from prior */
    CHECK_FINITE(theta[0], "Free rho finite in 4-param mode");
    CHECK_FINITE(theta[3], "Free mu_base finite in 4-param mode");

    printf("  4-param mode: rho=%.4f mu_base=%.4f (free), "
           "mu_scale=%.4f mu_rate=%.4f (fixed)\n",
           theta[0], theta[3], theta[4], theta[5]);

    smc2_cuda_free(s);
}

static void test_convergence_diag(void) {
    printf("\n=== Test: Convergence Diagnostics ===\n");

    ConvergenceDiag cd;
    conv_diag_init(&cd, 4);

    /* Push 4 synthetic window results */
    for (int k = 0; k < 4; k++) {
        float theta[CONV_DIAG_N_PARAMS];
        float sigma[CONV_DIAG_N_PARAMS];
        for (int i = 0; i < CONV_DIAG_N_PARAMS; i++) {
            theta[i] = 1.0f + 0.01f * k + 0.001f * i;  /* Slight variation */
            sigma[i] = 0.1f;
        }
        conv_diag_push(&cd, theta, sigma, (float)k * 0.5f, 10.0f - k);
    }

    float kalman_x[CONV_DIAG_N_PARAMS], kalman_P[CONV_DIAG_N_PARAMS];
    int free_mask[CONV_DIAG_N_PARAMS];
    for (int i = 0; i < CONV_DIAG_N_PARAMS; i++) {
        kalman_x[i] = 1.02f;
        kalman_P[i] = 0.01f;
        free_mask[i] = 1;
    }

    ConvergenceReport rpt;
    conv_diag_report(&cd, kalman_x, kalman_P, free_mask, 1.5f, &rpt);

    CHECK(rpt.ready == 1, "Report ready after 4 pushes (M=4)");
    CHECK(rpt.n_free == CONV_DIAG_N_PARAMS, "All params free");
    CHECK_FINITE(rpt.rhat[0], "R-hat[0] finite");
    CHECK_FINITE(rpt.mahal_mean, "Mahalanobis mean finite");

    printf("  R-hat[0]=%.3f  converged=%d/%d  mahal=%.2f\n",
           rpt.rhat[0], rpt.n_converged, rpt.n_free, rpt.mahal_mean);
}

/* ══════════════════════════════════════════════════════════════════════════ */

int main(void) {
    srand((unsigned)time(NULL));

    printf("SMC2 Test Suite\n");
    printf("  Noise precision: %s\n", noise_precision_str());

    test_engine_lifecycle();
    test_engine_update();
    test_batch_update();
    test_warm_init();
    test_fixed_params();
    test_convergence_diag();
    test_param_tracker();

    printf("\n══════════════════════════════════════\n");
    printf("  Results: %d passed, %d failed\n", g_passed, g_failed);
    printf("══════════════════════════════════════\n");

    return (g_failed > 0) ? 1 : 0;
}
