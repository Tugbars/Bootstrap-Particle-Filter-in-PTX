/*═══════════════════════════════════════════════════════════════════════════════
 * test_kalman_vs_rm.c — Validate Kalman parameter tracker vs Robbins-Monro
 *
 * Simulates the gradient/Fisher signals that kernel 14 would produce,
 * runs both update rules, compares convergence and adaptation.
 *
 * Build:  gcc -O2 -lm -o test_kalman_vs_rm test_kalman_vs_rm.c
 *    or:  cl /O2 test_kalman_vs_rm.c
 *
 * No GPU needed — pure CPU simulation of the host-side update logic.
 *═══════════════════════════════════════════════════════════════════════════════*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

/* ── Portable RNG (xoshiro128+) ──────────────────────────────────────────── */
static uint32_t rng_s[4] = {0x12345678, 0x9ABCDEF0, 0x13579BDF, 0x2468ACE0};

static uint32_t rotl(uint32_t x, int k) { return (x << k) | (x >> (32 - k)); }

static uint32_t rng_next(void) {
    uint32_t r = rng_s[0] + rng_s[3];
    uint32_t t = rng_s[1] << 9;
    rng_s[2] ^= rng_s[0]; rng_s[3] ^= rng_s[1];
    rng_s[1] ^= rng_s[2]; rng_s[0] ^= rng_s[3];
    rng_s[2] ^= t; rng_s[3] = rotl(rng_s[3], 11);
    return r;
}

static float randf(void) { return (rng_next() >> 8) * (1.0f / 16777216.0f); }

/* Box-Muller */
static float randn(void) {
    float u1 = randf(), u2 = randf();
    if (u1 < 1e-10f) u1 = 1e-10f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

/* ── Synthetic gradient/Fisher generator ─────────────────────────────────
 *
 * Simulates what kernel 14 produces for the μ parameter.
 *
 * At the true parameter θ*, the score (gradient of log-likelihood) is zero
 * in expectation. The Fisher information F is the variance of the score.
 *
 * Model:  mean_grad ~ N(F · (θ* - θ_current), F / K)
 *         mean_fisher ~ F + noise
 *
 * F · (θ* - θ) is the expected gradient when θ ≠ θ* (displaced from truth).
 * F / K is the variance of the mean gradient over K accumulated ticks.
 * ───────────────────────────────────────────────────────────────────────── */

typedef struct {
    float true_mu;         /* True parameter value */
    float fisher_base;     /* Base Fisher information (informative ticks) */
    float fisher_calm;     /* Fisher during calm periods (low signal) */
    int   K;               /* Accumulation window */
} GradSim;

static void grad_sim_query(const GradSim* gs, float current_mu, int tick,
                           float* out_mean_grad, float* out_mean_fisher) {
    /* Fisher varies: high during informative periods, low during calm */
    float F = gs->fisher_base;

    /* Calm periods: Fisher drops (less signal) */
    int period = (tick / 2000) % 4;
    if (period == 1 || period == 3) F = gs->fisher_calm;

    /* Expected gradient = F · (true - current) */
    float expected_grad = F * (gs->true_mu - current_mu);

    /* Noisy observation of mean gradient (variance = F/K) */
    float grad_std = sqrtf(F / (float)gs->K);
    *out_mean_grad = expected_grad + grad_std * randn();

    /* Noisy Fisher estimate (always positive) */
    *out_mean_fisher = F * (1.0f + 0.1f * randn());
    if (*out_mean_fisher < 1e-8f) *out_mean_fisher = 1e-8f;
}

/* ── Robbins-Monro updater ───────────────────────────────────────────────── */

typedef struct {
    float mu;
    int   step;
    float c, t0, gamma;
    float grad_clip;
} RMState;

static void rm_init(RMState* r, float mu0, float c, float t0, float gamma) {
    r->mu = mu0; r->step = 0;
    r->c = c; r->t0 = t0; r->gamma = gamma;
    r->grad_clip = 5.0f;
}

static void rm_update(RMState* r, float mean_g, float mean_f) {
    r->step++;
    float eta = r->c / powf((float)r->step + r->t0, r->gamma);

    float dir = mean_g / fmaxf(mean_f, 1e-8f);
    if (fabsf(dir) > r->grad_clip) dir *= r->grad_clip / fabsf(dir);

    /* SNR gate (the old code's approach) */
    float K_f = 50.0f;  /* assume K=50 */
    float snr = fabsf(mean_g) / fmaxf(sqrtf(mean_f / K_f), 1e-10f);
    float gate = fminf(snr / 2.0f, 1.0f);

    r->mu += eta * gate * dir;
    if (r->mu > 2.0f) r->mu = 2.0f;
    if (r->mu < -5.0f) r->mu = -5.0f;
}

/* ── Kalman updater ──────────────────────────────────────────────────────── */

typedef struct {
    float mu;
    float P;         /* Posterior variance */
    float Q;         /* Process noise (drift rate) */
    float P0;        /* Reset value */
    float grad_clip;
    int   K;
} KalmanState;

static void kalman_init(KalmanState* k, float mu0, float Q, float P0, int K) {
    k->mu = mu0; k->P = P0; k->Q = Q; k->P0 = P0;
    k->grad_clip = 5.0f; k->K = K;
}

static void kalman_update(KalmanState* k, float mean_g, float mean_f) {
    float dir = mean_g / fmaxf(mean_f, 1e-8f);
    if (fabsf(dir) > k->grad_clip) dir *= k->grad_clip / fabsf(dir);

    /* Predict: uncertainty grows */
    float P_pred = k->P + k->Q * (float)k->K;

    /* Observation noise in parameter space */
    float R = 1.0f / fmaxf(mean_f, 1e-8f);

    /* Kalman gain = adaptive step size */
    float K_gain = P_pred / (P_pred + R);

    /* Update */
    k->mu += K_gain * dir;
    if (k->mu > 2.0f) k->mu = 2.0f;
    if (k->mu < -5.0f) k->mu = -5.0f;

    /* Posterior variance shrinks */
    k->P = (1.0f - K_gain) * P_pred;
}

static void kalman_reset(KalmanState* k, float new_mu) {
    k->mu = new_mu;
    k->P = k->P0;
}

/* ── Test harness ────────────────────────────────────────────────────────── */

static void print_header(const char* title) {
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("  %s\n", title);
    printf("═══════════════════════════════════════════════════════════════\n");
}

/*
 * TEST 1: Convergence from misspecified μ
 *
 * Start μ at -2.0, true μ = -1.0. Run 10K updates (500K ticks at K=50).
 * Both methods should converge. Measure time to reach within 0.05 of truth.
 */
static void test_convergence(void) {
    print_header("TEST 1: Convergence from misspecified μ");

    float true_mu = -1.0f;
    float init_mu = -2.0f;
    int K = 50;
    int N_updates = 10000;  /* 500K ticks */

    GradSim gs = { .true_mu = true_mu, .fisher_base = 2.0f,
                   .fisher_calm = 2.0f, .K = K };

    RMState rm;
    rm_init(&rm, init_mu, 0.1f, 10.0f, 0.667f);

    KalmanState kl;
    kalman_init(&kl, init_mu, 5e-8f, 0.01f, K);

    int rm_converged = -1, kl_converged = -1;
    float threshold = 0.05f;

    printf("\n  %8s  %10s %10s  %10s %10s  %8s\n",
           "Update", "RM_mu", "RM_err", "KL_mu", "KL_err", "KL_P");
    printf("  ────────────────────────────────────────────────────────────\n");

    for (int i = 0; i < N_updates; i++) {
        float mg, mf;
        /* Use same noise for both (fair comparison) */
        uint32_t saved[4];
        memcpy(saved, rng_s, sizeof(saved));

        grad_sim_query(&gs, rm.mu, i * K, &mg, &mf);
        rm_update(&rm, mg, mf);

        memcpy(rng_s, saved, sizeof(saved));
        grad_sim_query(&gs, kl.mu, i * K, &mg, &mf);
        kalman_update(&kl, mg, mf);

        /* Advance RNG past the shared draw */
        randn(); randn();

        float rm_err = fabsf(rm.mu - true_mu);
        float kl_err = fabsf(kl.mu - true_mu);

        if (rm_converged < 0 && rm_err < threshold) rm_converged = i;
        if (kl_converged < 0 && kl_err < threshold) kl_converged = i;

        if (i < 20 || i % 500 == 0 || i == N_updates - 1) {
            printf("  %8d  %10.4f %10.4f  %10.4f %10.4f  %8.2e\n",
                   i, rm.mu, rm_err, kl.mu, kl_err, kl.P);
        }
    }

    printf("\n  Convergence (err < %.2f):\n", threshold);
    printf("    RM:     update %d  (tick %d)\n",
           rm_converged, rm_converged >= 0 ? rm_converged * K : -1);
    printf("    Kalman: update %d  (tick %d)\n",
           kl_converged, kl_converged >= 0 ? kl_converged * K : -1);
}

/*
 * TEST 2: Regime shift adaptation
 *
 * Run 5K updates at μ=-1.0, then shift true μ to +0.5.
 * RM step size has decayed by then — it can't adapt.
 * Kalman P keeps growing via Q, so it responds.
 */
static void test_regime_shift(void) {
    print_header("TEST 2: Regime shift (μ jumps at update 5000)");

    float true_mu_1 = -1.0f;
    float true_mu_2 = 0.5f;
    int shift_at = 5000;
    int K = 50;
    int N_updates = 12000;  /* 600K ticks */

    GradSim gs = { .true_mu = true_mu_1, .fisher_base = 2.0f,
                   .fisher_calm = 0.5f, .K = K };

    RMState rm;
    rm_init(&rm, true_mu_1, 0.1f, 10.0f, 0.667f);

    KalmanState kl;
    kalman_init(&kl, true_mu_1, 5e-8f, 0.01f, K);

    /* Accumulate RMSE in windows of 500 updates */
    int window = 500;
    float rm_sse = 0, kl_sse = 0;

    printf("\n  %8s  %10s %10s  %10s %10s  %8s  %8s  %s\n",
           "Update", "RM_mu", "RM_RMSE", "KL_mu", "KL_RMSE", "KL_P", "KL_gain", "true_mu");
    printf("  ──────────────────────────────────────────────────────────────────────\n");

    for (int i = 0; i < N_updates; i++) {
        if (i == shift_at) gs.true_mu = true_mu_2;

        float mg, mf;
        uint32_t saved[4];
        memcpy(saved, rng_s, sizeof(saved));

        grad_sim_query(&gs, rm.mu, i * K, &mg, &mf);
        rm_update(&rm, mg, mf);

        memcpy(rng_s, saved, sizeof(saved));
        grad_sim_query(&gs, kl.mu, i * K, &mg, &mf);

        /* Capture Kalman gain for display */
        float P_pred = kl.P + kl.Q * (float)K;
        float R = 1.0f / fmaxf(mf, 1e-8f);
        float K_gain_disp = P_pred / (P_pred + R);

        kalman_update(&kl, mg, mf);
        randn(); randn();

        float rm_e = rm.mu - gs.true_mu;
        float kl_e = kl.mu - gs.true_mu;
        rm_sse += rm_e * rm_e;
        kl_sse += kl_e * kl_e;

        if ((i + 1) % window == 0) {
            float rm_rmse = sqrtf(rm_sse / window);
            float kl_rmse = sqrtf(kl_sse / window);
            printf("  %8d  %10.4f %10.4f  %10.4f %10.4f  %8.2e  %8.5f  %.2f\n",
                   i, rm.mu, rm_rmse, kl.mu, kl_rmse, kl.P, K_gain_disp, gs.true_mu);
            rm_sse = 0; kl_sse = 0;
        }
    }

    printf("\n  Final errors (after shift):\n");
    printf("    RM:     %.4f  (err = %.4f)\n", rm.mu, fabsf(rm.mu - true_mu_2));
    printf("    Kalman: %.4f  (err = %.4f)\n", kl.mu, fabsf(kl.mu - true_mu_2));
}

/*
 * TEST 3: Long-run stability with calm periods
 *
 * Run 50K updates (2.5M ticks). Fisher drops to near-zero during calm.
 * RM step decays regardless. Kalman gain drops naturally during calm
 * (R → ∞ when Fisher → 0) but P accumulates, ready for next informative period.
 */
static void test_long_run(void) {
    print_header("TEST 3: Long-run stability (50K updates, varying Fisher)");

    float true_mu = -1.0f;
    int K = 50;
    int N_updates = 50000;

    GradSim gs = { .true_mu = true_mu, .fisher_base = 2.0f,
                   .fisher_calm = 0.01f, .K = K };  /* Fisher drops 200x in calm */

    RMState rm;
    rm_init(&rm, true_mu - 0.3f, 0.1f, 10.0f, 0.667f);

    KalmanState kl;
    kalman_init(&kl, true_mu - 0.3f, 5e-8f, 0.01f, K);

    int window = 2000;
    float rm_sse = 0, kl_sse = 0;

    printf("\n  %8s  %10s %10s  %10s %10s  %8s  %10s  %s\n",
           "Update", "RM_mu", "RM_RMSE", "KL_mu", "KL_RMSE", "KL_P",
           "RM_eta", "regime");
    printf("  ──────────────────────────────────────────────────────────────────────────\n");

    for (int i = 0; i < N_updates; i++) {
        float mg, mf;
        uint32_t saved[4];
        memcpy(saved, rng_s, sizeof(saved));

        grad_sim_query(&gs, rm.mu, i * K, &mg, &mf);

        /* Capture RM eta for display */
        float rm_eta = rm.c / powf((float)(rm.step + 1) + rm.t0, rm.gamma);

        rm_update(&rm, mg, mf);

        memcpy(rng_s, saved, sizeof(saved));
        grad_sim_query(&gs, kl.mu, i * K, &mg, &mf);
        kalman_update(&kl, mg, mf);
        randn(); randn();

        float rm_e = rm.mu - true_mu;
        float kl_e = kl.mu - true_mu;
        rm_sse += rm_e * rm_e;
        kl_sse += kl_e * kl_e;

        if ((i + 1) % window == 0) {
            float rm_rmse = sqrtf(rm_sse / window);
            float kl_rmse = sqrtf(kl_sse / window);
            int period = (i / 2000) % 4;
            const char* regime = (period == 1 || period == 3) ? "CALM" : "ACTIVE";
            printf("  %8d  %10.4f %10.4f  %10.4f %10.4f  %8.2e  %10.6f  %s\n",
                   i, rm.mu, rm_rmse, kl.mu, kl_rmse, kl.P, rm_eta, regime);
            rm_sse = 0; kl_sse = 0;
        }
    }

    printf("\n  Final state at 50K updates (2.5M ticks):\n");
    float rm_eta_final = rm.c / powf((float)rm.step + rm.t0, rm.gamma);
    printf("    RM:     mu=%.4f  eta=%.8f  (essentially frozen)\n", rm.mu, rm_eta_final);
    printf("    Kalman: mu=%.4f  P=%.2e  (still adaptive)\n", kl.mu, kl.P);
}

/*
 * TEST 4: Regime shift AFTER long convergence (the killer test)
 *
 * 30K updates converged, then μ shifts. RM is stiff. Kalman adapts.
 */
static void test_late_shift(void) {
    print_header("TEST 4: Late regime shift (shift at update 30000)");

    float true_mu_1 = -1.0f;
    float true_mu_2 = -0.3f;
    int shift_at = 30000;
    int K = 50;
    int N_updates = 40000;

    GradSim gs = { .true_mu = true_mu_1, .fisher_base = 2.0f,
                   .fisher_calm = 2.0f, .K = K };

    RMState rm;
    rm_init(&rm, true_mu_1, 0.1f, 10.0f, 0.667f);

    KalmanState kl;
    kalman_init(&kl, true_mu_1, 5e-8f, 0.01f, K);

    int window = 1000;
    float rm_sse = 0, kl_sse = 0;

    printf("\n  %8s  %10s %10s  %10s %10s  %8s  %10s\n",
           "Update", "RM_mu", "RM_RMSE", "KL_mu", "KL_RMSE", "KL_P", "RM_eta");
    printf("  ────────────────────────────────────────────────────────────────────\n");

    for (int i = 0; i < N_updates; i++) {
        if (i == shift_at) {
            gs.true_mu = true_mu_2;
            printf("  >>>>>> REGIME SHIFT: true_mu = %.2f → %.2f <<<<<<\n",
                   true_mu_1, true_mu_2);
        }

        float mg, mf;
        uint32_t saved[4];
        memcpy(saved, rng_s, sizeof(saved));

        grad_sim_query(&gs, rm.mu, i * K, &mg, &mf);
        float rm_eta = rm.c / powf((float)(rm.step + 1) + rm.t0, rm.gamma);
        rm_update(&rm, mg, mf);

        memcpy(rng_s, saved, sizeof(saved));
        grad_sim_query(&gs, kl.mu, i * K, &mg, &mf);
        kalman_update(&kl, mg, mf);
        randn(); randn();

        float rm_e = rm.mu - gs.true_mu;
        float kl_e = kl.mu - gs.true_mu;
        rm_sse += rm_e * rm_e;
        kl_sse += kl_e * kl_e;

        if ((i + 1) % window == 0) {
            float rm_rmse = sqrtf(rm_sse / window);
            float kl_rmse = sqrtf(kl_sse / window);
            printf("  %8d  %10.4f %10.4f  %10.4f %10.4f  %8.2e  %10.6f\n",
                   i, rm.mu, rm_rmse, kl.mu, kl_rmse, kl.P, rm_eta);
            rm_sse = 0; kl_sse = 0;
        }
    }

    printf("\n  Final (10K updates after shift):\n");
    printf("    RM:     mu=%.4f  (err = %.4f, step size dead)\n",
           rm.mu, fabsf(rm.mu - true_mu_2));
    printf("    Kalman: mu=%.4f  (err = %.4f, P still alive)\n",
           kl.mu, fabsf(kl.mu - true_mu_2));
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(void) {
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║   Kalman vs Robbins-Monro Parameter Learning Comparison     ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");

    test_convergence();
    test_regime_shift();
    test_long_run();
    test_late_shift();

    printf("\n══════════════════════════════════════════════════════════════\n");
    printf("  DONE. Key metrics to check:\n");
    printf("  • Test 1: Kalman converges at least as fast as RM\n");
    printf("  • Test 2: Kalman tracks shift, RM doesn't\n");
    printf("  • Test 3: RM eta → 0 (frozen), Kalman P stays alive\n");
    printf("  • Test 4: THE KILLER — RM stuck after late shift, Kalman adapts\n");
    printf("══════════════════════════════════════════════════════════════\n");

    return 0;
}
