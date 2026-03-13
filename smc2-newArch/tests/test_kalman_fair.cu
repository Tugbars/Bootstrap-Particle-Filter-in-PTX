/*═══════════════════════════════════════════════════════════════════════════════
 * @file test_kalman_fair.cu
 * @brief Fair comparison: traveling vs amnesiac cloud × with/without Kalman
 *
 * Four modes on identical data:
 *
 *   A. TRAVEL_RAW    — Cloud carries forward, push mu_base directly.
 *                      No Kalman. Current production winner.
 *
 *   B. TRAVEL_KALMAN — Cloud carries forward, Kalman smooths posterior means,
 *                      push Kalman's theta[MU_BASE] (NOT snap.mu / eval_curve).
 *
 *   C. AMNESIA_KALMAN — Cloud reinit from prior each window. Kalman stitches
 *                       independent posteriors. Push Kalman's theta[MU_BASE].
 *                       Fixed version of the original (broken) tracker.
 *
 *   D. AMNESIA_RAW   — Cloud reinit from prior each window. Push raw mu_base.
 *                      Baseline: no memory at all.
 *
 * This isolates two independent questions:
 *   1. Does the traveling cloud beat the amnesiac cloud? (compare A vs D, B vs C)
 *   2. Does the Kalman help? (compare A vs B, C vs D)
 *
 * DGP: Multi-regime with learning phase then scored crisis.
 * All modes use RAW mu_base push — no eval_curve anywhere.
 *
 * Build (via cmake):
 *   cmake --build build --target test_kalman_fair
 * Or standalone:
 *   nvcc -O3 test_kalman_fair.cu ../smc2_engine.cu -I.. -I../../gpu_bpf \
 *        -o test_kalman_fair -lcurand -arch=sm_120
 *═══════════════════════════════════════════════════════════════════════════════*/

#include "gpu_bpf_full.cuh"
#include "smc2_engine.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

/* ── PRNG ────────────────────────────────────────────────────────────────── */

struct PRNG { unsigned int state; };

static inline PRNG prng_create(unsigned int seed) {
    PRNG p; p.state = seed ? seed : 1; return p;
}

static inline float prng_randf(PRNG* p) {
    p->state = p->state * 1103515245 + 12345;
    return (float)((p->state >> 16) & 0x7FFF) / 32768.0f;
}

static inline float prng_randn(PRNG* p) {
    float u1 = prng_randf(p) + 1e-10f;
    float u2 = prng_randf(p);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

/* ── DGP ─────────────────────────────────────────────────────────────────── */

struct TrueDGP {
    float rho, sigma_z;                               /* z̃ dynamics only */
    float mu_base, mu_scale, mu_rate;                  /* μ(z) curve */
    float sigma_h_base, sigma_h_scale, sigma_h_rate;   /* σ_h(z) curve */
    float theta_base, theta_scale, theta_rate;          /* θ(z) mean-reversion speed */
};

static inline float sat_exp(float base, float scale, float rate, float z) {
    return base + scale * (1.0f - expf(-rate * z));
}

static TrueDGP default_dgp() {
    TrueDGP d;
    d.rho = 0.98f;  d.sigma_z = 0.15f;
    d.mu_base = -4.5f;  d.mu_scale = 3.0f;  d.mu_rate = 0.5f;
    d.sigma_h_base = 0.10f;  d.sigma_h_scale = 0.50f;  d.sigma_h_rate = 0.30f;
    /* θ(z) curve — matches smc2->theta_curve for model consistency.
     * h dynamics: h = (1-θ(z))·h + θ(z)·μ(z) + σ_h(z)·ε
     * At z=0: θ=0.02 → φ=0.98 (slow mean-reversion)
     * At z=3: θ≈0.10 → φ=0.90 (faster under stress) */
    d.theta_base = 0.02f;  d.theta_scale = 0.08f;  d.theta_rate = 1.5f;
    return d;
}

/* ── Segments ────────────────────────────────────────────────────────────── */

struct Segment {
    const char* name;
    int ticks;
    float z_bias;
    int score;
};

struct GeneratedData {
    std::vector<float> returns;        /* raw r_t — for production BPF */
    std::vector<float> log_returns_sq; /* log(r_t²) — for SMC²/RBPF (OCSN) */
    std::vector<float> true_h, true_z;
    std::vector<int> segment_starts;
    std::vector<const char*> segment_names;
    std::vector<int> segment_score;
    int N, score_start;
};

static GeneratedData generate_data(const TrueDGP& dgp,
                                    const std::vector<Segment>& segments,
                                    PRNG* rng) {
    GeneratedData gd;
    gd.score_start = -1;
    float z_tilde = 0.0f, h = dgp.mu_base;

    for (size_t s = 0; s < segments.size(); s++) {
        const Segment& seg = segments[s];
        gd.segment_starts.push_back((int)gd.returns.size());
        gd.segment_names.push_back(seg.name);
        gd.segment_score.push_back(seg.score);
        if (seg.score && gd.score_start < 0)
            gd.score_start = (int)gd.returns.size();

        for (int t = 0; t < seg.ticks; t++) {
            /* z̃ dynamics: AR(1) with constant ρ */
            z_tilde = dgp.rho * z_tilde + dgp.sigma_z * prng_randn(rng)
                      + seg.z_bias * (1.0f - dgp.rho);
            float z = 1.5f * (1.0f + tanhf(z_tilde));

            /* Curves evaluated at z */
            float mu_z    = sat_exp(dgp.mu_base, dgp.mu_scale, dgp.mu_rate, z);
            float sigma_h = sat_exp(dgp.sigma_h_base, dgp.sigma_h_scale, dgp.sigma_h_rate, z);
            float theta_z = sat_exp(dgp.theta_base, dgp.theta_scale, dgp.theta_rate, z);

            /* h dynamics: matches RBPF inner model exactly
             * h = (1-θ(z))·h + θ(z)·μ(z) + σ_h(z)·ε */
            float phi = 1.0f - theta_z;
            h = phi * h + theta_z * mu_z + sigma_h * prng_randn(rng);

            float y = expf(h * 0.5f) * prng_randn(rng);
            gd.returns.push_back(y);
            gd.log_returns_sq.push_back(logf(y * y + 1e-20f));
            gd.true_h.push_back(h);
            gd.true_z.push_back(z);
        }
    }
    gd.N = (int)gd.returns.size();
    return gd;
}

/* ── Minimal 8×8 Kalman (host-side, from param_tracker) ─────────────────── */

struct KalmanState {
    float x[N_PARAMS];
    float P[N_PARAMS * N_PARAMS];
    float Q[N_PARAMS * N_PARAMS];
    int   initialized;
};

static KalmanState kalman_create() {
    KalmanState k;
    memset(&k, 0, sizeof(k));
    for (int i = 0; i < N_PARAMS; i++)
        k.P[i * N_PARAMS + i] = 10.0f;  /* Wide initial uncertainty */

    /* Default Q: per-window drift */
    k.Q[0 * N_PARAMS + 0] = 1e-5f;   /* rho */
    k.Q[1 * N_PARAMS + 1] = 1e-5f;   /* sigma_total */
    k.Q[2 * N_PARAMS + 2] = 1e-5f;   /* r_split */
    k.Q[3 * N_PARAMS + 3] = 1e-3f;   /* mu_base */
    k.Q[4 * N_PARAMS + 4] = 1e-7f;   /* mu_scale */
    k.Q[5 * N_PARAMS + 5] = 1e-7f;   /* mu_rate */
    k.Q[6 * N_PARAMS + 6] = 1e-7f;   /* sigma_scale */
    k.Q[7 * N_PARAMS + 7] = 1e-7f;   /* sigma_rate */
    return k;
}

static void kalman_update(KalmanState* k, const float* z_meas, const float* R) {
    const int D = N_PARAMS;

    if (!k->initialized) {
        memcpy(k->x, z_meas, D * sizeof(float));
        memcpy(k->P, R, D * D * sizeof(float));
        for (int i = 0; i < D; i++)
            k->P[i * D + i] += 1e-6f;
        k->initialized = 1;
        return;
    }

    /* Predict: P_bar = P + Q */
    float P_bar[D * D];
    for (int i = 0; i < D * D; i++) P_bar[i] = k->P[i] + k->Q[i];

    /* Innovation covariance: S = P_bar + R */
    float S[D * D];
    for (int i = 0; i < D * D; i++) S[i] = P_bar[i] + R[i];

    /* Regularize diagonal */
    for (int i = 0; i < D; i++) S[i * D + i] += 1e-8f;

    /* Invert S using diagonal approximation (good enough for 8×8 with
     * dominant diagonal from posterior covariance R) */
    float S_inv_diag[D];
    for (int i = 0; i < D; i++)
        S_inv_diag[i] = 1.0f / S[i * D + i];

    /* Kalman gain (diagonal approximation): K_ii = P_bar_ii / S_ii */
    float K_diag[D];
    for (int i = 0; i < D; i++)
        K_diag[i] = P_bar[i * D + i] * S_inv_diag[i];

    /* Update state: x += K * (z - x) */
    for (int i = 0; i < D; i++)
        k->x[i] += K_diag[i] * (z_meas[i] - k->x[i]);

    /* Update covariance: P = (1 - K) * P_bar (diagonal approx) */
    for (int i = 0; i < D; i++)
        for (int j = 0; j < D; j++)
            k->P[i * D + j] = (1.0f - K_diag[i]) * P_bar[i * D + j];

    /* P floor */
    float P_floor[D] = {1e-4f, 1e-4f, 1e-3f, 1e-2f, 1e-3f, 1e-3f, 1e-4f, 1e-3f};
    for (int i = 0; i < D; i++)
        if (k->P[i * D + i] < P_floor[i])
            k->P[i * D + i] = P_floor[i];
}

/* ── Test modes ──────────────────────────────────────────────────────────── */

enum TestMode {
    MODE_TRAVEL_RAW,
    MODE_TRAVEL_KALMAN,
    MODE_AMNESIA_KALMAN,
    MODE_AMNESIA_RAW
};

static const char* mode_name(TestMode m) {
    switch (m) {
        case MODE_TRAVEL_RAW:     return "TRAVEL + RAW";
        case MODE_TRAVEL_KALMAN:  return "TRAVEL + KALMAN";
        case MODE_AMNESIA_KALMAN: return "AMNESIA + KALMAN";
        case MODE_AMNESIA_RAW:    return "AMNESIA + RAW";
    }
    return "?";
}

/* ── Run result ──────────────────────────────────────────────────────────── */

struct RunResult {
    std::vector<float> est_h;
    double rmse_total;
    double rmse_scored;
    int    n_pushes;
    float  last_mu_pushed;
};

/* ── Run one mode ───────────────────────────────────────────────────────── */

static RunResult run_mode(
    const GeneratedData& gd,
    const TrueDGP& dgp,
    int n_bpf, int n_theta, int n_inner,
    int window_size, int stride,
    TestMode mode,
    unsigned int bpf_seed
) {
    RunResult result;
    int N = gd.N;
    result.est_h.resize(N, 0.0f);
    result.n_pushes = 0;
    result.last_mu_pushed = dgp.mu_base;

    int traveling = (mode == MODE_TRAVEL_RAW || mode == MODE_TRAVEL_KALMAN);
    int use_kalman = (mode == MODE_TRAVEL_KALMAN || mode == MODE_AMNESIA_KALMAN);

    /* ── Create BPF ──────────────────────────────────────────────────── */
    GpuBpfState* bpf = gpu_bpf_create(n_bpf, dgp.rho, dgp.sigma_z, dgp.mu_base,
                                        0.0f, 0.0f, bpf_seed);
    gpu_bpf_disable_mu_learning(bpf);
    gpu_bpf_enable_rho_learning(bpf, 0);

    /* ── Create SMC² ─────────────────────────────────────────────────── */
    SMC2StateCUDA* smc2 = smc2_cuda_alloc(n_theta, n_inner);

    /* Priors centered on truth */
    smc2->prior.rho_mean         = dgp.rho;         smc2->prior.rho_std         = 0.05f;
    smc2->prior.sigma_total_mean = dgp.sigma_z;     smc2->prior.sigma_total_std = 0.1f;
    smc2->prior.r_split_mean     = 0.5f;             smc2->prior.r_split_std     = 0.2f;
    smc2->prior.mu_base_mean     = dgp.mu_base;     smc2->prior.mu_base_std     = 1.0f;
    smc2->prior.mu_scale_mean    = dgp.mu_scale;    smc2->prior.mu_scale_std    = 1.5f;
    smc2->prior.mu_rate_mean     = dgp.mu_rate;     smc2->prior.mu_rate_std     = 0.3f;
    smc2->prior.sigma_scale_mean = dgp.sigma_h_scale;
    smc2->prior.sigma_scale_std  = 0.3f;
    smc2->prior.sigma_rate_mean  = dgp.sigma_h_rate;
    smc2->prior.sigma_rate_std   = 0.2f;

    /* θ(z) curve must match DGP exactly */
    smc2->theta_curve.base  = dgp.theta_base;
    smc2->theta_curve.scale = dgp.theta_scale;
    smc2->theta_curve.rate  = dgp.theta_rate;

    /* Fix curve shape params — same as previous tests */
    {
        uint8_t mask[N_PARAMS] = {0};
        float   vals[N_PARAMS] = {0};
        mask[4] = 1; vals[4] = dgp.mu_scale;       /* mu_scale */
        mask[5] = 1; vals[5] = dgp.mu_rate;         /* mu_rate */
        mask[6] = 1; vals[6] = dgp.sigma_h_scale;   /* sigma_scale */
        mask[7] = 1; vals[7] = dgp.sigma_h_rate;    /* sigma_rate */
        smc2_cuda_set_fixed_params(smc2, mask, vals);
    }

    /* Initialize cloud once for all modes */
    smc2_cuda_init_from_prior(smc2);

    /* ── Create Kalman (only used if use_kalman) ─────────────────────── */
    KalmanState kalman = kalman_create();

    /* ── Observation buffer (log(r²) for SMC²/RBPF/OCSN) ────────────── */
    std::vector<float> obs_buffer;
    obs_buffer.reserve(N);

    /* ── Main loop ───────────────────────────────────────────────────── */
    int tick = 0;
    int next_window = window_size;

    for (int t = 0; t < N; t++) {
        obs_buffer.push_back(gd.log_returns_sq[t]);

        BpfResult r = gpu_bpf_step(bpf, gd.returns[t]);
        result.est_h[t] = r.h_mean;
        tick++;

        /* Window boundary */
        if (tick >= next_window && (int)obs_buffer.size() >= window_size) {
            float* win_obs = &obs_buffer[obs_buffer.size() - window_size];

            /* ── Amnesiac: reinit cloud from prior before each window ── */
            if (!traveling) {
                smc2_cuda_init_from_prior(smc2);
            }
            /* ── Traveling: cloud carries forward, no reinit ─────────── */

            /* Run SMC² */
            smc2_cuda_update_batch(smc2, win_obs, window_size);
            next_window += stride;

            /* Get posterior mean */
            float theta_mean[N_PARAMS];
            smc2_cuda_get_theta_mean(smc2, theta_mean);

            float mu_push;

            if (use_kalman) {
                /* Get posterior covariance for Kalman measurement noise */
                float R[N_PARAMS * N_PARAMS];
                float z_meas[N_PARAMS];
                smc2_cuda_get_theta_cov(smc2, z_meas, R);

                /* Kalman update */
                kalman_update(&kalman, z_meas, R);

                /* Push Kalman's smoothed mu_base — NOT eval_curve */
                mu_push = kalman.x[3];  /* theta[MU_BASE] from Kalman */
            } else {
                /* RAW: push cloud's posterior mean directly */
                mu_push = theta_mean[3];
            }

            gpu_bpf_set_mu(bpf, mu_push);
            gpu_bpf_set_rho(bpf, use_kalman ? kalman.x[0] : theta_mean[0]);

            result.last_mu_pushed = mu_push;
            result.n_pushes++;

            /* Diagnostic every 5 windows */
            if (result.n_pushes % 5 == 0 || result.n_pushes <= 3) {
                float ess = smc2_cuda_get_outer_ess(smc2);
                printf("      [win %2d] mu_base=%.3f  rho=%.4f  ESS=%.0f  "
                       "mu_push=%.3f %s\n",
                       result.n_pushes, theta_mean[3], theta_mean[0], ess,
                       mu_push, use_kalman ? "(kalman)" : "(raw)");
            }
        }
    }

    /* ── RMSE ────────────────────────────────────────────────────────── */
    {
        double sum_sq = 0; int count = 0;
        for (int t = 0; t < N; t++) {
            if (!std::isnan(result.est_h[t]) && !std::isinf(result.est_h[t])) {
                double err = (double)result.est_h[t] - (double)gd.true_h[t];
                sum_sq += err * err; count++;
            }
        }
        result.rmse_total = (count > 0) ? sqrt(sum_sq / count) : 999.0;
    }
    {
        double sum_sq = 0; int count = 0;
        int start = (gd.score_start >= 0) ? gd.score_start : 0;
        for (int t = start; t < N; t++) {
            if (!std::isnan(result.est_h[t]) && !std::isinf(result.est_h[t])) {
                double err = (double)result.est_h[t] - (double)gd.true_h[t];
                sum_sq += err * err; count++;
            }
        }
        result.rmse_scored = (count > 0) ? sqrt(sum_sq / count) : 999.0;
    }

    smc2_cuda_free(smc2);
    gpu_bpf_destroy(bpf);
    return result;
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(int argc, char** argv) {
    int n_bpf   = (argc > 1) ? atoi(argv[1]) : 30000;
    int n_theta = 1024;
    int n_inner = 512;
    int window_size = 3000;
    int stride      = 1500;

    TrueDGP dgp = default_dgp();

    /* ── Multi-regime DGP ────────────────────────────────────────────── */
    /* Learning phase: diverse z so SMC² can identify params              */
    /* Then long calm (tests degeneracy), then Crisis 2 (scored)          */
    std::vector<Segment> segments = {
        /* Learning: alternating z for within-window diversity */
        {"L: Low 1",              2000,  -2.0f,  0},
        {"L: High 1",             2000,   2.0f,  0},
        {"L: Low 2",              2000,  -2.0f,  0},
        {"L: High 2",             2000,   2.0f,  0},
        {"L: Low 3",              2000,  -2.0f,  0},
        {"L: High 3",             2000,   2.0f,  0},

        /* Long calm: 30K ticks = 20 windows of z ≈ 0 */
        {"Calm 1",                10000, -3.0f,  0},
        {"Calm 2",                10000, -3.0f,  0},
        {"Calm 3",                10000, -3.0f,  0},

        /* Crisis 2: scored — different from learning */
        {"T: Low 1",              2000,  -1.5f,  1},
        {"T: High 1",             2000,   1.5f,  1},
        {"T: Low 2",              2000,  -1.5f,  1},
        {"T: High 2",             2000,   1.5f,  1},
        {"T: Low 3",              2000,  -1.0f,  1},
        {"T: High 3",             2000,   2.0f,  1},
    };

    PRNG dgp_rng = prng_create(98765);
    GeneratedData gd = generate_data(dgp, segments, &dgp_rng);
    int N = gd.N;

    /* ── Header ──────────────────────────────────────────────────────── */
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║   Fair Kalman Comparison: Traveling vs Amnesiac × Raw vs Kalman      ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════════╣\n");
    printf("║  Cloud travels: no reinit between windows                            ║\n");
    printf("║  Cloud amnesiac: smc2_cuda_init_from_prior() each window            ║\n");
    printf("║  All pushes are RAW mu_base — NO eval_curve anywhere                ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n\n");

    printf("  Config: %d ticks | W=%d | stride=%d | BPF=%dK | SMC²=%d×%d\n",
           N, window_size, stride, n_bpf/1000, n_theta, n_inner);
    printf("  Score starts at tick %d\n", gd.score_start);
    printf("  Curve shape params FIXED to truth (mu_scale, mu_rate, sigma_scale, sigma_rate)\n\n");

    /* Segment summary */
    int n_seg = (int)segments.size();
    printf("  %-20s %6s %5s %6s %6s %6s\n",
           "Segment", "Ticks", "Score", "z_min", "z_max", "z_avg");
    printf("  ──────────────────── ────── ───── ────── ────── ──────\n");
    for (int s = 0; s < n_seg; s++) {
        int start = gd.segment_starts[s];
        int end = (s+1 < n_seg) ? gd.segment_starts[s+1] : N;
        float zmin=1e6f, zmax=-1e6f, zsum=0;
        for (int t = start; t < end; t++) {
            if (gd.true_z[t] < zmin) zmin = gd.true_z[t];
            if (gd.true_z[t] > zmax) zmax = gd.true_z[t];
            zsum += gd.true_z[t];
        }
        printf("  %-20s %6d %5s %6.2f %6.2f %6.2f\n",
               segments[s].name, end-start,
               segments[s].score ? "YES" : "---",
               zmin, zmax, zsum/(end-start));
    }
    printf("\n");

    /* ── Run all four modes ──────────────────────────────────────────── */
    unsigned int bpf_seed = 42;
    TestMode modes[] = {MODE_TRAVEL_RAW, MODE_TRAVEL_KALMAN,
                        MODE_AMNESIA_KALMAN, MODE_AMNESIA_RAW};
    RunResult results[4];

    for (int m = 0; m < 4; m++) {
        printf("  ════════════════════════════════════════════════════════\n");
        printf("  %s\n", mode_name(modes[m]));
        printf("  ────────────────────────────────────────────────────────\n");
        results[m] = run_mode(gd, dgp, n_bpf, n_theta, n_inner,
                               window_size, stride, modes[m], bpf_seed);
        printf("    → Total RMSE: %.4f  |  Scored RMSE: %.4f  |  last_μ: %.3f\n\n",
               results[m].rmse_total, results[m].rmse_scored,
               results[m].last_mu_pushed);
    }

    /* ── Results table ───────────────────────────────────────────────── */
    printf("  ════════════════════════════════════════════════════════════════════\n");
    printf("  RESULTS (Crisis 2 scored RMSE)\n");
    printf("  ────────────────────────────────────────────────────────────────────\n");
    printf("  %-22s %10s %10s %10s\n", "Mode", "Scored", "Total", "last μ");
    printf("  ────────────────────── ────────── ────────── ──────────\n");

    int best = 0;
    for (int m = 0; m < 4; m++) {
        printf("  %-22s %10.4f %10.4f %10.3f\n",
               mode_name(modes[m]),
               results[m].rmse_scored, results[m].rmse_total,
               results[m].last_mu_pushed);
        if (results[m].rmse_scored < results[best].rmse_scored)
            best = m;
    }
    printf("\n  Best: %s\n", mode_name(modes[best]));

    /* ── Factor analysis ─────────────────────────────────────────────── */
    printf("\n  ────────────────────────────────────────────────────────────────────\n");
    printf("  FACTOR ANALYSIS (scored RMSE)\n");
    printf("  ────────────────────────────────────────────────────────────────────\n");

    double tr = results[0].rmse_scored;  /* TRAVEL_RAW */
    double tk = results[1].rmse_scored;  /* TRAVEL_KALMAN */
    double ak = results[2].rmse_scored;  /* AMNESIA_KALMAN */
    double ar = results[3].rmse_scored;  /* AMNESIA_RAW */

    printf("\n  Q1: Does the traveling cloud help?\n");
    printf("    TRAVEL_RAW    vs AMNESIA_RAW:     %+.1f%%\n", 100*(tr/ar - 1));
    printf("    TRAVEL_KALMAN vs AMNESIA_KALMAN:  %+.1f%%\n", 100*(tk/ak - 1));
    if (tr < ar && tk < ak) {
        printf("    ✓ YES — traveling cloud beats amnesiac in both cases.\n");
    } else if (tr < ar || tk < ak) {
        printf("    ~ MIXED — traveling helps in one case but not the other.\n");
    } else {
        printf("    ✗ NO — amnesiac matches or beats traveling.\n");
    }

    printf("\n  Q2: Does the Kalman help?\n");
    printf("    TRAVEL_KALMAN  vs TRAVEL_RAW:    %+.1f%%\n", 100*(tk/tr - 1));
    printf("    AMNESIA_KALMAN vs AMNESIA_RAW:   %+.1f%%\n", 100*(ak/ar - 1));
    if (tk < tr && ak < ar) {
        printf("    ✓ YES — Kalman helps in both cases.\n");
    } else if (tk < tr || ak < ar) {
        printf("    ~ MIXED — Kalman helps in one case but not the other.\n");
    } else if (tk > tr * 1.02 || ak > ar * 1.02) {
        printf("    ✗ NO — Kalman is actively hurting (>2%% worse).\n");
    } else {
        printf("    ~ NEUTRAL — Kalman is within 2%%, neither helps nor hurts.\n");
    }

    printf("\n  Q3: Interaction — does Kalman help MORE with amnesiac cloud?\n");
    double kalman_benefit_travel  = (tr - tk) / tr;   /* positive = Kalman helps */
    double kalman_benefit_amnesia = (ar - ak) / ar;
    printf("    Kalman benefit on traveling cloud:  %+.1f%%\n", 100*kalman_benefit_travel);
    printf("    Kalman benefit on amnesiac cloud:   %+.1f%%\n", 100*kalman_benefit_amnesia);
    if (kalman_benefit_amnesia > kalman_benefit_travel + 0.02) {
        printf("    ✓ Kalman helps MORE with amnesiac cloud (as expected).\n");
        printf("      The Kalman compensates for lost cross-window memory.\n");
    } else {
        printf("    ~ No clear interaction effect.\n");
    }

    printf("\n  ────────────────────────────────────────────────────────────────────\n");
    printf("  VERDICT\n");
    printf("  ────────────────────────────────────────────────────────────────────\n");

    if (best == 0) {
        printf("  → TRAVEL + RAW wins. Cloud memory is sufficient.\n");
        printf("    Kalman is redundant. Ship the simple pipeline.\n");
    } else if (best == 1) {
        printf("  → TRAVEL + KALMAN wins. Kalman adds value even with traveling cloud.\n");
        printf("    Investigate: is this from the Kalman or from autocorrelation averaging?\n");
    } else if (best == 2) {
        printf("  → AMNESIA + KALMAN wins. Kalman stitching beats traveling cloud.\n");
        printf("    The reinit-per-window + Kalman architecture is actually better.\n");
        printf("    This means the original design was sound — just the push was broken.\n");
    } else {
        printf("  → AMNESIA + RAW wins. This is unexpected and likely noise.\n");
    }

    printf("  ════════════════════════════════════════════════════════════════════\n\n");

    return 0;
}
