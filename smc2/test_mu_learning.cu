/**
 * @file test_mu_learning.cu
 * @brief Vanilla BPF vs BPF+mu learning on regime-change DGP
 *
 * DGP: Stochastic volatility with mu shift
 *   Phase 1 (t=0..T/2):    mu = -1.0  (moderate vol)
 *   Phase 2 (t=T/2..T):    mu = -2.5  (lower vol regime)
 *
 * Both filters start with mu=-1.0 (SMC² last known value).
 * Vanilla BPF keeps mu=-1.0 forever.
 * dBPF adjusts mu online via kernel 14 gradient.
 *
 * If mu learning works: dBPF RMSE should be lower in phase 2,
 * because it tracks the shift while vanilla is stuck at stale params.
 *
 * Build:
 *   nvcc -O3 test_mu_learning.cu gpu_bpf_ptx_full.cu -o test_mu_learning \
 *        -lcuda -lcurand -I../include
 *
 * Needs bpf_kernels_full.ptx (with kernel 14 appended) in cwd.
 */

#include "gpu_bpf_full.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// ═════════════════════════════════════════════════════════════════
// Host RNG
// ═════════════════════════════════════════════════════════════════

static double host_randn(unsigned long long* rng) {
    unsigned long long s = *rng;
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    double u1 = (double)(s >> 11) * (1.0 / 9007199254740992.0);
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    double u2 = (double)(s >> 11) * (1.0 / 9007199254740992.0);
    *rng = s;
    if (u1 < 1e-15) u1 = 1e-15;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

static double host_sample_t(unsigned long long* rng, double nu) {
    if (nu <= 0.0 || nu > 100.0) return host_randn(rng);
    double z = host_randn(rng);
    double chi2 = 0.0;
    for (int k = 0; k < (int)nu; k++) {
        double g = host_randn(rng);
        chi2 += g * g;
    }
    return z / sqrt(chi2 / nu);
}

// ═════════════════════════════════════════════════════════════════
// DGP: SV with mu regime change
// ═════════════════════════════════════════════════════════════════

typedef struct {
    double* h;
    double* y;
    double* mu_true;    // per-tick true mu for reference
    double* rho_true;   // per-tick true rho for reference
    int     T;
    int     t_switch;
    double  mu1, mu2;
    double  rho, sigma_z, nu_obs;
} SvRegimeData;

static SvRegimeData generate_sv_mu_regime(
    int T, double mu1, double mu2, int t_switch,
    double rho, double sigma_z, double nu_obs,
    unsigned long long seed)
{
    SvRegimeData d;
    d.T = T;
    d.t_switch = t_switch;
    d.mu1 = mu1; d.mu2 = mu2;
    d.rho = rho; d.sigma_z = sigma_z; d.nu_obs = nu_obs;
    d.h        = (double*)malloc(T * sizeof(double));
    d.y        = (double*)malloc(T * sizeof(double));
    d.mu_true  = (double*)malloc(T * sizeof(double));
    d.rho_true = (double*)malloc(T * sizeof(double));

    unsigned long long rng = seed;
    double std_stat = sigma_z / sqrt(fmax(1.0 - rho * rho, 1e-10));
    d.h[0] = mu1 + std_stat * host_randn(&rng);
    d.mu_true[0] = mu1;
    d.rho_true[0] = rho;

    for (int t = 1; t < T; t++) {
        double mu = (t < t_switch) ? mu1 : mu2;
        d.mu_true[t] = mu;
        d.rho_true[t] = rho;
        d.h[t] = mu + rho * (d.h[t - 1] - mu) + sigma_z * host_randn(&rng);
    }
    for (int t = 0; t < T; t++) {
        double vol = exp(d.h[t] / 2.0);
        d.y[t] = vol * host_sample_t(&rng, nu_obs);
    }
    return d;
}

// DGP with rho regime change (mu constant)
static SvRegimeData generate_sv_rho_regime(
    int T, double mu, double rho1, double rho2, int t_switch,
    double sigma_z, double nu_obs,
    unsigned long long seed)
{
    SvRegimeData d;
    d.T = T;
    d.t_switch = t_switch;
    d.mu1 = mu; d.mu2 = mu;
    d.rho = rho1; d.sigma_z = sigma_z; d.nu_obs = nu_obs;
    d.h        = (double*)malloc(T * sizeof(double));
    d.y        = (double*)malloc(T * sizeof(double));
    d.mu_true  = (double*)malloc(T * sizeof(double));
    d.rho_true = (double*)malloc(T * sizeof(double));

    unsigned long long rng = seed;
    double std_stat = sigma_z / sqrt(fmax(1.0 - rho1 * rho1, 1e-10));
    d.h[0] = mu + std_stat * host_randn(&rng);
    d.mu_true[0] = mu;
    d.rho_true[0] = rho1;

    for (int t = 1; t < T; t++) {
        double rho_t = (t < t_switch) ? rho1 : rho2;
        d.mu_true[t] = mu;
        d.rho_true[t] = rho_t;
        d.h[t] = mu + rho_t * (d.h[t - 1] - mu) + sigma_z * host_randn(&rng);
    }
    for (int t = 0; t < T; t++) {
        double vol = exp(d.h[t] / 2.0);
        d.y[t] = vol * host_sample_t(&rng, nu_obs);
    }
    return d;
}

static void free_sv_regime(SvRegimeData* d) {
    free(d->h); free(d->y); free(d->mu_true); free(d->rho_true);
}

// ═════════════════════════════════════════════════════════════════
// Run one filter, collect per-phase RMSE
// ═════════════════════════════════════════════════════════════════

typedef struct {
    double rmse_total;
    double rmse_phase1;
    double rmse_phase2;
    float  mu_final;
    float  rho_final;
} RunResult;

static RunResult run_bpf(
    const SvRegimeData* data,
    int N,               // particles
    float init_mu,       // starting mu (= SMC²'s last known)
    float init_rho,      // starting rho
    float rho,           // DGP rho (for reference, init_rho may differ)
    float sigma_z,
    float nu_obs,
    int   learn_mode,    // 0=vanilla, 1=natural gradient, 2=robbins-monro
    int   learn_rho,     // 0=off, 1=also learn rho
    float rm_c,
    int   update_K,
    float rm_t0,
    float rm_gamma,
    int   seed,
    int   log_every)     // 0 = no logging
{
    int T = data->T;
    int t_switch = data->t_switch;
    int skip = 50;
    (void)rho;  // DGP reference only, init_rho goes to create

    GpuBpfState* s = gpu_bpf_create(N, init_rho, sigma_z, init_mu,
                                      0.0f, nu_obs, seed);

    if (learn_mode > 0) {
        gpu_bpf_enable_mu_learning(s, learn_mode, update_K, rm_c, rm_t0, rm_gamma);
        if (learn_rho) {
            gpu_bpf_enable_rho_learning(s, 1);
        }
    }

    double sse_total = 0.0, sse_p1 = 0.0, sse_p2 = 0.0;
    int    cnt_total = 0,   cnt_p1 = 0,   cnt_p2 = 0;

    for (int t = 0; t < T; t++) {
        BpfResult r = gpu_bpf_step(s, (float)data->y[t]);

        if (t >= skip) {
            double err = (double)r.h_mean - data->h[t];
            double err2 = err * err;
            sse_total += err2; cnt_total++;

            if (t < t_switch) {
                sse_p1 += err2; cnt_p1++;
            } else {
                sse_p2 += err2; cnt_p2++;
            }
        }

        if (log_every > 0 && t > 0 && t % log_every == 0) {
            float mu_cur = gpu_bpf_get_mu(s);
            double rmse_so_far = (cnt_total > 0) ? sqrt(sse_total / cnt_total) : 0;
            const char* phase = (t < t_switch) ? "phase1" : "PHASE2";
            printf("    [t=%5d %s]  mu=%+.4f  true_mu=%+.2f  RMSE=%.4f  h_est=%+.3f\n",
                   t, phase, mu_cur, data->mu_true[t], rmse_so_far, r.h_mean);
        }
    }

    RunResult res;
    res.mu_final    = gpu_bpf_get_mu(s);
    res.rho_final   = gpu_bpf_get_rho(s);
    res.rmse_total  = (cnt_total > 0) ? sqrt(sse_total / cnt_total) : -1;
    res.rmse_phase1 = (cnt_p1 > 0)    ? sqrt(sse_p1 / cnt_p1)      : -1;
    res.rmse_phase2 = (cnt_p2 > 0)    ? sqrt(sse_p2 / cnt_p2)      : -1;

    gpu_bpf_destroy(s);
    return res;
}

// ═════════════════════════════════════════════════════════════════
// TEST 1: Head-to-head — vanilla vs mu learning
// ═════════════════════════════════════════════════════════════════

static void test_head_to_head(void) {
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  TEST 1: Vanilla BPF vs BPF+natural_gradient — Regime Change\n");
    printf("  DGP: mu=-1.0 (t<2500) → mu=-2.5 (t≥2500)\n");
    printf("       rho=0.98  sigma_z=0.15  nu_obs=5.0  T=5000  N=4096\n");
    printf("  Both start at mu=-1.0 (SMC²'s last known value)\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    int T = 5000, t_switch = 2500, N = 4096;
    float rho = 0.98f, sigma_z = 0.15f, nu_obs = 5.0f;
    double mu1 = -1.0, mu2 = -2.5;

    SvRegimeData data = generate_sv_mu_regime(
        T, mu1, mu2, t_switch, rho, sigma_z, nu_obs, 42ULL);

    printf("  ── Vanilla BPF (mu=-1.0 fixed) ──\n");
    RunResult r_vanilla = run_bpf(&data, N, -1.0f, rho, rho, sigma_z, nu_obs,
                                    0, 0, 0, 0, 0, 0, 42, 500);

    printf("\n  ── BPF + natural gradient (K=50, c=0.1, t0=10, γ=2/3) ──\n");
    RunResult r_learn = run_bpf(&data, N, -1.0f, rho, rho, sigma_z, nu_obs,
                                  1, 0, 0.1f, 50, 10.0f, 0.667f, 42, 500);

    printf("\n  ┌─────────────────────┬───────────────┬───────────────┐\n");
    printf("  │                     │ Vanilla BPF   │ BPF+nat_grad  │\n");
    printf("  ├─────────────────────┼───────────────┼───────────────┤\n");
    printf("  │ Phase 1 RMSE        │    %.4f     │    %.4f     │\n",
           r_vanilla.rmse_phase1, r_learn.rmse_phase1);
    printf("  │ Phase 2 RMSE        │    %.4f     │    %.4f     │\n",
           r_vanilla.rmse_phase2, r_learn.rmse_phase2);
    printf("  │ Total RMSE          │    %.4f     │    %.4f     │\n",
           r_vanilla.rmse_total, r_learn.rmse_total);
    printf("  │ Final mu            │   %+.4f     │   %+.4f     │\n",
           -1.0f, r_learn.mu_final);
    printf("  └─────────────────────┴───────────────┴───────────────┘\n");
    printf("  True mu: -1.0 (phase1), -2.5 (phase2)\n");

    double p2_improvement = 100.0 * (r_vanilla.rmse_phase2 - r_learn.rmse_phase2)
                            / r_vanilla.rmse_phase2;
    printf("  Phase 2 improvement: %+.1f%%\n", p2_improvement);

    if (r_learn.rmse_phase2 < r_vanilla.rmse_phase2)
        printf("  ✓ natural gradient WINS in phase 2\n\n");
    else
        printf("  ✗ natural gradient did NOT help in phase 2\n\n");

    free_sv_regime(&data);
}

// ═══════════════════════════════════════════════════════════════════
// TEST 2: Multiple seeds — statistical significance
// ═══════════════════════════════════════════════════════════════════

static void test_multi_seed(void) {
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  TEST 2: Multi-seed comparison (10 seeds)\n");
    printf("  DGP: mu=-1.0 → mu=-2.5 at T/2\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    int T = 5000, t_switch = 2500, N = 4096;
    float rho = 0.98f, sigma_z = 0.15f, nu_obs = 5.0f;
    int n_seeds = 10;

    int vanilla_wins_p2 = 0, learn_wins_p2 = 0;
    double sum_van_p2 = 0, sum_lrn_p2 = 0;
    double sum_van_total = 0, sum_lrn_total = 0;

    printf("  %-6s  %-10s %-10s  %-10s %-10s  %s\n",
           "Seed", "Van P2", "Learn P2", "Van Total", "Learn Tot", "Winner P2");
    printf("  ─────  ────────── ──────────  ────────── ──────────  ─────────\n");

    for (int si = 0; si < n_seeds; si++) {
        unsigned long long seed = 42ULL + si * 1337ULL;

        SvRegimeData data = generate_sv_mu_regime(
            T, -1.0, -2.5, t_switch, rho, sigma_z, nu_obs, seed);

        RunResult rv = run_bpf(&data, N, -1.0f, rho, rho, sigma_z, nu_obs,
                                 0, 0, 0, 0, 0, 0, (int)seed, 0);
        RunResult rl = run_bpf(&data, N, -1.0f, rho, rho, sigma_z, nu_obs,
                                 1, 0, 0.1f, 50, 10.0f, 0.667f, (int)seed, 0);

        const char* winner = (rl.rmse_phase2 < rv.rmse_phase2) ? "LEARN" : "vanilla";
        if (rl.rmse_phase2 < rv.rmse_phase2) learn_wins_p2++; else vanilla_wins_p2++;

        sum_van_p2 += rv.rmse_phase2;    sum_lrn_p2 += rl.rmse_phase2;
        sum_van_total += rv.rmse_total;   sum_lrn_total += rl.rmse_total;

        printf("  %-6d  %.4f     %.4f      %.4f     %.4f      %s\n",
               si, rv.rmse_phase2, rl.rmse_phase2,
               rv.rmse_total, rl.rmse_total, winner);

        free_sv_regime(&data);
    }

    printf("\n  ── Summary ──\n");
    printf("  Phase 2 wins: learn=%d  vanilla=%d  (out of %d)\n",
           learn_wins_p2, vanilla_wins_p2, n_seeds);
    printf("  Mean Phase 2 RMSE: vanilla=%.4f  learn=%.4f  (Δ=%+.4f)\n",
           sum_van_p2 / n_seeds, sum_lrn_p2 / n_seeds,
           sum_lrn_p2 / n_seeds - sum_van_p2 / n_seeds);
    printf("  Mean Total RMSE:   vanilla=%.4f  learn=%.4f  (Δ=%+.4f)\n\n",
           sum_van_total / n_seeds, sum_lrn_total / n_seeds,
           sum_lrn_total / n_seeds - sum_van_total / n_seeds);
}

// ═══════════════════════════════════════════════════════════════════
// TEST 3: No regime change — does mu learning hurt when stable?
// ═══════════════════════════════════════════════════════════════════

static void test_no_regime_change(void) {
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  TEST 3: No regime change — does mu learning hurt?\n");
    printf("  DGP: mu=-1.0 constant, T=5000\n");
    printf("  Both start at mu=-1.0 (correct params)\n");
    printf("  If learning drifts away, it's harmful.\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    int T = 5000, N = 4096;
    float rho = 0.98f, sigma_z = 0.15f, nu_obs = 5.0f;
    int n_seeds = 10;

    double sum_van = 0, sum_lrn = 0;
    int learn_worse = 0;

    printf("  %-6s  %-10s %-10s  %-10s  %s\n",
           "Seed", "Vanilla", "Learn", "mu_final", "Harm?");
    printf("  ─────  ────────── ──────────  ──────────  ─────\n");

    for (int si = 0; si < n_seeds; si++) {
        unsigned long long seed = 42ULL + si * 1337ULL;

        // No regime change: mu=-1.0 → mu=-1.0
        SvRegimeData data = generate_sv_mu_regime(
            T, -1.0, -1.0, T, rho, sigma_z, nu_obs, seed);

        RunResult rv = run_bpf(&data, N, -1.0f, rho, rho, sigma_z, nu_obs,
                                 0, 0, 0, 0, 0, 0, (int)seed, 0);
        RunResult rl = run_bpf(&data, N, -1.0f, rho, rho, sigma_z, nu_obs,
                                 1, 0, 0.1f, 50, 10.0f, 0.667f, (int)seed, 0);

        sum_van += rv.rmse_total;
        sum_lrn += rl.rmse_total;
        int worse = (rl.rmse_total > rv.rmse_total + 0.005);  // >0.5% margin
        if (worse) learn_worse++;

        printf("  %-6d  %.4f     %.4f      %+.4f     %s\n",
               si, rv.rmse_total, rl.rmse_total, rl.mu_final,
               worse ? "YES" : "no");

        free_sv_regime(&data);
    }

    printf("\n  ── Summary ──\n");
    printf("  Mean RMSE: vanilla=%.4f  learn=%.4f  (Δ=%+.4f)\n",
           sum_van / n_seeds, sum_lrn / n_seeds,
           sum_lrn / n_seeds - sum_van / n_seeds);
    printf("  Harmful (>0.5%% worse): %d / %d seeds\n\n", learn_worse, n_seeds);
}

// ═══════════════════════════════════════════════════════════════════
// TEST 4: SMC² correction mid-stream
// ═══════════════════════════════════════════════════════════════════

static void test_smc2_correction(void) {
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  TEST 4: SMC² correction + online maintenance\n");
    printf("  DGP: mu=-1.0 → mu=-2.5 at t=2000\n");
    printf("  SMC² fires at t=3000, corrects mu to -2.5\n");
    printf("  Question: does online learning maintain the correction?\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    int T = 5000, t_switch = 2000, N = 4096;
    float rho = 0.98f, sigma_z = 0.15f, nu_obs = 5.0f;
    int t_smc2 = 3000;
    int skip = 50;

    SvRegimeData data = generate_sv_mu_regime(
        T, -1.0, -2.5, t_switch, rho, sigma_z, nu_obs, 42ULL);

    // Vanilla: SMC² corrects at t=3000, fixed after
    GpuBpfState* s_van = gpu_bpf_create(N, rho, sigma_z, -1.0f, 0.0f, nu_obs, 42);

    // Learning: SMC² corrects at t=3000, online continues
    GpuBpfState* s_lrn = gpu_bpf_create(N, rho, sigma_z, -1.0f, 0.0f, nu_obs, 42);
    gpu_bpf_enable_mu_learning(s_lrn, 1, 50, 0.1f, 10.0f, 0.667f);

    double sse_van_post = 0, sse_lrn_post = 0;
    int cnt_post = 0;

    for (int t = 0; t < T; t++) {
        float y = (float)data.y[t];

        // SMC² correction at t=3000
        if (t == t_smc2) {
            printf("    >>> SMC² fires at t=%d — setting mu=-2.5 <<<\n", t_smc2);
            s_van->mu = -2.5f;
            gpu_bpf_set_mu(s_lrn, -2.5f);
        }

        BpfResult rv = gpu_bpf_step(s_van, y);
        BpfResult rl = gpu_bpf_step(s_lrn, y);

        if (t >= t_smc2 + skip) {
            double ev = (double)rv.h_mean - data.h[t];
            double el = (double)rl.h_mean - data.h[t];
            sse_van_post += ev * ev;
            sse_lrn_post += el * el;
            cnt_post++;
        }

        if (t > 0 && t % 500 == 0) {
            float mu_van = gpu_bpf_get_mu(s_van);
            float mu_lrn = gpu_bpf_get_mu(s_lrn);
            const char* phase = (t < t_switch) ? "pre " :
                                (t < t_smc2)  ? "GAP " : "post";
            printf("    [t=%5d %s]  van_mu=%+.4f  lrn_mu=%+.4f  true=%+.2f\n",
                   t, phase, mu_van, mu_lrn, data.mu_true[t]);
        }
    }

    double rmse_van_post = sqrt(sse_van_post / cnt_post);
    double rmse_lrn_post = sqrt(sse_lrn_post / cnt_post);

    printf("\n  Post-correction RMSE (t=%d..%d):\n", t_smc2 + skip, T);
    printf("    Vanilla:  %.4f\n", rmse_van_post);
    printf("    Learning: %.4f  (mu_final=%+.4f)\n",
           rmse_lrn_post, gpu_bpf_get_mu(s_lrn));
    printf("    Δ = %+.4f\n\n", rmse_lrn_post - rmse_van_post);

    gpu_bpf_destroy(s_van);
    gpu_bpf_destroy(s_lrn);
    free_sv_regime(&data);
}

// ═══════════════════════════════════════════════════════════════════
// TEST 5: ρ regime change — does rho learning help?
// ═══════════════════════════════════════════════════════════════════

static void test_rho_learning(void) {
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  TEST 5: ρ regime change — vanilla vs mu-only vs mu+rho learning\n");
    printf("  DGP: ρ=0.98 (t<2500) → ρ=0.90 (t≥2500), μ=-1.0 constant\n");
    printf("       sigma_z=0.15  nu_obs=5.0  T=5000  N=4096\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    int T = 5000, t_switch = 2500, N = 4096;
    float sigma_z = 0.15f, nu_obs = 5.0f;
    double rho1 = 0.98, rho2 = 0.90;
    int n_seeds = 10;

    double sum_van_p2 = 0, sum_mu_p2 = 0, sum_both_p2 = 0;
    double sum_van_tot = 0, sum_mu_tot = 0, sum_both_tot = 0;
    int mu_wins = 0, both_wins = 0, van_wins = 0;

    printf("  %-5s  %-8s %-8s %-8s  %-8s %-8s  %s\n",
           "Seed", "Van P2", "mu P2", "mu+ρ P2", "ρ_final", "μ_final", "Winner");
    printf("  ─────  ──────── ──────── ────────  ──────── ────────  ─────────\n");

    for (int si = 0; si < n_seeds; si++) {
        unsigned long long seed = 42ULL + si * 1337ULL;

        SvRegimeData data = generate_sv_rho_regime(
            T, -1.0, rho1, rho2, t_switch, sigma_z, nu_obs, seed);

        // Vanilla: fixed rho=0.98, mu=-1.0
        RunResult rv = run_bpf(&data, N, -1.0f, (float)rho1, (float)rho1, sigma_z, nu_obs,
                                 0, 0, 0, 0, 0, 0, (int)seed, 0);

        // Mu learning only: rho fixed at 0.98
        RunResult rm = run_bpf(&data, N, -1.0f, (float)rho1, (float)rho1, sigma_z, nu_obs,
                                 1, 0, 0.1f, 50, 10.0f, 0.667f, (int)seed, 0);

        // Mu + rho learning: both adapt
        RunResult rb = run_bpf(&data, N, -1.0f, (float)rho1, (float)rho1, sigma_z, nu_obs,
                                 1, 1, 0.1f, 50, 10.0f, 0.667f, (int)seed, 0);

        sum_van_p2  += rv.rmse_phase2;   sum_van_tot  += rv.rmse_total;
        sum_mu_p2   += rm.rmse_phase2;   sum_mu_tot   += rm.rmse_total;
        sum_both_p2 += rb.rmse_phase2;   sum_both_tot += rb.rmse_total;

        // Best of 3
        const char* winner;
        if (rb.rmse_phase2 <= rm.rmse_phase2 && rb.rmse_phase2 <= rv.rmse_phase2) {
            winner = "mu+ρ"; both_wins++;
        } else if (rm.rmse_phase2 <= rv.rmse_phase2) {
            winner = "mu"; mu_wins++;
        } else {
            winner = "vanilla"; van_wins++;
        }

        printf("  %-5d  %.4f   %.4f   %.4f    %.4f   %+.4f   %s\n",
               si, rv.rmse_phase2, rm.rmse_phase2, rb.rmse_phase2,
               rb.rho_final, rb.mu_final, winner);

        free_sv_regime(&data);
    }

    printf("\n  ── Summary ──\n");
    printf("  Phase 2 wins: vanilla=%d  mu=%d  mu+ρ=%d\n", van_wins, mu_wins, both_wins);
    printf("  Mean P2 RMSE:  vanilla=%.4f  mu=%.4f  mu+ρ=%.4f\n",
           sum_van_p2/n_seeds, sum_mu_p2/n_seeds, sum_both_p2/n_seeds);
    printf("  Mean Total:    vanilla=%.4f  mu=%.4f  mu+ρ=%.4f\n\n",
           sum_van_tot/n_seeds, sum_mu_tot/n_seeds, sum_both_tot/n_seeds);
}

// ═══════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════

int main(void) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  BPF Online Parameter Learning — Natural Gradient + Robbins-Monro\n");
    printf("  Kernel 14: Fisher-normalized gradient for μ and ρ\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    test_head_to_head();
    test_multi_seed();
    test_no_regime_change();
    test_smc2_correction();
    test_rho_learning();

    return 0;
}
