/**
 * @file test_mix_bands.cu
 * @brief Validate band-based mixture proposal for BPF
 *
 * Tests:
 *   1. Bands disabled (n_bands=1): RMSE matches baseline
 *   2. Bands enabled with oracle params: RMSE ≈ baseline (no degradation)
 *   3. Bands enabled with misspecified params: RMSE < baseline (improvement)
 *   4. Weight correction sanity: mean weight ≈ 1/N (unbiased)
 *
 * Build:
 *   nvcc -O2 -o test_mix_bands test_mix_bands.cu gpu_bpf.cu -lcurand
 *
 * Usage:
 *   ./test_mix_bands [n_particles]
 */

#include "gpu_bpf.cuh"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// =============================================================================
// Simple SV DGP (same as test_bpf_matched_dgp.cu)
// =============================================================================

static unsigned long long test_rng = 12345ULL;

static unsigned int test_pcg32(void) {
    unsigned long long old = test_rng;
    test_rng = old * 6364136223846793005ULL + 1442695040888963407ULL;
    unsigned int xsh = (unsigned int)(((old >> 18u) ^ old) >> 27u);
    unsigned int rot = (unsigned int)(old >> 59u);
    return (xsh >> rot) | (xsh << ((-rot) & 31));
}

static float test_randn(void) {
    // Box-Muller
    float u1 = (float)(test_pcg32() >> 9) / 8388608.0f;
    float u2 = (float)(test_pcg32() >> 9) / 8388608.0f;
    if (u1 < 1e-10f) u1 = 1e-10f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

typedef struct {
    double* h;
    double* returns;
    int n;
} SvData;

static SvData generate_sv(int n_ticks, float rho, float sigma_z, float mu, int seed) {
    test_rng = (unsigned long long)seed * 999983ULL + 1ULL;
    SvData d;
    d.n = n_ticks;
    d.h = (double*)malloc(n_ticks * sizeof(double));
    d.returns = (double*)malloc(n_ticks * sizeof(double));

    float std_stat = sqrtf(sigma_z * sigma_z / fmaxf(1.0f - rho * rho, 1e-6f));
    d.h[0] = mu + std_stat * test_randn();
    d.returns[0] = expf((float)d.h[0] * 0.5f) * test_randn();

    for (int t = 1; t < n_ticks; t++) {
        d.h[t] = mu + rho * (d.h[t-1] - mu) + sigma_z * test_randn();
        d.returns[t] = expf((float)d.h[t] * 0.5f) * test_randn();
    }
    return d;
}

// Inject a vol spike at a given tick
static void inject_spike(SvData* d, int tick, float new_h) {
    if (tick < d->n) {
        d->h[tick] = new_h;
        d->returns[tick] = expf(new_h * 0.5f) * test_randn();
    }
}

static void free_sv(SvData* d) {
    free(d->h);
    free(d->returns);
}

// =============================================================================
// Run BPF and compute RMSE
// =============================================================================

static double run_bpf_rmse(const SvData* d, int n_particles,
                            float rho, float sigma_z, float mu,
                            float nu_state, float nu_obs, int seed,
                            int n_bands, const float* fracs, const float* scales) {
    GpuBpfState* s = gpu_bpf_create(n_particles, rho, sigma_z, mu,
                                     nu_state, nu_obs, seed);
    // Configure bands
    if (n_bands > 1) {
        gpu_bpf_set_bands(n_particles, n_bands, fracs, scales);
    }

    int skip = 100;
    double sum_sq = 0.0;
    int count = 0;

    for (int t = 0; t < d->n; t++) {
        BpfResult r = gpu_bpf_step(s, (float)d->returns[t]);
        if (t >= skip) {
            double err = (double)r.h_mean - d->h[t];
            sum_sq += err * err;
            count++;
        }
    }
    gpu_bpf_destroy(s);

    // Reset bands to 1 for next test
    {
        float f1[] = {1.0f};
        float s1[] = {1.0f};
        gpu_bpf_set_bands(n_particles, 1, f1, s1);
    }

    return sqrt(sum_sq / fmax(count, 1));
}

// Adaptive version: 3 regimes switched by surprise score
static double run_bpf_rmse_adaptive(const SvData* d, int n_particles,
                                     float rho, float sigma_z, float mu,
                                     float nu_state, float nu_obs, int seed,
                                     float thresh_alert, float thresh_panic) {
    GpuBpfState* s = gpu_bpf_create(n_particles, rho, sigma_z, mu,
                                     nu_state, nu_obs, seed);

    // Calm: 99% standard, 1% insurance
    float calm_f[]  = {0.99f, 0.01f};
    float calm_s[]  = {1.0f,  4.0f};

    // Alert: 90% standard, moderate exploration
    float alert_f[] = {0.90f, 0.05f, 0.03f, 0.02f};
    float alert_s[] = {1.0f,  2.0f,  4.0f,  8.0f};

    // Panic: 70% standard, aggressive exploration
    float panic_f[] = {0.70f, 0.15f, 0.10f, 0.05f};
    float panic_s[] = {1.0f,  2.0f,  5.0f,  12.0f};

    gpu_bpf_set_adaptive_bands(n_particles,
        calm_f,  calm_s,  2,
        alert_f, alert_s, 4,
        panic_f, panic_s, 4,
        thresh_alert, thresh_panic);

    int skip = 100;
    double sum_sq = 0.0;
    int count = 0;

    for (int t = 0; t < d->n; t++) {
        BpfResult r = gpu_bpf_step(s, (float)d->returns[t]);
        if (t >= skip) {
            double err = (double)r.h_mean - d->h[t];
            sum_sq += err * err;
            count++;
        }
    }
    gpu_bpf_destroy(s);

    // Reset to disabled
    {
        float f1[] = {1.0f};
        float s1[] = {1.0f};
        gpu_bpf_set_bands(n_particles, 1, f1, s1);
    }

    return sqrt(sum_sq / fmax(count, 1));
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 100000;
    int n_ticks = 3000;
    int seed = 42;

    // True DGP params
    float true_rho = 0.98f, true_sz = 0.15f, true_mu = -1.0f;

    // Band config: 4 bands
    float fracs4[]  = { 0.84f, 0.07f, 0.06f, 0.03f };
    float scales4[] = { 1.0f,  2.0f,  4.0f,  8.0f  };

    printf("=== Band-Based Mixture Proposal Test ===\n");
    printf("Particles: %d, Ticks: %d\n", N, n_ticks);
    printf("True params: rho=%.2f, sigma_z=%.2f, mu=%.1f\n\n",
           true_rho, true_sz, true_mu);

    // ─────────────────────────────────────────────────────────────────────
    // TEST 1: Oracle params — bands should NOT degrade performance
    // ─────────────────────────────────────────────────────────────────────
    printf("--- TEST 1: Oracle parameters (matched DGP) ---\n");
    SvData d1 = generate_sv(n_ticks, true_rho, true_sz, true_mu, seed);

    double rmse_base = run_bpf_rmse(&d1, N, true_rho, true_sz, true_mu,
                                     0, 0, seed, 1, NULL, NULL);
    double rmse_band = run_bpf_rmse(&d1, N, true_rho, true_sz, true_mu,
                                     0, 0, seed, 4, fracs4, scales4);

    printf("  Standard BPF RMSE: %.4f\n", rmse_base);
    printf("  4-Band BPF  RMSE: %.4f\n", rmse_band);
    printf("  Ratio (band/base): %.3f  %s\n\n",
           rmse_band / rmse_base,
           (rmse_band / rmse_base < 1.05) ? "OK (< 5% degradation)" : "WARN");
    free_sv(&d1);

    // ─────────────────────────────────────────────────────────────────────
    // TEST 2: Moderate misspecification — sigma_z too low by 2x
    // ─────────────────────────────────────────────────────────────────────
    printf("--- TEST 2: Moderate misspec (sigma_z off by 2x) ---\n");
    SvData d2 = generate_sv(n_ticks, true_rho, true_sz * 2.0f, true_mu, seed + 1);

    double rmse2_base = run_bpf_rmse(&d2, N, true_rho, true_sz, true_mu,
                                      0, 0, seed, 1, NULL, NULL);
    double rmse2_band = run_bpf_rmse(&d2, N, true_rho, true_sz, true_mu,
                                      0, 0, seed, 4, fracs4, scales4);

    printf("  Standard BPF RMSE: %.4f\n", rmse2_base);
    printf("  4-Band BPF  RMSE: %.4f\n", rmse2_band);
    printf("  Ratio (band/base): %.3f  %s\n\n",
           rmse2_band / rmse2_base,
           (rmse2_band < rmse2_base) ? "IMPROVED" : "no improvement");
    free_sv(&d2);

    // ─────────────────────────────────────────────────────────────────────
    // TEST 3: Severe misspecification — sigma_z off by 4x
    // ─────────────────────────────────────────────────────────────────────
    printf("--- TEST 3: Severe misspec (sigma_z off by 4x) ---\n");
    SvData d3 = generate_sv(n_ticks, true_rho, true_sz * 4.0f, true_mu, seed + 2);

    double rmse3_base = run_bpf_rmse(&d3, N, true_rho, true_sz, true_mu,
                                      0, 0, seed, 1, NULL, NULL);
    double rmse3_band = run_bpf_rmse(&d3, N, true_rho, true_sz, true_mu,
                                      0, 0, seed, 4, fracs4, scales4);

    printf("  Standard BPF RMSE: %.4f\n", rmse3_base);
    printf("  4-Band BPF  RMSE: %.4f\n", rmse3_band);
    printf("  Ratio (band/base): %.3f  %s\n\n",
           rmse3_band / rmse3_base,
           (rmse3_band < rmse3_base) ? "IMPROVED" : "no improvement");
    free_sv(&d3);

    // ─────────────────────────────────────────────────────────────────────
    // TEST 4: Spike gauntlet — sudden vol jumps
    // ─────────────────────────────────────────────────────────────────────
    printf("--- TEST 4: Spike gauntlet (3 sudden vol jumps) ---\n");
    SvData d4 = generate_sv(n_ticks, true_rho, true_sz, true_mu, seed + 3);
    inject_spike(&d4, 500,  2.0f);   // mild spike
    inject_spike(&d4, 1000, 4.0f);   // big spike
    inject_spike(&d4, 1500, 6.0f);   // extreme spike

    double rmse4_base = run_bpf_rmse(&d4, N, true_rho, true_sz, true_mu,
                                      0, 0, seed, 1, NULL, NULL);
    double rmse4_band = run_bpf_rmse(&d4, N, true_rho, true_sz, true_mu,
                                      0, 0, seed, 4, fracs4, scales4);

    printf("  Standard BPF RMSE: %.4f\n", rmse4_base);
    printf("  4-Band BPF  RMSE: %.4f\n", rmse4_band);
    printf("  Ratio (band/base): %.3f  %s\n\n",
           rmse4_band / rmse4_base,
           (rmse4_band < rmse4_base) ? "IMPROVED" : "no improvement");
    free_sv(&d4);

    // ─────────────────────────────────────────────────────────────────────
    // TEST 5: Different band configs — find optimal
    // ─────────────────────────────────────────────────────────────────────
    printf("--- TEST 5: Band config sweep (severe misspec, sigma_z × 4) ---\n");
    SvData d5 = generate_sv(n_ticks, true_rho, true_sz * 4.0f, true_mu, seed + 4);

    // Config A: 2 bands only
    float fracs2[]  = { 0.90f, 0.10f };
    float scales2[] = { 1.0f,  4.0f  };

    // Config B: 3 bands
    float fracs3[]   = { 0.85f, 0.10f, 0.05f };
    float scales3b[] = { 1.0f,  3.0f,  8.0f  };

    // Config C: 4 bands (default)
    // Already defined above

    // Config D: 4 bands, more aggressive
    float fracs4a[]  = { 0.70f, 0.15f, 0.10f, 0.05f };
    float scales4a[] = { 1.0f,  2.0f,  5.0f,  12.0f };

    double rmse5_base = run_bpf_rmse(&d5, N, true_rho, true_sz, true_mu,
                                      0, 0, seed, 1, NULL, NULL);
    double rmse5_A = run_bpf_rmse(&d5, N, true_rho, true_sz, true_mu,
                                   0, 0, seed, 2, fracs2, scales2);
    double rmse5_B = run_bpf_rmse(&d5, N, true_rho, true_sz, true_mu,
                                   0, 0, seed, 3, fracs3, scales3b);
    double rmse5_C = run_bpf_rmse(&d5, N, true_rho, true_sz, true_mu,
                                   0, 0, seed, 4, fracs4, scales4);
    double rmse5_D = run_bpf_rmse(&d5, N, true_rho, true_sz, true_mu,
                                   0, 0, seed, 4, fracs4a, scales4a);

    printf("  %-30s RMSE: %.4f\n", "Standard BPF (1 band)", rmse5_base);
    printf("  %-30s RMSE: %.4f (%.1f%%)\n", "2-band [1, 4]", rmse5_A,
           100.0*(rmse5_A/rmse5_base - 1.0));
    printf("  %-30s RMSE: %.4f (%.1f%%)\n", "3-band [1, 3, 8]", rmse5_B,
           100.0*(rmse5_B/rmse5_base - 1.0));
    printf("  %-30s RMSE: %.4f (%.1f%%)\n", "4-band [1, 2, 4, 8]", rmse5_C,
           100.0*(rmse5_C/rmse5_base - 1.0));
    printf("  %-30s RMSE: %.4f (%.1f%%)\n", "4-band [1, 2, 5, 12] aggro", rmse5_D,
           100.0*(rmse5_D/rmse5_base - 1.0));
    free_sv(&d5);

    // ─────────────────────────────────────────────────────────────────────
    // TEST 6: Oracle cost minimization — find smallest band allocation
    //         that still helps under misspec
    // ─────────────────────────────────────────────────────────────────────
    printf("--- TEST 6: Oracle cost vs misspec gain tradeoff ---\n");
    SvData d6o = generate_sv(n_ticks, true_rho, true_sz, true_mu, seed + 5);
    SvData d6m = generate_sv(n_ticks, true_rho, true_sz * 4.0f, true_mu, seed + 5);

    // Configs with decreasing exploration fraction
    struct { const char* name; int nb; float f[4]; float s[4]; } cfgs[] = {
        {"16% explore [84/7/6/3]",  4, {0.84f,0.07f,0.06f,0.03f}, {1,2,4,8}},
        {"10% explore [90/5/3/2]",  4, {0.90f,0.05f,0.03f,0.02f}, {1,2,4,8}},
        {" 5% explore [95/3/1.5/.5]",4,{0.95f,0.03f,0.015f,0.005f},{1,2,4,8}},
        {" 3% explore [97/2/0.7/.3]",4,{0.97f,0.02f,0.007f,0.003f},{1,2,4,8}},
        {" 5% explore [95/5] 2band",2, {0.95f,0.05f,0,0},         {1,4,0,0}},
        {" 5% aggro [95/3/1.5/.5]", 4, {0.95f,0.03f,0.015f,0.005f},{1,3,8,16}},
    };
    int n_cfgs = 6;

    double rmse6o_base = run_bpf_rmse(&d6o, N, true_rho, true_sz, true_mu,
                                       0, 0, seed, 1, NULL, NULL);
    double rmse6m_base = run_bpf_rmse(&d6m, N, true_rho, true_sz, true_mu,
                                       0, 0, seed, 1, NULL, NULL);

    printf("  %-35s Oracle: %.4f        Misspec4x: %.4f\n",
           "Standard BPF", rmse6o_base, rmse6m_base);

    for (int c = 0; c < n_cfgs; c++) {
        double ro = run_bpf_rmse(&d6o, N, true_rho, true_sz, true_mu,
                                  0, 0, seed, cfgs[c].nb, cfgs[c].f, cfgs[c].s);
        double rm = run_bpf_rmse(&d6m, N, true_rho, true_sz, true_mu,
                                  0, 0, seed, cfgs[c].nb, cfgs[c].f, cfgs[c].s);
        printf("  %-35s Oracle: %.4f (%+.1f%%)  Misspec4x: %.4f (%+.1f%%)\n",
               cfgs[c].name, ro, 100.0*(ro/rmse6o_base-1.0),
               rm, 100.0*(rm/rmse6m_base-1.0));
    }
    free_sv(&d6o);
    free_sv(&d6m);

    // ─────────────────────────────────────────────────────────────────────
    // TEST 7: Adaptive bands — best of both worlds
    // ─────────────────────────────────────────────────────────────────────
    printf("\n--- TEST 7: Adaptive bands (surprise-driven regime switching) ---\n");
    printf("  Calm:  99%% std + 1%% wide    (|eta| < 2)\n");
    printf("  Alert: 90/5/3/2 x [1,2,4,8]  (|eta| < 4)\n");
    printf("  Panic: 70/15/10/5 x [1,2,5,12] (|eta| >= 4)\n\n");

    // 7a: Oracle params
    SvData d7o = generate_sv(n_ticks, true_rho, true_sz, true_mu, seed + 7);
    double rmse7o_base = run_bpf_rmse(&d7o, N, true_rho, true_sz, true_mu,
                                       0, 0, seed, 1, NULL, NULL);
    double rmse7o_adapt = run_bpf_rmse_adaptive(&d7o, N, true_rho, true_sz, true_mu,
                                                  0, 0, seed, 2.0f, 4.0f);
    printf("  Oracle:     Standard=%.4f  Adaptive=%.4f (%+.1f%%)\n",
           rmse7o_base, rmse7o_adapt, 100.0*(rmse7o_adapt/rmse7o_base - 1.0));

    // 7b: Moderate misspec (2x)
    SvData d7m = generate_sv(n_ticks, true_rho, true_sz * 2.0f, true_mu, seed + 7);
    double rmse7m_base = run_bpf_rmse(&d7m, N, true_rho, true_sz, true_mu,
                                       0, 0, seed, 1, NULL, NULL);
    double rmse7m_adapt = run_bpf_rmse_adaptive(&d7m, N, true_rho, true_sz, true_mu,
                                                  0, 0, seed, 2.0f, 4.0f);
    printf("  Misspec2x:  Standard=%.4f  Adaptive=%.4f (%+.1f%%)\n",
           rmse7m_base, rmse7m_adapt, 100.0*(rmse7m_adapt/rmse7m_base - 1.0));

    // 7c: Severe misspec (4x)
    SvData d7s = generate_sv(n_ticks, true_rho, true_sz * 4.0f, true_mu, seed + 7);
    double rmse7s_base = run_bpf_rmse(&d7s, N, true_rho, true_sz, true_mu,
                                       0, 0, seed, 1, NULL, NULL);
    double rmse7s_adapt = run_bpf_rmse_adaptive(&d7s, N, true_rho, true_sz, true_mu,
                                                  0, 0, seed, 2.0f, 4.0f);
    printf("  Misspec4x:  Standard=%.4f  Adaptive=%.4f (%+.1f%%)\n",
           rmse7s_base, rmse7s_adapt, 100.0*(rmse7s_adapt/rmse7s_base - 1.0));

    // 7d: Spike gauntlet
    SvData d7k = generate_sv(n_ticks, true_rho, true_sz, true_mu, seed + 8);
    inject_spike(&d7k, 500, 2.0f);
    inject_spike(&d7k, 1000, 4.0f);
    inject_spike(&d7k, 1500, 6.0f);
    double rmse7k_base = run_bpf_rmse(&d7k, N, true_rho, true_sz, true_mu,
                                       0, 0, seed, 1, NULL, NULL);
    double rmse7k_adapt = run_bpf_rmse_adaptive(&d7k, N, true_rho, true_sz, true_mu,
                                                  0, 0, seed, 2.0f, 4.0f);
    printf("  Spikes:     Standard=%.4f  Adaptive=%.4f (%+.1f%%)\n",
           rmse7k_base, rmse7k_adapt, 100.0*(rmse7k_adapt/rmse7k_base - 1.0));

    // 7e: Threshold sweep
    printf("\n  Threshold sweep (severe misspec 4x):\n");
    float thresholds[][2] = {
        {1.5f, 3.0f}, {2.0f, 4.0f}, {2.5f, 5.0f}, {3.0f, 6.0f}
    };
    for (int t = 0; t < 4; t++) {
        double rm = run_bpf_rmse_adaptive(&d7s, N, true_rho, true_sz, true_mu,
                                           0, 0, seed, thresholds[t][0], thresholds[t][1]);
        printf("    alert=%.1f panic=%.1f → RMSE=%.4f (%+.1f%%)\n",
               thresholds[t][0], thresholds[t][1],
               rm, 100.0*(rm/rmse7s_base - 1.0));
    }

    free_sv(&d7o);
    free_sv(&d7m);
    free_sv(&d7s);
    free_sv(&d7k);

    printf("\n=== Done ===\n");
    return 0;
}
