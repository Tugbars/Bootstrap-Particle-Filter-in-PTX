/**
 * @file test_jump_bands.cu
 * @brief Combined jump diffusion + adaptive bands test
 *
 * Four configurations across standard scenarios:
 *   A. Standard BPF (no jumps, no bands)
 *   B. Jumps only   (K15, fixed lambda=0.03, sigma_J=2.5)
 *   C. Bands only   (adaptive: calm 99/1, alert 90/5/3/2, panic 70/15/10/5)
 *   D. Jumps + Bands (both enabled simultaneously)
 *
 * Scenarios:
 *   1. Oracle (matched DGP)
 *   2. Misspec 2x sigma_z
 *   3. Misspec 4x sigma_z
 *   4. Spike gauntlet (3 sudden vol jumps)
 *   5. Regime teleport (sigma_z flips between 0.1 and 0.6)
 *   6. Combined stress: misspec 2x + spikes
 *
 * Build (with PTX):
 *   cmake --build build --target test_jump_bands
 *
 * Usage:
 *   ./test_jump_bands [n_particles]
 */

#include "gpu_bpf_full.cuh"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// =============================================================================
// DGP: identical to test_mix_bands / test_jump_stress
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
    float u1 = (float)(test_pcg32() >> 9) / 8388608.0f;
    float u2 = (float)(test_pcg32() >> 9) / 8388608.0f;
    if (u1 < 1e-10f) u1 = 1e-10f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

typedef struct {
    double* h;
    double* returns;
    int     n;
} SvData;

static SvData generate_sv(int n_ticks, float rho, float sigma_z, float mu, int seed) {
    test_rng = (unsigned long long)seed * 999983ULL + 1ULL;
    SvData d;
    d.n = n_ticks;
    d.h       = (double*)malloc(n_ticks * sizeof(double));
    d.returns = (double*)malloc(n_ticks * sizeof(double));

    float std_stat = sqrtf(sigma_z * sigma_z / fmaxf(1.0f - rho * rho, 1e-6f));
    d.h[0] = mu + std_stat * test_randn();
    d.returns[0] = expf((float)d.h[0] * 0.5f) * test_randn();

    for (int t = 1; t < n_ticks; t++) {
        d.h[t] = mu + rho * (d.h[t - 1] - mu) + sigma_z * test_randn();
        d.returns[t] = expf((float)d.h[t] * 0.5f) * test_randn();
    }
    return d;
}

static void inject_spike(SvData* d, int tick, float new_h) {
    if (tick < d->n) {
        d->h[tick] = new_h;
        d->returns[tick] = expf(new_h * 0.5f) * test_randn();
    }
}

// Regime teleport: sigma_z alternates between low and high every block_len ticks
static SvData generate_sv_regime_teleport(int n_ticks, float rho, float mu,
                                           float sz_low, float sz_high,
                                           int block_len, int seed) {
    test_rng = (unsigned long long)seed * 999983ULL + 1ULL;
    SvData d;
    d.n = n_ticks;
    d.h       = (double*)malloc(n_ticks * sizeof(double));
    d.returns = (double*)malloc(n_ticks * sizeof(double));

    float std_stat = sqrtf(sz_low * sz_low / fmaxf(1.0f - rho * rho, 1e-6f));
    d.h[0] = mu + std_stat * test_randn();
    d.returns[0] = expf((float)d.h[0] * 0.5f) * test_randn();

    for (int t = 1; t < n_ticks; t++) {
        float sz = ((t / block_len) % 2 == 0) ? sz_low : sz_high;
        d.h[t] = mu + rho * (d.h[t - 1] - mu) + sz * test_randn();
        d.returns[t] = expf((float)d.h[t] * 0.5f) * test_randn();
    }
    return d;
}

static void free_sv(SvData* d) {
    free(d->h);
    free(d->returns);
}

// =============================================================================
// Configuration modes
// =============================================================================

typedef enum {
    MODE_STANDARD  = 0,   // no jumps, no bands
    MODE_JUMPS     = 1,   // kernel 15 only
    MODE_BANDS     = 2,   // adaptive bands only
    MODE_COMBINED  = 3    // jumps + adaptive bands
} RunMode;

static const char* mode_name(RunMode m) {
    switch (m) {
        case MODE_STANDARD: return "Standard";
        case MODE_JUMPS:    return "Jumps";
        case MODE_BANDS:    return "Bands";
        case MODE_COMBINED: return "J+B";
    }
    return "?";
}

// =============================================================================
// Enable adaptive bands (helper)
// =============================================================================

static void enable_adaptive_bands(int n_particles) {
    // Calm: 99% standard, 1% insurance at 4x
    float calm_f[]  = {0.99f, 0.01f};
    float calm_s[]  = {1.0f,  4.0f};

    // Alert: 90% standard + moderate exploration
    float alert_f[] = {0.90f, 0.05f, 0.03f, 0.02f};
    float alert_s[] = {1.0f,  2.0f,  4.0f,  8.0f};

    // Panic: 70% standard + aggressive exploration
    float panic_f[] = {0.70f, 0.15f, 0.10f, 0.05f};
    float panic_s[] = {1.0f,  2.0f,  5.0f,  12.0f};

    gpu_bpf_set_adaptive_bands(n_particles,
        calm_f,  calm_s,  2,
        alert_f, alert_s, 4,
        panic_f, panic_s, 4,
        2.0f, 4.0f);
}

static void disable_bands(int n_particles) {
    float f1[] = {1.0f};
    float s1[] = {1.0f};
    gpu_bpf_set_bands(n_particles, 1, f1, s1);
}

// =============================================================================
// Run BPF in a given mode, return RMSE
// =============================================================================

static double run_bpf(const SvData* d, int n_particles,
                      float rho, float sigma_z, float mu,
                      int seed, RunMode mode) {
    GpuBpfState* s = gpu_bpf_create(n_particles, rho, sigma_z, mu,
                                     0, 0, seed);

    // Configure jumps
    if (mode == MODE_JUMPS || mode == MODE_COMBINED) {
        gpu_bpf_enable_jump_diffusion(s, 2.5f);
        gpu_bpf_set_jump_lambda_fixed(s, 0.03f);
    }

    // Configure bands
    if (mode == MODE_BANDS || mode == MODE_COMBINED) {
        enable_adaptive_bands(n_particles);
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
    disable_bands(n_particles);

    return sqrt(sum_sq / fmax(count, 1));
}

// =============================================================================
// Run one scenario across all 4 modes
// =============================================================================

typedef struct {
    double rmse[4];   // indexed by RunMode
} ScenarioResult;

static ScenarioResult run_scenario(const SvData* d, int n_particles,
                                    float rho, float sigma_z, float mu,
                                    int seed) {
    ScenarioResult res;
    for (int m = 0; m < 4; m++) {
        res.rmse[m] = run_bpf(d, n_particles, rho, sigma_z, mu,
                               seed, (RunMode)m);
    }
    return res;
}

static void print_scenario(const char* name, const ScenarioResult* r) {
    double base = r->rmse[MODE_STANDARD];
    printf("  %-22s", name);
    for (int m = 0; m < 4; m++) {
        if (m == 0) {
            printf("  %8.4f", r->rmse[m]);
        } else {
            double pct = 100.0 * (r->rmse[m] / base - 1.0);
            printf("  %8.4f (%+5.1f%%)", r->rmse[m], pct);
        }
    }

    // Mark winner
    int best = 0;
    for (int m = 1; m < 4; m++) {
        if (r->rmse[m] < r->rmse[best]) best = m;
    }
    if (best > 0) {
        printf("  <- %s", mode_name((RunMode)best));
    }
    printf("\n");
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 40000;
    int n_ticks = 3000;
    int seed = 42;

    float true_rho = 0.98f, true_sz = 0.15f, true_mu = -1.0f;

    printf("══════════════════════════════════════════════════════════════════════\n");
    printf("  COMBINED TEST: Jump Diffusion + Adaptive Bands\n");
    printf("══════════════════════════════════════════════════════════════════════\n");
    printf("  Particles: %d   Ticks: %d\n", N, n_ticks);
    printf("  Jumps: lambda=0.03 fixed, sigma_J=2.5\n");
    printf("  Bands: adaptive (calm 99/1, alert 90/5/3/2, panic 70/15/10/5)\n");
    printf("  DGP: rho=%.2f, sigma_z=%.2f, mu=%.1f\n\n", true_rho, true_sz, true_mu);

    // Header
    printf("  %-22s  %8s  %18s  %18s  %18s\n",
           "Scenario", "Standard", "Jumps", "Bands", "J+B");
    printf("  ──────────────────────  ────────  ──────────────────  "
           "──────────────────  ──────────────────\n");

    // ── Scenario 1: Oracle ──
    SvData d1 = generate_sv(n_ticks, true_rho, true_sz, true_mu, seed);
    ScenarioResult r1 = run_scenario(&d1, N, true_rho, true_sz, true_mu, seed);
    print_scenario("Oracle", &r1);
    free_sv(&d1);

    // ── Scenario 2: Misspec 2x ──
    SvData d2 = generate_sv(n_ticks, true_rho, true_sz * 2.0f, true_mu, seed + 1);
    ScenarioResult r2 = run_scenario(&d2, N, true_rho, true_sz, true_mu, seed);
    print_scenario("Misspec 2x", &r2);
    free_sv(&d2);

    // ── Scenario 3: Misspec 4x ──
    SvData d3 = generate_sv(n_ticks, true_rho, true_sz * 4.0f, true_mu, seed + 2);
    ScenarioResult r3 = run_scenario(&d3, N, true_rho, true_sz, true_mu, seed);
    print_scenario("Misspec 4x", &r3);
    free_sv(&d3);

    // ── Scenario 4: Spike gauntlet ──
    SvData d4 = generate_sv(n_ticks, true_rho, true_sz, true_mu, seed + 3);
    inject_spike(&d4, 500,  2.0f);
    inject_spike(&d4, 1000, 4.0f);
    inject_spike(&d4, 1500, 6.0f);
    ScenarioResult r4 = run_scenario(&d4, N, true_rho, true_sz, true_mu, seed);
    print_scenario("Spike gauntlet", &r4);
    free_sv(&d4);

    // ── Scenario 5: Regime teleport ──
    SvData d5 = generate_sv_regime_teleport(n_ticks, true_rho, true_mu,
                                             0.10f, 0.60f, 200, seed + 5);
    ScenarioResult r5 = run_scenario(&d5, N, true_rho, true_sz, true_mu, seed);
    print_scenario("Regime teleport", &r5);
    free_sv(&d5);

    // ── Scenario 6: Misspec 2x + spikes ──
    SvData d6 = generate_sv(n_ticks, true_rho, true_sz * 2.0f, true_mu, seed + 6);
    inject_spike(&d6, 400,  3.0f);
    inject_spike(&d6, 800,  5.0f);
    inject_spike(&d6, 1200, 4.0f);
    inject_spike(&d6, 1800, 6.0f);
    ScenarioResult r6 = run_scenario(&d6, N, true_rho, true_sz, true_mu, seed);
    print_scenario("Misspec2x + spikes", &r6);
    free_sv(&d6);

    // ── Summary ──
    printf("\n  ──────────────────────────────────────────────────────────────────\n");
    printf("  Summary: win counts across %d scenarios\n", 6);
    int wins[4] = {0, 0, 0, 0};
    const ScenarioResult* all[] = {&r1, &r2, &r3, &r4, &r5, &r6};
    for (int s = 0; s < 6; s++) {
        int best = 0;
        for (int m = 1; m < 4; m++) {
            if (all[s]->rmse[m] < all[s]->rmse[best]) best = m;
        }
        wins[best]++;
    }
    printf("    Standard: %d   Jumps: %d   Bands: %d   J+B: %d\n",
           wins[0], wins[1], wins[2], wins[3]);

    // Average relative improvement vs standard
    printf("\n  Average RMSE change vs Standard:\n");
    for (int m = 1; m < 4; m++) {
        double sum_pct = 0.0;
        for (int s = 0; s < 6; s++) {
            sum_pct += 100.0 * (all[s]->rmse[m] / all[s]->rmse[MODE_STANDARD] - 1.0);
        }
        printf("    %-10s: %+.1f%% avg\n", mode_name((RunMode)m), sum_pct / 6.0);
    }

    // Does J+B beat the better of J and B individually?
    printf("\n  J+B vs best-of(J,B) per scenario:\n");
    for (int s = 0; s < 6; s++) {
        double best_single = fmin(all[s]->rmse[MODE_JUMPS], all[s]->rmse[MODE_BANDS]);
        double combined    = all[s]->rmse[MODE_COMBINED];
        double delta = 100.0 * (combined / best_single - 1.0);
        const char* names[] = {"Oracle", "Misspec2x", "Misspec4x",
                               "Spikes", "RegTeleport", "Mis2x+Spk"};
        printf("    %-14s  best_single=%.4f  J+B=%.4f  (%+.1f%%) %s\n",
               names[s], best_single, combined, delta,
               (combined < best_single) ? "SYNERGY" : "no synergy");
    }

    printf("\n══════════════════════════════════════════════════════════════════════\n");
    return 0;
}
