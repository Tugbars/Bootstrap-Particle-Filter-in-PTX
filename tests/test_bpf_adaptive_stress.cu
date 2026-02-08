// =============================================================================
// STRESS TEST: Standard BPF vs Adaptive BPF
//
// Reuses the brutal DGP scenarios from test_stress_compare.cu:
//   A: Spike Gauntlet (20σ→50σ)
//   B: Regime Teleport (vol 3%→78%)
//   C: Pure Chaos (random walk + 10% ±2 jumps)
//   D: Crypto Meltdown (t(3) state+obs, 2x σ_z)
//   E: Periodic Regimes (8 teleports, 1200 ticks)
//   F: Sawtooth Ramp (4x ramp+crash cycles)
//
// Each scenario run with oracle → mild → moderate → severe → extreme misspec.
//
// Build: nvcc -O3 test_bpf_adaptive_stress.cu gpu_bpf.cu -o test_bpf_stress -lcurand
// =============================================================================

#include "gpu_bpf.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>

// =============================================================================
// PRNG (identical to test_stress_compare)
// =============================================================================

static inline float randf(unsigned int* seed) {
    *seed = *seed * 1103515245 + 12345;
    return (float)((*seed >> 16) & 0x7FFF) / 32768.0f;
}

static inline float randn(unsigned int* seed) {
    float u1 = randf(seed) + 1e-10f;
    float u2 = randf(seed);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

static inline float rand_t(unsigned int* seed, float nu) {
    if (nu <= 0.0f || nu > 100.0f) return randn(seed);
    float z = randn(seed);
    float chi2 = 0.0f;
    for (int k = 0; k < (int)nu; k++) {
        float g = randn(seed);
        chi2 += g * g;
    }
    return z * sqrtf(nu / fmaxf(chi2, 1e-8f));
}

// =============================================================================
// DGP generation (identical to test_stress_compare)
// =============================================================================

struct SVDGPParams { float mu, rho, sigma; };

static void gen_sv(std::vector<float>& ret, std::vector<float>& th,
                   int n, float rho, float sigma_z, float mu,
                   float nu_state, float nu_obs,
                   float& h, unsigned int* seed) {
    for (int i = 0; i < n; i++) {
        float eps = (nu_state > 0) ? rand_t(seed, nu_state) : randn(seed);
        h = mu + rho * (h - mu) + sigma_z * eps;
        th.push_back(h);
        float eta = (nu_obs > 0) ? rand_t(seed, nu_obs) : randn(seed);
        ret.push_back(expf(h * 0.5f) * eta);
    }
}

static void gen_calm(std::vector<float>& ret, std::vector<float>& th,
                     int n, const SVDGPParams& dgp, float& h, unsigned int* seed) {
    gen_sv(ret, th, n, dgp.rho, dgp.sigma, dgp.mu, 0.0f, 0.0f, h, seed);
}

static void gen_spike(std::vector<float>& ret, std::vector<float>& th,
                      float h_jump, float& h, float mu, unsigned int* seed) {
    float elevated = mu + h_jump;
    if (h < elevated) h = elevated; else h += h_jump * 0.3f;
    th.push_back(h);
    ret.push_back(expf(h * 0.5f) * randn(seed));
}

static void gen_recovery(std::vector<float>& ret, std::vector<float>& th,
                         int n, const SVDGPParams& dgp, float& h, unsigned int* seed) {
    gen_calm(ret, th, n, dgp, h, seed);
}

// =============================================================================
// Scenario data
// =============================================================================

struct ScenarioData {
    std::string name;
    std::vector<float> returns;
    std::vector<float> true_h;
    int spike_t;
    int spike_window;
};

// =============================================================================
// Scenario generators (identical to test_stress_compare)
// =============================================================================

static ScenarioData make_spike_gauntlet(const SVDGPParams& dgp, unsigned int sv) {
    ScenarioData sc; sc.name = "Spike Gauntlet";
    unsigned int seed = sv; float h = dgp.mu;
    gen_calm(sc.returns, sc.true_h, 200, dgp, h, &seed);
    sc.spike_t = (int)sc.returns.size(); sc.spike_window = 250;
    float jumps[] = {1.0f, 1.5f, 2.0f, 2.5f};
    for (int s = 0; s < 4; s++) {
        h = dgp.mu + jumps[s];
        sc.true_h.push_back(h);
        sc.returns.push_back(expf(h * 0.5f) * randn(&seed));
        gen_recovery(sc.returns, sc.true_h, 50, dgp, h, &seed);
    }
    gen_calm(sc.returns, sc.true_h, 200, dgp, h, &seed);
    return sc;
}

static ScenarioData make_regime_teleport(const SVDGPParams& dgp, unsigned int sv) {
    ScenarioData sc; sc.name = "Regime Teleport";
    unsigned int seed = sv; float h = dgp.mu;
    gen_calm(sc.returns, sc.true_h, 200, dgp, h, &seed);
    sc.spike_t = (int)sc.returns.size(); sc.spike_window = 450;
    float mus[] = {-2.0f, -7.0f, -0.5f, -4.5f};
    for (int r = 0; r < 4; r++) {
        h = mus[r];
        gen_sv(sc.returns, sc.true_h, 100, dgp.rho, dgp.sigma, mus[r],
               0.0f, 0.0f, h, &seed);
    }
    gen_calm(sc.returns, sc.true_h, 200, dgp, h, &seed);
    return sc;
}

static ScenarioData make_pure_chaos(const SVDGPParams& dgp, unsigned int sv) {
    ScenarioData sc; sc.name = "Pure Chaos";
    unsigned int seed = sv; float h = dgp.mu;
    gen_calm(sc.returns, sc.true_h, 100, dgp, h, &seed);
    sc.spike_t = (int)sc.returns.size(); sc.spike_window = 200;
    for (int i = 0; i < 200; i++) {
        if (randf(&seed) < 0.10f)
            h += (randf(&seed) - 0.5f) * 4.0f;
        float eps = randn(&seed);
        h = dgp.mu + 0.5f * (h - dgp.mu) + dgp.sigma * 3.0f * eps;
        if (h > 2.0f) h = 2.0f; if (h < -10.0f) h = -10.0f;
        sc.true_h.push_back(h);
        sc.returns.push_back(expf(h * 0.5f) * randn(&seed));
    }
    gen_recovery(sc.returns, sc.true_h, 200, dgp, h, &seed);
    return sc;
}

static ScenarioData make_crypto_meltdown(const SVDGPParams& dgp, unsigned int sv) {
    ScenarioData sc; sc.name = "Crypto Meltdown";
    unsigned int seed = sv; float h = dgp.mu;
    gen_calm(sc.returns, sc.true_h, 100, dgp, h, &seed);
    sc.spike_t = (int)sc.returns.size(); sc.spike_window = 200;
    gen_sv(sc.returns, sc.true_h, 150, dgp.rho, dgp.sigma * 2.0f, dgp.mu,
           3.0f, 3.0f, h, &seed);
    gen_calm(sc.returns, sc.true_h, 250, dgp, h, &seed);
    return sc;
}

static ScenarioData make_periodic_regimes(const SVDGPParams& dgp, unsigned int sv) {
    ScenarioData sc; sc.name = "Periodic Regimes";
    unsigned int seed = sv; float h = dgp.mu;
    sc.spike_t = 0; sc.spike_window = 600;
    float mus[] = {-6.0f, -3.0f, -5.0f, -1.5f, -4.5f, -2.0f, -6.5f, -3.5f};
    for (int r = 0; r < 8; r++) {
        h = mus[r];
        gen_sv(sc.returns, sc.true_h, 150, dgp.rho, dgp.sigma, mus[r],
               0.0f, 0.0f, h, &seed);
    }
    return sc;
}

static ScenarioData make_sawtooth(const SVDGPParams& dgp, unsigned int sv) {
    ScenarioData sc; sc.name = "Sawtooth Ramp";
    unsigned int seed = sv; float h = dgp.mu;
    gen_calm(sc.returns, sc.true_h, 50, dgp, h, &seed);
    sc.spike_t = (int)sc.returns.size(); sc.spike_window = 400;
    for (int cyc = 0; cyc < 4; cyc++) {
        for (int i = 0; i < 100; i++) {
            float target = dgp.mu + 3.0f * (float)i / 100.0f;
            h = target + dgp.sigma * randn(&seed);
            sc.true_h.push_back(h);
            sc.returns.push_back(expf(h * 0.5f) * randn(&seed));
        }
        h = dgp.mu;
        sc.true_h.push_back(h);
        sc.returns.push_back(expf(h * 0.5f) * randn(&seed));
    }
    gen_calm(sc.returns, sc.true_h, 100, dgp, h, &seed);
    return sc;
}

// =============================================================================
// Metrics
// =============================================================================

struct Metrics {
    double rmse, spike_rmse, bias, max_err;
    bool had_nan;
};

// =============================================================================
// Run BPF (standard or adaptive depending on setup before call)
// =============================================================================

static Metrics run_bpf(const ScenarioData& sc,
                       float f_rho, float f_sigma_z, float f_mu,
                       float nu_obs, int n_particles, int seed,
                       bool adaptive) {
    Metrics m = {};
    int n = (int)sc.returns.size();

    GpuBpfState* state = gpu_bpf_create(n_particles, f_rho, f_sigma_z, f_mu,
                                         0.0f, nu_obs, seed);

    if (adaptive) {
        float cf[] = {0.99f, 0.01f};
        float cs[] = {1.0f,  4.0f};
        float af[] = {0.90f, 0.05f, 0.03f, 0.02f};
        float as[] = {1.0f,  2.0f,  4.0f,  8.0f};
        float pf[] = {0.70f, 0.15f, 0.10f, 0.05f};
        float ps[] = {1.0f,  2.0f,  5.0f,  12.0f};
        gpu_bpf_set_adaptive_bands(n_particles,
            cf, cs, 2, af, as, 4, pf, ps, 4, 2.0f, 4.0f);
    }

    int skip = 20;
    double sum_sq = 0, sum_bias = 0, spike_sq = 0, worst = 0;
    int count = 0, spike_n = 0;

    for (int t = 0; t < n; t++) {
        BpfResult r = gpu_bpf_step(state, sc.returns[t]);
        if (std::isnan(r.h_mean) || std::isinf(r.h_mean)) { m.had_nan = true; break; }
        if (t >= skip) {
            double err = (double)r.h_mean - (double)sc.true_h[t];
            sum_sq += err * err;
            sum_bias += err;
            if (fabs(err) > worst) worst = fabs(err);
            count++;
            if (t >= sc.spike_t && t < sc.spike_t + sc.spike_window) {
                spike_sq += err * err;
                spike_n++;
            }
        }
    }
    gpu_bpf_destroy(state);

    // Reset bands for next run
    float f1[] = {1.0f}; float s1[] = {1.0f};
    gpu_bpf_set_bands(n_particles, 1, f1, s1);

    if (count > 0 && !m.had_nan) {
        m.rmse = sqrt(sum_sq / count);
        m.bias = sum_bias / count;
        m.max_err = worst;
        m.spike_rmse = (spike_n > 0) ? sqrt(spike_sq / spike_n) : 0;
    }
    return m;
}

// =============================================================================
// Misspec configs
// =============================================================================

struct MisspecConfig { const char* label; float rho, sigma_z, mu; };

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    int N = 300000;
    float bnu = 50.0f;
    int seed = 42;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--n") && i+1 < argc) N = atoi(argv[++i]);
    }

    float true_rho = 0.98f, true_sz = 0.15f, true_mu = -4.5f;
    SVDGPParams dgp = {true_mu, true_rho, true_sz};

    MisspecConfig mc[] = {
        {"Oracle",    0.98f, 0.15f, -4.5f},
        {"Mild",      0.95f, 0.12f, -4.0f},
        {"Moderate",  0.90f, 0.10f, -3.5f},
        {"Severe",    0.80f, 0.05f, -3.0f},
        {"Extreme",   0.70f, 0.03f, -2.0f},
        {"Wrong μ",   0.98f, 0.15f, -6.5f},
        {"Wrong ρ",   0.80f, 0.15f, -4.5f},
        {"Wrong σ_z", 0.98f, 0.02f, -4.5f},
    };
    int n_mc = 8;

    printf("═══════════════════════════════════════════════════════════════════════════════\n");
    printf("  Standard BPF vs Adaptive BPF — Stress Test\n");
    printf("  Particles: %dK  ν_obs=%.0f  True DGP: ρ=%.2f σ_z=%.2f μ=%.1f\n",
           N/1000, bnu, true_rho, true_sz, true_mu);
    printf("═══════════════════════════════════════════════════════════════════════════════\n");

    // Generate all scenarios once
    ScenarioData scenarios[] = {
        make_spike_gauntlet(dgp, 42),
        make_regime_teleport(dgp, 43),
        make_pure_chaos(dgp, 44),
        make_crypto_meltdown(dgp, 45),
        make_periodic_regimes(dgp, 46),
        make_sawtooth(dgp, 47),
    };
    int n_sc = 6;

    // Accumulators
    double std_rmse_sum = 0, adp_rmse_sum = 0;
    double std_spike_sum = 0, adp_spike_sum = 0;
    int std_wins = 0, adp_wins = 0, total = 0;

    for (int s = 0; s < n_sc; s++) {
        printf("\n┌── %s (%d ticks) ", scenarios[s].name.c_str(),
               (int)scenarios[s].returns.size());
        for (int p = 0; p < (int)(60 - scenarios[s].name.size()); p++) printf("─");
        printf("┐\n");

        printf("│ %-12s │ %7s %7s │ %7s %7s │ %+7s %+7s │ %6s %6s │ %s\n",
               "Misspec", "Std", "Adapt", "SpStd", "SpAdp",
               "bStd", "bAdp", "mxStd", "mxAdp", "Δ RMSE");
        printf("│ ──────────── │ ─────── ─────── │ ─────── ─────── │ ─────── ─────── │ ────── ────── │ ──────\n");

        for (int m = 0; m < n_mc; m++) {
            Metrics std_m = run_bpf(scenarios[s], mc[m].rho, mc[m].sigma_z, mc[m].mu,
                                     bnu, N, seed, false);
            Metrics adp_m = run_bpf(scenarios[s], mc[m].rho, mc[m].sigma_z, mc[m].mu,
                                     bnu, N, seed, true);

            const char* winner = "";
            if (!std_m.had_nan && !adp_m.had_nan) {
                if (adp_m.rmse < std_m.rmse) { winner = "←"; adp_wins++; }
                else { winner = "→"; std_wins++; }
            }

            double pct = (!std_m.had_nan && !adp_m.had_nan && std_m.rmse > 0)
                ? 100.0 * (adp_m.rmse / std_m.rmse - 1.0) : 0.0;

            printf("│ %-12s │ %7.4f %7.4f │ %7.4f %7.4f │ %+7.3f %+7.3f │ %6.2f %6.2f │ %+5.1f%% %s\n",
                   mc[m].label,
                   std_m.had_nan ? 0.0 : std_m.rmse,
                   adp_m.had_nan ? 0.0 : adp_m.rmse,
                   std_m.had_nan ? 0.0 : std_m.spike_rmse,
                   adp_m.had_nan ? 0.0 : adp_m.spike_rmse,
                   std_m.had_nan ? 0.0 : std_m.bias,
                   adp_m.had_nan ? 0.0 : adp_m.bias,
                   std_m.had_nan ? 0.0 : std_m.max_err,
                   adp_m.had_nan ? 0.0 : adp_m.max_err,
                   pct, winner);

            if (!std_m.had_nan) { std_rmse_sum += std_m.rmse; std_spike_sum += std_m.spike_rmse; }
            if (!adp_m.had_nan) { adp_rmse_sum += adp_m.rmse; adp_spike_sum += adp_m.spike_rmse; }
            total++;
        }
    }

    // Grand summary
    printf("\n═══════════════════════════════════════════════════════════════════════════════\n");
    printf("  GRAND SUMMARY (%d scenarios × %d misspec configs = %d runs)\n",
           n_sc, n_mc, total);
    printf("  ─────────────────────────────────────────────\n");
    printf("  %-20s %10s %10s\n",                   "", "Standard", "Adaptive");
    printf("  %-20s %10.4f %10.4f  (%+.1f%%)\n", "Avg RMSE",
           std_rmse_sum/total, adp_rmse_sum/total,
           100.0*(adp_rmse_sum/std_rmse_sum - 1.0));
    printf("  %-20s %10.4f %10.4f  (%+.1f%%)\n", "Avg Spike RMSE",
           std_spike_sum/total, adp_spike_sum/total,
           100.0*(adp_spike_sum/std_spike_sum - 1.0));
    printf("  %-20s %10d %10d\n", "Wins", std_wins, adp_wins);
    printf("═══════════════════════════════════════════════════════════════════════════════\n");

    return 0;
}
