// =============================================================================
// STRESS TEST: Combined BPF vs Combined + Jump Diffusion (Bernoulli MIM)
//
// Phase 1:  sigma_J sweep (fixed lambda=0.02)
// Phase 1b: lambda sweep at best sigma_J
// Phase 2:  Full stress test at best (lambda, sigma_J)
//
// Jump model: each particle draws J_t ~ Bernoulli(lambda) independently.
// If jump: h[i] += sigma_J * N(0,1). Jumpers uniformly distributed across
// all indices = natural overlap with adaptive sigma_z bands.
// =============================================================================

#include "gpu_bpf_full.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>

// =============================================================================
// PRNG
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
// DGP
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

static void gen_recovery(std::vector<float>& ret, std::vector<float>& th,
                         int n, const SVDGPParams& dgp, float& h, unsigned int* seed) {
    gen_calm(ret, th, n, dgp, h, seed);
}

// =============================================================================
// Scenarios
// =============================================================================

struct ScenarioData {
    std::string name;
    std::vector<float> returns;
    std::vector<float> true_h;
    int spike_t, spike_window;
};

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
    float  final_mu, final_rho;
    bool   had_nan;
};

// =============================================================================
// Run BPF (combined = adaptive bands + NatGrad)
// enable_jump: if true, calls gpu_bpf_enable_jump_diffusion
// =============================================================================

static Metrics run_bpf(const ScenarioData& sc,
                       float f_rho, float f_sigma_z, float f_mu,
                       float nu_obs, int n_particles, int seed,
                       bool enable_jump, float lambda, float sigma_J) {
    Metrics m = {};
    m.final_mu = f_mu; m.final_rho = f_rho;
    int n = (int)sc.returns.size();

    GpuBpfState* state = gpu_bpf_create(n_particles, f_rho, f_sigma_z, f_mu,
                                         0.0f, nu_obs, seed);

    /* Adaptive bands */
    float cf[] = {0.99f, 0.01f};
    float cs[] = {1.0f,  4.0f};
    float af[] = {0.90f, 0.05f, 0.03f, 0.02f};
    float as[] = {1.0f,  2.0f,  4.0f,  8.0f};
    float pf[] = {0.70f, 0.15f, 0.10f, 0.05f};
    float ps[] = {1.0f,  2.0f,  5.0f,  12.0f};
    gpu_bpf_set_adaptive_bands(n_particles,
        cf, cs, 2, af, as, 4, pf, ps, 4, 2.0f, 4.0f);

    /* NatGrad mu/rho learning */
    gpu_bpf_enable_mu_learning(state, 1, 50, 0.1f, 10.0f, 0.667f);
    gpu_bpf_enable_rho_learning(state, 1);
    gpu_bpf_set_ess_threshold(state, 0.5f);

    /* Bernoulli jump diffusion */
    if (enable_jump) {
        gpu_bpf_enable_jump_diffusion(state, lambda, sigma_J, seed + 9999);
    }

    int skip = 20;
    double sum_sq = 0, sum_bias = 0, spike_sq = 0, worst = 0;
    int count = 0, spike_n = 0;

    for (int t = 0; t < n; t++) {
        BpfResult r = gpu_bpf_step(state, sc.returns[t]);
        if (std::isnan(r.h_mean) || std::isinf(r.h_mean)) { m.had_nan = true; break; }
        if (t >= skip) {
            double err = (double)r.h_mean - (double)sc.true_h[t];
            sum_sq += err * err; sum_bias += err;
            if (fabs(err) > worst) worst = fabs(err);
            count++;
            if (t >= sc.spike_t && t < sc.spike_t + sc.spike_window) {
                spike_sq += err * err; spike_n++;
            }
        }
    }

    m.final_mu = gpu_bpf_get_mu(state);
    m.final_rho = gpu_bpf_get_rho(state);
    gpu_bpf_destroy(state);

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
    int   N    = 40000;
    float bnu  = 50.0f;
    int   seed = 42;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--n") && i+1 < argc) N = atoi(argv[++i]);
        if (!strcmp(argv[i], "--seed") && i+1 < argc) seed = atoi(argv[++i]);
    }

    float true_rho = 0.98f, true_sz = 0.15f, true_mu = -4.5f;
    SVDGPParams dgp = {true_mu, true_rho, true_sz};

    printf("\n");
    printf("══════════════════════════════════════════════════════════════════════════════════\n");
    printf("  Combined BPF vs Combined + Bernoulli Jump Diffusion (MIM)\n");
    printf("──────────────────────────────────────────────────────────────────────────────────\n");
    printf("  Particles: %dK   nu_obs=%.0f   True DGP: rho=%.2f sigma_z=%.2f mu=%.1f\n",
           N/1000, bnu, true_rho, true_sz, true_mu);
    printf("══════════════════════════════════════════════════════════════════════════════════\n");

    // =========================================================================
    // PHASE 1: sigma_J sweep on Spike Gauntlet + Regime Teleport (oracle)
    //
    // Fix lambda=0.02. Sweep sigma_J from 0.3 to 3.0.
    // Find the sigma_J that minimizes spike RMSE without hurting calm RMSE.
    // =========================================================================

    printf("\n  ─── PHASE 1: sigma_J sweep (lambda=0.02, oracle misspec) ───\n\n");

    ScenarioData spike_sc = make_spike_gauntlet(dgp, 42);
    ScenarioData regime_sc = make_regime_teleport(dgp, 43);

    /* Combined baseline (no jump) */
    Metrics cb_spike = run_bpf(spike_sc, 0.98f, 0.15f, -4.5f, bnu, N, seed,
                               false, 0, 0);
    Metrics cb_regime = run_bpf(regime_sc, 0.98f, 0.15f, -4.5f, bnu, N, seed,
                                false, 0, 0);

    printf("  Baseline (no jump):\n");
    printf("    Spike Gauntlet: RMSE=%.4f  Spike=%.4f\n", cb_spike.rmse, cb_spike.spike_rmse);
    printf("    Regime Teleport: RMSE=%.4f  Spike=%.4f\n\n", cb_regime.rmse, cb_regime.spike_rmse);

    float lambda_fixed = 0.02f;
    float sigma_Js[] = {0.3f, 0.5f, 0.75f, 1.0f, 1.25f, 1.5f, 2.0f, 2.5f, 3.0f};
    int n_sJ = 9;

    printf("  %8s | %8s %8s %8s | %8s %8s %8s\n",
           "sigma_J", "SpkRMSE", "SpkSpike", "SpkDelta",
                      "RegRMSE", "RegSpike", "RegDelta");
    printf("  ──────── | ──────── ──────── ──────── | ──────── ──────── ────────\n");

    float best_sJ = 1.0f;
    double best_combined_spike = 1e9;

    for (int i = 0; i < n_sJ; i++) {
        float sJ = sigma_Js[i];
        Metrics jd_s = run_bpf(spike_sc, 0.98f, 0.15f, -4.5f, bnu, N, seed,
                               true, lambda_fixed, sJ);
        Metrics jd_r = run_bpf(regime_sc, 0.98f, 0.15f, -4.5f, bnu, N, seed,
                               true, lambda_fixed, sJ);

        double spk_delta = 100.0 * (jd_s.spike_rmse / cb_spike.spike_rmse - 1.0);
        double reg_delta = 100.0 * (jd_r.spike_rmse / cb_regime.spike_rmse - 1.0);

        printf("  %8.2f | %8.4f %8.4f %+7.1f%% | %8.4f %8.4f %+7.1f%%\n",
               sJ,
               jd_s.rmse, jd_s.spike_rmse, spk_delta,
               jd_r.rmse, jd_r.spike_rmse, reg_delta);

        /* Best = lowest average spike RMSE across both scenarios */
        double avg_spike = (jd_s.spike_rmse + jd_r.spike_rmse) / 2.0;
        if (avg_spike < best_combined_spike) {
            best_combined_spike = avg_spike;
            best_sJ = sJ;
        }
    }

    printf("\n  Best sigma_J = %.2f (avg spike RMSE = %.4f)\n", best_sJ, best_combined_spike);

    // =========================================================================
    // PHASE 1b: lambda sweep at best sigma_J
    // =========================================================================

    printf("\n  ─── PHASE 1b: lambda sweep (sigma_J=%.2f, oracle misspec) ───\n\n", best_sJ);

    float lambdas[] = {0.005f, 0.01f, 0.02f, 0.03f, 0.05f, 0.08f, 0.10f, 0.15f};
    int n_lam = 8;

    printf("  %8s | %8s %8s %8s | %8s %8s %8s\n",
           "lambda", "SpkRMSE", "SpkSpike", "SpkDelta",
                     "RegRMSE", "RegSpike", "RegDelta");
    printf("  ──────── | ──────── ──────── ──────── | ──────── ──────── ────────\n");

    float best_lam = 0.02f;
    double best_combined_spike2 = 1e9;

    for (int i = 0; i < n_lam; i++) {
        float lam = lambdas[i];
        Metrics jd_s = run_bpf(spike_sc, 0.98f, 0.15f, -4.5f, bnu, N, seed,
                               true, lam, best_sJ);
        Metrics jd_r = run_bpf(regime_sc, 0.98f, 0.15f, -4.5f, bnu, N, seed,
                               true, lam, best_sJ);

        double spk_delta = 100.0 * (jd_s.spike_rmse / cb_spike.spike_rmse - 1.0);
        double reg_delta = 100.0 * (jd_r.spike_rmse / cb_regime.spike_rmse - 1.0);

        printf("  %8.3f | %8.4f %8.4f %+7.1f%% | %8.4f %8.4f %+7.1f%%\n",
               lam,
               jd_s.rmse, jd_s.spike_rmse, spk_delta,
               jd_r.rmse, jd_r.spike_rmse, reg_delta);

        double avg_spike = (jd_s.spike_rmse + jd_r.spike_rmse) / 2.0;
        if (avg_spike < best_combined_spike2) {
            best_combined_spike2 = avg_spike;
            best_lam = lam;
        }
    }

    printf("\n  Best lambda = %.3f (avg spike RMSE = %.4f)\n", best_lam, best_combined_spike2);

    // =========================================================================
    // PHASE 2: Full stress test at best (lambda, sigma_J)
    // =========================================================================

    printf("\n");
    printf("══════════════════════════════════════════════════════════════════════════════════\n");
    printf("  PHASE 2: Full stress test  lambda=%.3f  sigma_J=%.2f\n", best_lam, best_sJ);
    printf("══════════════════════════════════════════════════════════════════════════════════\n");

    MisspecConfig mc[] = {
        {"Oracle",    0.98f, 0.15f, -4.5f},
        {"Mild",      0.95f, 0.12f, -4.0f},
        {"Moderate",  0.90f, 0.10f, -3.5f},
        {"Severe",    0.80f, 0.05f, -3.0f},
        {"Extreme",   0.70f, 0.03f, -2.0f},
        {"Wrong mu",  0.98f, 0.15f, -6.5f},
        {"Wrong rho", 0.80f, 0.15f, -4.5f},
        {"Wrong sz",  0.98f, 0.02f, -4.5f},
    };
    int n_mc = 8;

    ScenarioData scenarios[] = {
        make_spike_gauntlet(dgp, 42),
        make_regime_teleport(dgp, 43),
        make_pure_chaos(dgp, 44),
        make_crypto_meltdown(dgp, 45),
        make_periodic_regimes(dgp, 46),
        make_sawtooth(dgp, 47),
    };
    int n_sc = 6;

    double cb_rmse_sum = 0, jd_rmse_sum = 0;
    double cb_spike_sum = 0, jd_spike_sum = 0;
    int cb_wins = 0, jd_wins = 0, total = 0;

    for (int s = 0; s < n_sc; s++) {
        const ScenarioData& sc = scenarios[s];
        printf("\n  ═══ %s (%d ticks) ", sc.name.c_str(), (int)sc.returns.size());
        for (int p = 0; p < (int)(60 - sc.name.size()); p++) printf("═");
        printf("\n\n");

        printf("  %-12s | %8s %8s | %8s %8s | Winner\n",
               "Misspec", "CbRMSE", "CbSpike", "JdRMSE", "JdSpike");
        printf("  ──────────── | ──────── ──────── | ──────── ──────── | ──────\n");

        for (int mi = 0; mi < n_mc; mi++) {
            Metrics cb = run_bpf(sc, mc[mi].rho, mc[mi].sigma_z, mc[mi].mu,
                                 bnu, N, seed, false, 0, 0);
            Metrics jd = run_bpf(sc, mc[mi].rho, mc[mi].sigma_z, mc[mi].mu,
                                 bnu, N, seed, true, best_lam, best_sJ);

            const char* winner = "???";
            if (!cb.had_nan && !jd.had_nan) {
                if (cb.rmse <= jd.rmse) { winner = "Combo"; cb_wins++; }
                else                    { winner = "Jump";  jd_wins++; }
            }

            printf("  %-12s | %8.4f %8.4f | %8.4f %8.4f | %s",
                   mc[mi].label,
                   cb.had_nan ? 0.0 : cb.rmse,
                   cb.had_nan ? 0.0 : cb.spike_rmse,
                   jd.had_nan ? 0.0 : jd.rmse,
                   jd.had_nan ? 0.0 : jd.spike_rmse,
                   winner);

            if (mi > 0 && !cb.had_nan && !jd.had_nan) {
                printf("  cb:mu=%.2f,rho=%.3f  jd:mu=%.2f,rho=%.3f",
                       cb.final_mu, cb.final_rho, jd.final_mu, jd.final_rho);
            }
            printf("\n");

            if (!cb.had_nan) { cb_rmse_sum += cb.rmse; cb_spike_sum += cb.spike_rmse; }
            if (!jd.had_nan) { jd_rmse_sum += jd.rmse; jd_spike_sum += jd.spike_rmse; }
            total++;
        }
    }

    /* Grand summary */
    printf("\n");
    printf("══════════════════════════════════════════════════════════════════════════════════\n");
    printf("  GRAND SUMMARY (%d scenarios x %d misspec = %d runs)\n", n_sc, n_mc, total);
    printf("  Jump params: lambda=%.3f  sigma_J=%.2f\n", best_lam, best_sJ);
    printf("──────────────────────────────────────────────────────────────────────────────────\n");
    printf("  %-22s %14s %14s\n",     "", "Combined", "Combined+Jump");
    printf("  %-22s %14.4f %14.4f\n", "Avg RMSE",
           cb_rmse_sum/total, jd_rmse_sum/total);
    printf("  %-22s %14.4f %14.4f\n", "Avg Spike RMSE",
           cb_spike_sum/total, jd_spike_sum/total);
    printf("  %-22s %14d %14d\n",     "Wins", cb_wins, jd_wins);

    printf("──────────────────────────────────────────────────────────────────────────────────\n");
    if (cb_rmse_sum > 0) {
        printf("  Jump vs Combined RMSE:  %+.1f%%\n",
               100.0 * (jd_rmse_sum / cb_rmse_sum - 1.0));
        printf("  Jump vs Combined Spike: %+.1f%%\n",
               100.0 * (jd_spike_sum / cb_spike_sum - 1.0));
    }
    printf("══════════════════════════════════════════════════════════════════════════════════\n\n");

    return 0;
}
