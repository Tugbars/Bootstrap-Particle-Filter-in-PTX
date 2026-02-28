// =============================================================================
// STRESS TEST: Combined BPF vs Combined + Jump Diffusion (PTX Kernel 15)
//
// Phase 1:  sigma_J sweep (fixed lambda=0.02)
// Phase 1b: lambda sweep at best sigma_J
// Phase 2:  Full stress test at best fixed (lambda, sigma_J)
// Phase 3:  Regime-adaptive lambda vs fixed lambda
// Phase 4:  Lambda triple sweep (calm × alert × panic)
//
// Jump model: PTX kernel 15, Bernoulli MIM with Acklam ICDF.
// Regime-adaptive lambda: calm=0.01, alert=0.03, panic=0.08 (default).
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
//
// Jump modes:
//   enable_jump=false           → no jump
//   enable_jump=true, lambda>0  → fixed lambda (all regimes same)
//   enable_jump=true, lambda<=0 → per-regime lambdas from lam_calm/alert/panic
// =============================================================================

static Metrics run_bpf(const ScenarioData& sc,
                       float f_rho, float f_sigma_z, float f_mu,
                       float nu_obs, int n_particles, int seed,
                       bool enable_jump, float lambda, float sigma_J,
                       float lam_calm = -1.f, float lam_alert = -1.f,
                       float lam_panic = -1.f) {
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

    /* Jump diffusion via PTX kernel 15 */
    if (enable_jump) {
        gpu_bpf_enable_jump_diffusion(state, sigma_J);
        if (lam_calm > 0 && lam_alert > 0 && lam_panic > 0) {
            gpu_bpf_set_jump_lambdas(state, lam_calm, lam_alert, lam_panic);
        } else if (lambda > 0) {
            gpu_bpf_set_jump_lambda_fixed(state, lambda);
        }
        /* else: default regime-adaptive (0.01/0.03/0.08) */
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
    printf("  Combined BPF vs Combined + Jump (PTX Kernel 15, Bernoulli MIM)\n");
    printf("──────────────────────────────────────────────────────────────────────────────────\n");
    printf("  Particles: %dK   nu_obs=%.0f   True DGP: rho=%.2f sigma_z=%.2f mu=%.1f\n",
           N/1000, bnu, true_rho, true_sz, true_mu);
    printf("══════════════════════════════════════════════════════════════════════════════════\n");

    // =========================================================================
    // PHASE 1: sigma_J sweep on Spike Gauntlet + Regime Teleport (oracle)
    // =========================================================================

    printf("\n  ─── PHASE 1: sigma_J sweep (fixed lambda=0.02, oracle misspec) ───\n\n");

    ScenarioData spike_sc = make_spike_gauntlet(dgp, 42);
    ScenarioData regime_sc = make_regime_teleport(dgp, 43);

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
    // PHASE 2: Full stress test at best fixed (lambda, sigma_J)
    // =========================================================================

    printf("\n");
    printf("══════════════════════════════════════════════════════════════════════════════════\n");
    printf("  PHASE 2: Full stress test  fixed lambda=%.3f  sigma_J=%.2f\n", best_lam, best_sJ);
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

    printf("\n");
    printf("══════════════════════════════════════════════════════════════════════════════════\n");
    printf("  PHASE 2 SUMMARY (%d scenarios x %d misspec = %d runs)\n", n_sc, n_mc, total);
    printf("  Fixed: lambda=%.3f  sigma_J=%.2f\n", best_lam, best_sJ);
    printf("──────────────────────────────────────────────────────────────────────────────────\n");
    printf("  %-22s %14s %14s\n",     "", "Combined", "Combined+Jump");
    printf("  %-22s %14.4f %14.4f\n", "Avg RMSE",
           cb_rmse_sum/total, jd_rmse_sum/total);
    printf("  %-22s %14.4f %14.4f\n", "Avg Spike RMSE",
           cb_spike_sum/total, jd_spike_sum/total);
    printf("  %-22s %14d %14d\n",     "Wins", cb_wins, jd_wins);
    printf("──────────────────────────────────────────────────────────────────────────────────\n");
    if (cb_rmse_sum > 0) {
        printf("  Fixed Jump vs Combined RMSE:  %+.1f%%\n",
               100.0 * (jd_rmse_sum / cb_rmse_sum - 1.0));
        printf("  Fixed Jump vs Combined Spike: %+.1f%%\n",
               100.0 * (jd_spike_sum / cb_spike_sum - 1.0));
    }
    printf("══════════════════════════════════════════════════════════════════════════════════\n");

    // =========================================================================
    // PHASE 3: Regime-adaptive lambda vs fixed lambda
    // =========================================================================

    printf("\n");
    printf("══════════════════════════════════════════════════════════════════════════════════\n");
    printf("  PHASE 3: Regime-adaptive lambda  sigma_J=%.2f\n", best_sJ);
    printf("  Calm=0.01  Alert=0.03  Panic=0.08\n");
    printf("══════════════════════════════════════════════════════════════════════════════════\n");

    double ad_rmse_sum = 0, ad_spike_sum = 0;
    int ad_wins_vs_cb = 0, ad_wins_vs_fix = 0;
    int fix_wins_vs_ad = 0, total3 = 0;

    for (int s = 0; s < n_sc; s++) {
        const ScenarioData& sc = scenarios[s];
        printf("\n  ═══ %s ", sc.name.c_str());
        for (int p = 0; p < (int)(65 - sc.name.size()); p++) printf("═");
        printf("\n\n");

        printf("  %-12s | %8s %8s | %8s %8s | %8s %8s | Best\n",
               "Misspec", "CbRMSE", "CbSpike", "FixRMSE", "FixSpike",
                          "AdpRMSE", "AdpSpike");
        printf("  ──────────── | ──────── ──────── | ──────── ──────── | ──────── ──────── | ────\n");

        for (int mi = 0; mi < n_mc; mi++) {
            Metrics cb  = run_bpf(sc, mc[mi].rho, mc[mi].sigma_z, mc[mi].mu,
                                  bnu, N, seed, false, 0, 0);
            Metrics fix = run_bpf(sc, mc[mi].rho, mc[mi].sigma_z, mc[mi].mu,
                                  bnu, N, seed, true, best_lam, best_sJ);
            Metrics adp = run_bpf(sc, mc[mi].rho, mc[mi].sigma_z, mc[mi].mu,
                                  bnu, N, seed, true, -1.0f, best_sJ,
                                  0.01f, 0.03f, 0.08f);

            const char* best = "???";
            if (!cb.had_nan && !fix.had_nan && !adp.had_nan) {
                if (cb.rmse <= fix.rmse && cb.rmse <= adp.rmse) best = "Cb";
                else if (fix.rmse <= adp.rmse) best = "Fix";
                else best = "Adp";

                if (adp.rmse < cb.rmse)  ad_wins_vs_cb++;
                if (adp.rmse < fix.rmse) ad_wins_vs_fix++;
                if (fix.rmse < adp.rmse) fix_wins_vs_ad++;
            }

            printf("  %-12s | %8.4f %8.4f | %8.4f %8.4f | %8.4f %8.4f | %s\n",
                   mc[mi].label,
                   cb.had_nan  ? 0.0 : cb.rmse,  cb.had_nan  ? 0.0 : cb.spike_rmse,
                   fix.had_nan ? 0.0 : fix.rmse, fix.had_nan ? 0.0 : fix.spike_rmse,
                   adp.had_nan ? 0.0 : adp.rmse, adp.had_nan ? 0.0 : adp.spike_rmse,
                   best);

            if (!adp.had_nan) { ad_rmse_sum += adp.rmse; ad_spike_sum += adp.spike_rmse; }
            total3++;
        }
    }

    printf("\n");
    printf("──────────────────────────────────────────────────────────────────────────────────\n");
    printf("  PHASE 3 SUMMARY (%d runs)\n", total3);
    printf("  %-22s %14s %14s %14s\n", "", "Combined", "Fixed Jump", "Adaptive Jump");
    printf("  %-22s %14.4f %14.4f %14.4f\n", "Avg RMSE",
           cb_rmse_sum/total, jd_rmse_sum/total, ad_rmse_sum/total3);
    printf("  %-22s %14.4f %14.4f %14.4f\n", "Avg Spike RMSE",
           cb_spike_sum/total, jd_spike_sum/total, ad_spike_sum/total3);
    printf("  Adaptive wins vs Combined: %d/%d\n", ad_wins_vs_cb, total3);
    printf("  Adaptive wins vs Fixed:    %d/%d\n", ad_wins_vs_fix, total3);
    printf("  Fixed wins vs Adaptive:    %d/%d\n", fix_wins_vs_ad, total3);
    if (ad_rmse_sum > 0 && jd_rmse_sum > 0) {
        printf("  Adaptive vs Fixed RMSE:  %+.1f%%\n",
               100.0 * (ad_rmse_sum / jd_rmse_sum - 1.0));
        printf("  Adaptive vs Combined RMSE: %+.1f%%\n",
               100.0 * (ad_rmse_sum / cb_rmse_sum - 1.0));
    }
    printf("══════════════════════════════════════════════════════════════════════════════════\n");

    // =========================================================================
    // PHASE 4: Lambda triple sweep (calm × alert × panic)
    // =========================================================================

    printf("\n");
    printf("══════════════════════════════════════════════════════════════════════════════════\n");
    printf("  PHASE 4: Lambda triple sweep  sigma_J=%.2f\n", best_sJ);
    printf("══════════════════════════════════════════════════════════════════════════════════\n\n");

    float lam_c[] = {0.005f, 0.01f, 0.02f};
    float lam_a[] = {0.02f,  0.03f, 0.05f};
    float lam_p[] = {0.05f,  0.08f, 0.12f, 0.15f};
    int nc = 3, na = 3, np = 4;

    printf("  %6s %6s %6s | %8s %8s | %8s %8s | %8s\n",
           "Calm", "Alert", "Panic",
           "SpkRMSE", "SpkSpike", "RegRMSE", "RegSpike", "AvgSpike");
    printf("  ────── ────── ────── | ──────── ──────── | ──────── ──────── | ────────\n");

    float  best_tc = 0.01f, best_ta = 0.03f, best_tp = 0.08f;
    double best_triple_spike = 1e9;

    for (int ic = 0; ic < nc; ic++) {
        for (int ia = 0; ia < na; ia++) {
            for (int ip = 0; ip < np; ip++) {
                float c = lam_c[ic], a = lam_a[ia], p = lam_p[ip];

                if (c >= a || a >= p) continue;

                Metrics js = run_bpf(spike_sc, 0.98f, 0.15f, -4.5f, bnu, N, seed,
                                     true, -1.0f, best_sJ, c, a, p);
                Metrics jr = run_bpf(regime_sc, 0.98f, 0.15f, -4.5f, bnu, N, seed,
                                     true, -1.0f, best_sJ, c, a, p);

                double avg_spike = (js.spike_rmse + jr.spike_rmse) / 2.0;

                printf("  %6.3f %6.3f %6.3f | %8.4f %8.4f | %8.4f %8.4f | %8.4f",
                       c, a, p,
                       js.rmse, js.spike_rmse,
                       jr.rmse, jr.spike_rmse,
                       avg_spike);

                if (avg_spike < best_triple_spike) {
                    best_triple_spike = avg_spike;
                    best_tc = c; best_ta = a; best_tp = p;
                    printf(" *");
                }
                printf("\n");
            }
        }
    }

    printf("\n  Best triple: calm=%.3f  alert=%.3f  panic=%.3f\n",
           best_tc, best_ta, best_tp);
    printf("  Avg spike RMSE: %.4f\n", best_triple_spike);
    printf("══════════════════════════════════════════════════════════════════════════════════\n\n");

    return 0;
}
