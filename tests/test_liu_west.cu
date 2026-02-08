// =============================================================================
// Liu-West Filter Convergence Test (CPU)
//
// Each particle carries {h_i, σ_z_i}.  After resampling, σ_z is jittered:
//   σ̃_z = a * σ_z_i  +  (1-a) * σ̄_z  +  sqrt(1-a²) * σ_hat * ε
// where σ̄_z = weighted mean, σ_hat = weighted std, ε ~ N(0,1).
//
// Tests:
//   1. Learn σ_z only (ρ, μ fixed correct)
//   2. Learn σ_z starting from very wrong value
//   3. Learn σ_z + μ jointly
//   4. Learn σ_z + ρ jointly
//   5. Learn all three {σ_z, ρ, μ}
//   6. Regime change: σ_z jumps mid-series
//
// Build: nvcc -O3 test_liu_west.cu -o test_liu_west
//        (no GPU needed — pure CPU, .cu just for nvcc compat)
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

// =============================================================================
// PRNG
// =============================================================================

static unsigned int g_seed = 42;

static inline float randf() {
    g_seed = g_seed * 1103515245 + 12345;
    return (float)((g_seed >> 16) & 0x7FFF) / 32768.0f;
}

static inline float randn() {
    float u1 = randf() + 1e-10f;
    float u2 = randf();
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

// =============================================================================
// DGP: standard stochastic volatility
// =============================================================================

struct DGPState {
    float h;
    float rho, sigma_z, mu;
};

static float dgp_step(DGPState& d) {
    float eps = randn();
    d.h = d.mu + d.rho * (d.h - d.mu) + d.sigma_z * eps;
    float eta = randn();
    return expf(d.h * 0.5f) * eta;
}

// =============================================================================
// Liu-West particle
// =============================================================================

struct LWParticle {
    float h;
    float sigma_z;   // learned
    float rho;       // learned (or fixed)
    float mu;        // learned (or fixed)
    float log_w;
    float w;
};

// =============================================================================
// Liu-West filter
// =============================================================================

struct LWConfig {
    float a;          // shrinkage (0.95–0.99)
    bool learn_sigma_z;
    bool learn_rho;
    bool learn_mu;
    float fixed_rho;
    float fixed_mu;
    float fixed_sigma_z;
};

struct LWState {
    std::vector<LWParticle> particles;
    std::vector<LWParticle> resampled;
    int N;
    LWConfig cfg;
};

static LWState lw_create(int N, LWConfig cfg,
                           float init_sigma_z, float init_rho, float init_mu) {
    LWState s;
    s.N = N;
    s.cfg = cfg;
    s.particles.resize(N);
    s.resampled.resize(N);

    // Stationary variance: sigma_stat = sigma_z / sqrt(1 - rho^2)
    float rho = cfg.learn_rho ? init_rho : cfg.fixed_rho;
    float sz  = cfg.learn_sigma_z ? init_sigma_z : cfg.fixed_sigma_z;
    float mu  = cfg.learn_mu ? init_mu : cfg.fixed_mu;
    float sigma_stat = sz / sqrtf(1.0f - rho * rho + 1e-8f);

    for (int i = 0; i < N; i++) {
        s.particles[i].h = mu + sigma_stat * randn();
        // Disperse initial params around starting point
        s.particles[i].sigma_z = cfg.learn_sigma_z
            ? fmaxf(0.01f, init_sigma_z + 0.02f * randn()) : cfg.fixed_sigma_z;
        s.particles[i].rho = cfg.learn_rho
            ? fminf(0.999f, fmaxf(0.5f, init_rho + 0.02f * randn())) : cfg.fixed_rho;
        s.particles[i].mu = cfg.learn_mu
            ? init_mu + 0.3f * randn() : cfg.fixed_mu;
        s.particles[i].w = 1.0f / N;
    }
    return s;
}

static void lw_step(LWState& s, float y_t) {
    int N = s.N;
    float a = s.cfg.a;
    float h2 = 1.0f - a * a;  // kernel variance fraction

    // --- 1. Propagate + weight ---
    float max_lw = -1e30f;
    for (int i = 0; i < N; i++) {
        LWParticle& p = s.particles[i];
        float eps = randn();
        p.h = p.mu + p.rho * (p.h - p.mu) + p.sigma_z * eps;

        // Gaussian obs log-likelihood
        float eta = y_t * expf(-p.h * 0.5f);
        p.log_w = -0.5f * logf(2.0f * 3.14159265f) - 0.5f * eta * eta - 0.5f * p.h;
        if (p.log_w > max_lw) max_lw = p.log_w;
    }

    // --- 2. Normalize weights ---
    float sum_w = 0.0f;
    for (int i = 0; i < N; i++) {
        s.particles[i].w = expf(s.particles[i].log_w - max_lw);
        sum_w += s.particles[i].w;
    }
    for (int i = 0; i < N; i++) {
        s.particles[i].w /= sum_w;
    }

    // --- 3. Compute weighted mean and std of learned params ---
    float mean_sz = 0, mean_rho = 0, mean_mu = 0;
    for (int i = 0; i < N; i++) {
        float w = s.particles[i].w;
        mean_sz  += w * s.particles[i].sigma_z;
        mean_rho += w * s.particles[i].rho;
        mean_mu  += w * s.particles[i].mu;
    }

    float var_sz = 0, var_rho = 0, var_mu = 0;
    for (int i = 0; i < N; i++) {
        float w = s.particles[i].w;
        float d;
        d = s.particles[i].sigma_z - mean_sz;  var_sz  += w * d * d;
        d = s.particles[i].rho - mean_rho;     var_rho += w * d * d;
        d = s.particles[i].mu - mean_mu;        var_mu  += w * d * d;
    }
    float std_sz  = sqrtf(fmaxf(var_sz,  1e-12f));
    float std_rho = sqrtf(fmaxf(var_rho, 1e-12f));
    float std_mu  = sqrtf(fmaxf(var_mu,  1e-12f));

    float h_scale = sqrtf(fmaxf(h2, 0.0f));

    // --- 4. Systematic resampling ---
    // Build CDF
    std::vector<float> cdf(N);
    cdf[0] = s.particles[0].w;
    for (int i = 1; i < N; i++) cdf[i] = cdf[i-1] + s.particles[i].w;

    float u = randf() / N;
    int j = 0;
    for (int i = 0; i < N; i++) {
        float target = u + (float)i / N;
        while (j < N - 1 && cdf[j] < target) j++;
        s.resampled[i] = s.particles[j];
    }

    // --- 5. Liu-West jitter on params ---
    for (int i = 0; i < N; i++) {
        LWParticle& p = s.resampled[i];
        if (s.cfg.learn_sigma_z) {
            float shrunk = a * p.sigma_z + (1.0f - a) * mean_sz;
            p.sigma_z = shrunk + h_scale * std_sz * randn();
            p.sigma_z = fmaxf(p.sigma_z, 0.005f);   // floor
            p.sigma_z = fminf(p.sigma_z, 1.0f);      // ceiling
        }
        if (s.cfg.learn_rho) {
            float shrunk = a * p.rho + (1.0f - a) * mean_rho;
            p.rho = shrunk + h_scale * std_rho * randn();
            p.rho = fmaxf(p.rho, 0.5f);
            p.rho = fminf(p.rho, 0.999f);
        }
        if (s.cfg.learn_mu) {
            float shrunk = a * p.mu + (1.0f - a) * mean_mu;
            p.mu = shrunk + h_scale * std_mu * randn();
            p.mu = fmaxf(p.mu, -10.0f);
            p.mu = fminf(p.mu, 0.0f);
        }
        p.w = 1.0f / N;
    }

    std::swap(s.particles, s.resampled);
}

// Weighted mean of h
static float lw_h_mean(const LWState& s) {
    float sum = 0;
    for (int i = 0; i < s.N; i++) sum += s.particles[i].w * s.particles[i].h;
    return sum;
}

// Weighted mean of parameter
static float lw_param_mean(const LWState& s, int which) {
    // 0=sigma_z, 1=rho, 2=mu
    float sum = 0;
    for (int i = 0; i < s.N; i++) {
        float w = s.particles[i].w;
        switch (which) {
            case 0: sum += w * s.particles[i].sigma_z; break;
            case 1: sum += w * s.particles[i].rho; break;
            case 2: sum += w * s.particles[i].mu; break;
        }
    }
    return sum;
}

static float lw_param_std(const LWState& s, int which) {
    float mean = lw_param_mean(s, which);
    float var = 0;
    for (int i = 0; i < s.N; i++) {
        float w = s.particles[i].w;
        float v;
        switch (which) {
            case 0: v = s.particles[i].sigma_z; break;
            case 1: v = s.particles[i].rho; break;
            case 2: v = s.particles[i].mu; break;
            default: v = 0;
        }
        var += w * (v - mean) * (v - mean);
    }
    return sqrtf(var);
}

// =============================================================================
// Run one convergence test
// =============================================================================

struct TestResult {
    float final_sz, final_rho, final_mu;
    float std_sz, std_rho, std_mu;
    float h_rmse;
    int ticks;
};

static TestResult run_test(int N, int T, float a,
                            float true_rho, float true_sz, float true_mu,
                            float init_rho, float init_sz, float init_mu,
                            bool learn_sz, bool learn_rho, bool learn_mu,
                            bool print_trace, int trace_interval) {
    LWConfig cfg;
    cfg.a = a;
    cfg.learn_sigma_z = learn_sz;
    cfg.learn_rho = learn_rho;
    cfg.learn_mu = learn_mu;
    cfg.fixed_rho = learn_rho ? 0 : init_rho;
    cfg.fixed_mu = learn_mu ? 0 : init_mu;
    cfg.fixed_sigma_z = learn_sz ? 0 : init_sz;

    LWState lw = lw_create(N, cfg, init_sz, init_rho, init_mu);

    DGPState dgp;
    dgp.rho = true_rho;
    dgp.sigma_z = true_sz;
    dgp.mu = true_mu;
    dgp.h = true_mu;

    // Generate full series
    std::vector<float> returns(T), true_h(T);
    for (int t = 0; t < T; t++) {
        returns[t] = dgp_step(dgp);
        true_h[t] = dgp.h;
    }

    if (print_trace) {
        printf("  %5s │ %8s", "tick", "h_est");
        if (learn_sz)  printf("  %8s ±%6s", "σ_z", "std");
        if (learn_rho) printf("  %8s ±%6s", "ρ", "std");
        if (learn_mu)  printf("  %8s ±%6s", "μ", "std");
        printf("  │ %8s", "true_h");
        printf("\n  ───── │ ────────");
        if (learn_sz)  printf("  ──────── ──────");
        if (learn_rho) printf("  ──────── ──────");
        if (learn_mu)  printf("  ──────── ──────");
        printf("  │ ────────\n");
    }

    double sse = 0;
    int skip = 50, count = 0;

    for (int t = 0; t < T; t++) {
        lw_step(lw, returns[t]);

        float h_est = lw_h_mean(lw);
        if (t >= skip) {
            double err = h_est - true_h[t];
            sse += err * err;
            count++;
        }

        if (print_trace && (t % trace_interval == 0 || t == T - 1)) {
            printf("  %5d │ %+8.4f", t, h_est);
            if (learn_sz)  printf("  %8.5f ±%6.4f", lw_param_mean(lw, 0), lw_param_std(lw, 0));
            if (learn_rho) printf("  %8.5f ±%6.4f", lw_param_mean(lw, 1), lw_param_std(lw, 1));
            if (learn_mu)  printf("  %+8.4f ±%6.4f", lw_param_mean(lw, 2), lw_param_std(lw, 2));
            printf("  │ %+8.4f", true_h[t]);
            printf("\n");
        }
    }

    TestResult r;
    r.final_sz  = lw_param_mean(lw, 0);
    r.final_rho = lw_param_mean(lw, 1);
    r.final_mu  = lw_param_mean(lw, 2);
    r.std_sz    = lw_param_std(lw, 0);
    r.std_rho   = lw_param_std(lw, 1);
    r.std_mu    = lw_param_std(lw, 2);
    r.h_rmse    = count > 0 ? (float)sqrt(sse / count) : 0;
    r.ticks     = T;
    return r;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    float true_rho = 0.98f, true_sz = 0.15f, true_mu = -4.5f;
    int N = 10000;
    float a = 0.97f;

    printf("═══════════════════════════════════════════════════════════════════════════\n");
    printf("  Liu-West Convergence Test\n");
    printf("  True DGP: ρ=%.2f  σ_z=%.2f  μ=%.1f  │  N=%d  a=%.2f\n",
           true_rho, true_sz, true_mu, N, a);
    printf("═══════════════════════════════════════════════════════════════════════════\n");

    // ===== TEST 1: Learn σ_z only (moderate misspec) =====
    printf("\n┌── TEST 1: Learn σ_z only (start=0.05, true=0.15) ─────────────────────┐\n");
    g_seed = 42;
    TestResult r1 = run_test(N, 3000, a,
        true_rho, true_sz, true_mu,
        true_rho, 0.05f, true_mu,
        true, false, false,
        true, 200);
    printf("  Final: σ_z=%.5f (true=%.2f)  RMSE=%.4f\n", r1.final_sz, true_sz, r1.h_rmse);

    // ===== TEST 2: Learn σ_z (severe misspec: start=0.50) =====
    printf("\n┌── TEST 2: Learn σ_z only (start=0.50, true=0.15) ─────────────────────┐\n");
    g_seed = 42;
    TestResult r2 = run_test(N, 3000, a,
        true_rho, true_sz, true_mu,
        true_rho, 0.50f, true_mu,
        true, false, false,
        true, 200);
    printf("  Final: σ_z=%.5f (true=%.2f)  RMSE=%.4f\n", r2.final_sz, true_sz, r2.h_rmse);

    // ===== TEST 3: Learn σ_z + μ jointly =====
    printf("\n┌── TEST 3: Learn σ_z + μ jointly (start σ_z=0.05, μ=-3.0) ────────────┐\n");
    g_seed = 42;
    TestResult r3 = run_test(N, 5000, a,
        true_rho, true_sz, true_mu,
        true_rho, 0.05f, -3.0f,
        true, false, true,
        true, 300);
    printf("  Final: σ_z=%.5f (true=%.2f)  μ=%.3f (true=%.1f)  RMSE=%.4f\n",
           r3.final_sz, true_sz, r3.final_mu, true_mu, r3.h_rmse);

    // ===== TEST 4: Learn σ_z + ρ jointly =====
    printf("\n┌── TEST 4: Learn σ_z + ρ jointly (start σ_z=0.05, ρ=0.85) ────────────┐\n");
    g_seed = 42;
    TestResult r4 = run_test(N, 5000, a,
        true_rho, true_sz, true_mu,
        0.85f, 0.05f, true_mu,
        true, true, false,
        true, 300);
    printf("  Final: σ_z=%.5f (true=%.2f)  ρ=%.5f (true=%.2f)  RMSE=%.4f\n",
           r4.final_sz, true_sz, r4.final_rho, true_rho, r4.h_rmse);

    // ===== TEST 5: Learn all three =====
    printf("\n┌── TEST 5: Learn all {σ_z, ρ, μ} (start 0.05, 0.85, -3.0) ───────────┐\n");
    g_seed = 42;
    TestResult r5 = run_test(N, 8000, a,
        true_rho, true_sz, true_mu,
        0.85f, 0.05f, -3.0f,
        true, true, true,
        true, 500);
    printf("  Final: σ_z=%.5f  ρ=%.5f  μ=%.3f  RMSE=%.4f\n",
           r5.final_sz, r5.final_rho, r5.final_mu, r5.h_rmse);
    printf("  True:  σ_z=%.5f  ρ=%.5f  μ=%.3f\n", true_sz, true_rho, true_mu);

    // ===== TEST 6: Regime change — σ_z jumps mid-series =====
    printf("\n┌── TEST 6: Regime change — σ_z jumps 0.15→0.40 at t=2000 ─────────────┐\n");
    g_seed = 42;
    {
        LWConfig cfg;
        cfg.a = a;
        cfg.learn_sigma_z = true;
        cfg.learn_rho = false;
        cfg.learn_mu = false;
        cfg.fixed_rho = true_rho;
        cfg.fixed_mu = true_mu;
        cfg.fixed_sigma_z = 0;

        LWState lw = lw_create(N, cfg, true_sz, true_rho, true_mu);

        DGPState dgp;
        dgp.rho = true_rho;
        dgp.sigma_z = true_sz;
        dgp.mu = true_mu;
        dgp.h = true_mu;

        int T = 5000;
        int change_t = 2000;
        float new_sz = 0.40f;

        printf("  %5s │ %8s  %8s ±%6s │ %8s  │ true σ_z\n",
               "tick", "h_est", "σ_z", "std", "true_h");
        printf("  ───── │ ────────  ──────── ────── │ ────────  │ ────────\n");

        for (int t = 0; t < T; t++) {
            if (t == change_t) {
                dgp.sigma_z = new_sz;
                printf("  ───── │ ──── σ_z JUMPS TO %.2f ────────────────────────────\n", new_sz);
            }
            float y = dgp_step(dgp);
            lw_step(lw, y);

            if (t % 200 == 0 || t == T - 1 || (t >= change_t - 1 && t <= change_t + 1)) {
                float true_now = (t < change_t) ? true_sz : new_sz;
                printf("  %5d │ %+8.4f  %8.5f ±%6.4f │ %+8.4f  │ %.2f\n",
                       t, lw_h_mean(lw), lw_param_mean(lw, 0), lw_param_std(lw, 0),
                       dgp.h, true_now);
            }
        }
        printf("  Final: σ_z=%.5f (true=%.2f)\n", lw_param_mean(lw, 0), new_sz);
    }

    // ===== ORACLE COMPARISON =====
    printf("\n┌── ORACLE COMPARISON ───────────────────────────────────────────────────┐\n");
    printf("  Running fixed-param BPF with oracle params as RMSE baseline...\n");
    g_seed = 42;
    TestResult oracle = run_test(N, 3000, a,
        true_rho, true_sz, true_mu,
        true_rho, true_sz, true_mu,
        false, false, false,
        false, 9999);
    printf("  Oracle RMSE = %.4f\n", oracle.h_rmse);
    printf("  ──────────────────────────────────────────────────────────────────────\n");
    printf("  Test 1 (σ_z from 0.05):  RMSE=%.4f  (vs oracle: %+.1f%%)\n",
           r1.h_rmse, 100.0*(r1.h_rmse/oracle.h_rmse - 1.0));
    printf("  Test 2 (σ_z from 0.50):  RMSE=%.4f  (vs oracle: %+.1f%%)\n",
           r2.h_rmse, 100.0*(r2.h_rmse/oracle.h_rmse - 1.0));
    printf("  Test 3 (σ_z + μ):        RMSE=%.4f  (vs oracle: %+.1f%%)\n",
           r3.h_rmse, 100.0*(r3.h_rmse/oracle.h_rmse - 1.0));
    printf("  Test 4 (σ_z + ρ):        RMSE=%.4f  (vs oracle: %+.1f%%)\n",
           r4.h_rmse, 100.0*(r4.h_rmse/oracle.h_rmse - 1.0));
    printf("  Test 5 (all three):      RMSE=%.4f  (vs oracle: %+.1f%%)\n",
           r5.h_rmse, 100.0*(r5.h_rmse/oracle.h_rmse - 1.0));

    printf("═══════════════════════════════════════════════════════════════════════════\n");
    return 0;
}
