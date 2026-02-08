// =============================================================================
// Correlated Pseudo-Marginal MH (CPMMH) — Convergence Test
//
// Online windowed CPMMH for learning σ_z (and optionally μ, ρ).
//
// Key idea (Deligiannidis, Doucet & Pitt 2018):
//   - Main filter runs with current θ, accumulates log p̂(y₁:W|θ)
//   - Shadow filter runs with proposed θ', using CORRELATED random numbers
//   - Correlation: u'_i = ρ_c * u_i + sqrt(1-ρ_c²) * v_i
//   - Every W ticks: MH accept/reject on likelihood ratio
//   - Variance of log-ratio scales as (1-ρ_c) instead of 2×Var
//
// Tests:
//   1. Learn σ_z only, moderate misspec
//   2. Learn σ_z only, severe misspec
//   3. Learn σ_z + μ jointly
//   4. Learn σ_z + ρ jointly
//   5. Learn all three
//   6. Regime change (σ_z jumps)
//   7. Window size sweep
//   8. Correlation parameter sweep
//
// Build: nvcc -O3 test_cpmmh.cu -o test_cpmmh
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

// =============================================================================
// PRNG — PCG32 (same as PTX kernel, for reproducibility)
// =============================================================================

struct PCG32 {
    unsigned long long state, inc;
};

static inline unsigned int pcg32_next(PCG32& rng) {
    unsigned long long old = rng.state;
    rng.state = old * 6364136223846793005ULL + rng.inc;
    unsigned int xsh = (unsigned int)(((old >> 18u) ^ old) >> 27u);
    unsigned int rot = (unsigned int)(old >> 59u);
    return (xsh >> rot) | (xsh << ((-rot) & 31));
}

static inline float pcg32_float(PCG32& rng) {
    return (float)(pcg32_next(rng) >> 9) * (1.0f / 8388608.0f);
}

// Box-Muller
static inline float pcg32_randn(PCG32& rng) {
    float u1 = pcg32_float(rng) + 1e-10f;
    float u2 = pcg32_float(rng);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979f * u2);
}

static PCG32 pcg32_seed(int seed) {
    PCG32 r;
    r.state = 0;
    r.inc = ((unsigned long long)seed << 1u) | 1u;
    pcg32_next(r);
    r.state += 0x853c49e6748fea9bULL;
    pcg32_next(r);
    return r;
}

// Global RNG for DGP and proposal
static PCG32 g_rng;

static float randn() { return pcg32_randn(g_rng); }
static float randf() { return pcg32_float(g_rng); }

// =============================================================================
// DGP
// =============================================================================

struct DGPState {
    float h, rho, sigma_z, mu;
};

static float dgp_step(DGPState& d) {
    d.h = d.mu + d.rho * (d.h - d.mu) + d.sigma_z * randn();
    return expf(d.h * 0.5f) * randn();
}

// =============================================================================
// Simple bootstrap particle filter (for PMMH likelihood evaluation)
// =============================================================================

struct BPF {
    int N;
    std::vector<float> h;
    std::vector<float> h2;     // scratch for resampling
    std::vector<float> w;
    std::vector<float> log_w;
    std::vector<float> cdf;
    std::vector<float> normals; // stored normal draws (for correlation)
    float h_est;
    double cum_log_lik;        // accumulated log-likelihood over window
};

static BPF bpf_create(int N) {
    BPF f;
    f.N = N;
    f.h.resize(N);
    f.h2.resize(N);
    f.w.resize(N);
    f.log_w.resize(N);
    f.cdf.resize(N);
    f.normals.resize(N);
    f.h_est = 0;
    f.cum_log_lik = 0;
    return f;
}

static void bpf_init(BPF& f, float mu, float sigma_stat, PCG32& rng) {
    for (int i = 0; i < f.N; i++) {
        f.h[i] = mu + sigma_stat * pcg32_randn(rng);
        f.w[i] = 1.0f / f.N;
    }
    f.cum_log_lik = 0;
}

// Run one BPF step, storing normal draws used. Returns per-step log-likelihood.
static float bpf_step(BPF& f, float y_t, float rho, float sigma_z, float mu,
                       PCG32& rng, bool store_normals) {
    int N = f.N;

    // Propagate
    for (int i = 0; i < N; i++) {
        float z = pcg32_randn(rng);
        if (store_normals) f.normals[i] = z;
        f.h[i] = mu + rho * (f.h[i] - mu) + sigma_z * z;
    }

    // Weight
    float max_lw = -1e30f;
    for (int i = 0; i < N; i++) {
        float eta = y_t * expf(-f.h[i] * 0.5f);
        f.log_w[i] = -0.9189385f - 0.5f * f.h[i] - 0.5f * eta * eta;
        if (f.log_w[i] > max_lw) max_lw = f.log_w[i];
    }

    float sum_w = 0;
    for (int i = 0; i < N; i++) {
        f.w[i] = expf(f.log_w[i] - max_lw);
        sum_w += f.w[i];
    }

    // Log-likelihood for this step: log(sum_w/N) + max_lw
    float step_ll = logf(sum_w / N) + max_lw;
    f.cum_log_lik += step_ll;

    // Normalize
    for (int i = 0; i < N; i++) f.w[i] /= sum_w;

    // Weighted mean
    float hm = 0;
    for (int i = 0; i < N; i++) hm += f.w[i] * f.h[i];
    f.h_est = hm;

    // Systematic resampling
    f.cdf[0] = f.w[0];
    for (int i = 1; i < N; i++) f.cdf[i] = f.cdf[i-1] + f.w[i];

    float u = pcg32_float(rng) / N;
    int j = 0;
    for (int i = 0; i < N; i++) {
        float target = u + (float)i / N;
        while (j < N - 1 && f.cdf[j] < target) j++;
        f.h2[i] = f.h[j];
    }
    std::swap(f.h, f.h2);
    for (int i = 0; i < N; i++) f.w[i] = 1.0f / N;

    return step_ll;
}

// Same as bpf_step but uses correlated normals from reference filter
static float bpf_step_correlated(BPF& f, float y_t, float rho, float sigma_z,
                                  float mu, PCG32& rng,
                                  const std::vector<float>& ref_normals,
                                  float rho_corr) {
    int N = f.N;
    float rc = rho_corr;
    float sc = sqrtf(1.0f - rc * rc);

    // Propagate with correlated normals
    for (int i = 0; i < N; i++) {
        float v = pcg32_randn(rng);
        float z = rc * ref_normals[i] + sc * v;
        f.normals[i] = z;
        f.h[i] = mu + rho * (f.h[i] - mu) + sigma_z * z;
    }

    // Weight (same as standard)
    float max_lw = -1e30f;
    for (int i = 0; i < N; i++) {
        float eta = y_t * expf(-f.h[i] * 0.5f);
        f.log_w[i] = -0.9189385f - 0.5f * f.h[i] - 0.5f * eta * eta;
        if (f.log_w[i] > max_lw) max_lw = f.log_w[i];
    }

    float sum_w = 0;
    for (int i = 0; i < N; i++) {
        f.w[i] = expf(f.log_w[i] - max_lw);
        sum_w += f.w[i];
    }

    float step_ll = logf(sum_w / N) + max_lw;
    f.cum_log_lik += step_ll;

    for (int i = 0; i < N; i++) f.w[i] /= sum_w;

    float hm = 0;
    for (int i = 0; i < N; i++) hm += f.w[i] * f.h[i];
    f.h_est = hm;

    // Resampling (use own RNG — not correlated for resampling)
    f.cdf[0] = f.w[0];
    for (int i = 1; i < N; i++) f.cdf[i] = f.cdf[i-1] + f.w[i];

    float u = pcg32_float(rng) / N;
    int j = 0;
    for (int i = 0; i < N; i++) {
        float target = u + (float)i / N;
        while (j < N - 1 && f.cdf[j] < target) j++;
        f.h2[i] = f.h[j];
    }
    std::swap(f.h, f.h2);
    for (int i = 0; i < N; i++) f.w[i] = 1.0f / N;

    return step_ll;
}

// =============================================================================
// CPMMH parameter proposal
// =============================================================================

struct CPMMHConfig {
    int n_pf_particles;      // particles per filter
    int window;              // ticks between MH decisions
    float rho_corr;          // correlation between filters (0.95-0.999)
    bool learn_sigma_z;
    bool learn_rho;
    bool learn_mu;
    float proposal_std_sz;   // random walk proposal std for σ_z
    float proposal_std_rho;  // for ρ
    float proposal_std_mu;   // for μ
    float fixed_rho;
    float fixed_mu;
    float fixed_sigma_z;
};

struct CPMMHState {
    // Current accepted params
    float sigma_z, rho, mu;

    // Proposed params (for shadow filter)
    float prop_sigma_z, prop_rho, prop_mu;

    // Filters
    BPF main_filter;
    BPF shadow_filter;
    PCG32 main_rng;
    PCG32 shadow_rng;

    // MH tracking
    int window_tick;          // ticks since last MH decision
    int total_proposals;
    int total_accepts;
    double main_ll_window;    // log-lik accumulated in current window
    double shadow_ll_window;

    CPMMHConfig cfg;
};

static CPMMHState cpmmh_create(CPMMHConfig cfg, float init_sz, float init_rho,
                                float init_mu, int seed) {
    CPMMHState s;
    s.cfg = cfg;
    s.sigma_z = cfg.learn_sigma_z ? init_sz : cfg.fixed_sigma_z;
    s.rho     = cfg.learn_rho ? init_rho : cfg.fixed_rho;
    s.mu      = cfg.learn_mu ? init_mu : cfg.fixed_mu;

    s.main_filter   = bpf_create(cfg.n_pf_particles);
    s.shadow_filter = bpf_create(cfg.n_pf_particles);
    s.main_rng   = pcg32_seed(seed);
    s.shadow_rng = pcg32_seed(seed + 7919);

    float sigma_stat = s.sigma_z / sqrtf(1.0f - s.rho * s.rho + 1e-8f);
    bpf_init(s.main_filter, s.mu, sigma_stat, s.main_rng);

    s.window_tick = 0;
    s.total_proposals = 0;
    s.total_accepts = 0;
    s.main_ll_window = 0;
    s.shadow_ll_window = 0;

    // Generate first proposal
    s.prop_sigma_z = s.sigma_z;
    s.prop_rho = s.rho;
    s.prop_mu = s.mu;

    return s;
}

static void cpmmh_propose(CPMMHState& s) {
    CPMMHConfig& c = s.cfg;

    s.prop_sigma_z = s.sigma_z;
    s.prop_rho = s.rho;
    s.prop_mu = s.mu;

    if (c.learn_sigma_z) {
        s.prop_sigma_z = s.sigma_z + c.proposal_std_sz * randn();
        s.prop_sigma_z = fmaxf(0.005f, fminf(1.0f, s.prop_sigma_z));
    }
    if (c.learn_rho) {
        s.prop_rho = s.rho + c.proposal_std_rho * randn();
        s.prop_rho = fmaxf(0.5f, fminf(0.999f, s.prop_rho));
    }
    if (c.learn_mu) {
        s.prop_mu = s.mu + c.proposal_std_mu * randn();
        s.prop_mu = fmaxf(-10.0f, fminf(0.0f, s.prop_mu));
    }

    // Reset shadow filter to match main filter's particle cloud
    s.shadow_filter.h = s.main_filter.h;
    for (int i = 0; i < s.cfg.n_pf_particles; i++)
        s.shadow_filter.w[i] = 1.0f / s.cfg.n_pf_particles;

    s.main_ll_window = 0;
    s.shadow_ll_window = 0;
    s.window_tick = 0;
}

// Returns h_est from main filter
static float cpmmh_step(CPMMHState& s, float y_t) {
    // If first tick or just accepted, generate new proposal
    if (s.window_tick == 0 && s.total_proposals == 0) {
        cpmmh_propose(s);
        s.total_proposals++;
    }

    // Run main filter (stores normals)
    float main_ll = bpf_step(s.main_filter, y_t,
                              s.rho, s.sigma_z, s.mu,
                              s.main_rng, true);
    s.main_ll_window += main_ll;

    // Run shadow filter with correlated normals
    float shadow_ll = bpf_step_correlated(s.shadow_filter, y_t,
                                           s.prop_rho, s.prop_sigma_z, s.prop_mu,
                                           s.shadow_rng,
                                           s.main_filter.normals,
                                           s.cfg.rho_corr);
    s.shadow_ll_window += shadow_ll;

    s.window_tick++;

    // MH decision at end of window
    if (s.window_tick >= s.cfg.window) {
        float log_alpha = (float)(s.shadow_ll_window - s.main_ll_window);
        // Flat prior → no prior ratio term (within bounds)
        float u = randf();
        bool accept = (logf(u + 1e-30f) < log_alpha);

        if (accept) {
            s.sigma_z = s.prop_sigma_z;
            s.rho = s.prop_rho;
            s.mu = s.prop_mu;
            // Shadow becomes main
            std::swap(s.main_filter.h, s.shadow_filter.h);
            std::swap(s.main_rng, s.shadow_rng);
            s.total_accepts++;
        }

        s.total_proposals++;
        cpmmh_propose(s);
    }

    return s.main_filter.h_est;
}

// =============================================================================
// Test runner
// =============================================================================

struct TestResult {
    float final_sz, final_rho, final_mu;
    float h_rmse;
    float accept_rate;
};

static TestResult run_test(const char* name,
                            int T, float true_rho, float true_sz, float true_mu,
                            float init_rho, float init_sz, float init_mu,
                            bool learn_sz, bool learn_rho, bool learn_mu,
                            int n_pf, int window, float rho_corr,
                            float prop_std_sz, float prop_std_rho, float prop_std_mu,
                            int trace_interval) {
    printf("\n┌── %s ", name);
    for (int p = (int)strlen(name); p < 68; p++) printf("─");
    printf("┐\n");
    printf("  True: σ_z=%.3f  ρ=%.3f  μ=%.1f\n", true_sz, true_rho, true_mu);
    printf("  Init: σ_z=%.3f  ρ=%.3f  μ=%.1f\n", init_sz, init_rho, init_mu);
    printf("  N_pf=%d  W=%d  ρ_c=%.3f  prop_std(σ_z)=%.4f\n",
           n_pf, window, rho_corr, prop_std_sz);

    CPMMHConfig cfg;
    cfg.n_pf_particles = n_pf;
    cfg.window = window;
    cfg.rho_corr = rho_corr;
    cfg.learn_sigma_z = learn_sz;
    cfg.learn_rho = learn_rho;
    cfg.learn_mu = learn_mu;
    cfg.proposal_std_sz  = prop_std_sz;
    cfg.proposal_std_rho = prop_std_rho;
    cfg.proposal_std_mu  = prop_std_mu;
    cfg.fixed_rho = init_rho;
    cfg.fixed_mu  = init_mu;
    cfg.fixed_sigma_z = init_sz;

    CPMMHState s = cpmmh_create(cfg, init_sz, init_rho, init_mu, 42);

    DGPState dgp;
    dgp.rho = true_rho; dgp.sigma_z = true_sz; dgp.mu = true_mu; dgp.h = true_mu;

    // Generate data
    std::vector<float> returns(T), true_h(T);
    for (int t = 0; t < T; t++) {
        returns[t] = dgp_step(dgp);
        true_h[t] = dgp.h;
    }

    printf("  %5s │ %8s", "tick", "h_est");
    if (learn_sz)  printf("  %8s", "σ_z");
    if (learn_rho) printf("  %8s", "ρ");
    if (learn_mu)  printf("  %8s", "μ");
    printf("  │ %8s  %6s  %6s\n", "true_h", "acc%", "prop#");
    printf("  ───── │ ────────");
    if (learn_sz)  printf("  ────────");
    if (learn_rho) printf("  ────────");
    if (learn_mu)  printf("  ────────");
    printf("  │ ────────  ──────  ──────\n");

    double sse = 0;
    int skip = 100, count = 0;

    for (int t = 0; t < T; t++) {
        float h_est = cpmmh_step(s, returns[t]);

        if (t >= skip) {
            double err = h_est - true_h[t];
            sse += err * err;
            count++;
        }

        if (t % trace_interval == 0 || t == T - 1) {
            float ar = s.total_proposals > 0
                ? 100.0f * s.total_accepts / s.total_proposals : 0;
            printf("  %5d │ %+8.4f", t, h_est);
            if (learn_sz)  printf("  %8.5f", s.sigma_z);
            if (learn_rho) printf("  %8.5f", s.rho);
            if (learn_mu)  printf("  %+8.4f", s.mu);
            printf("  │ %+8.4f  %5.1f%%  %5d\n", true_h[t], ar, s.total_proposals);
        }
    }

    TestResult r;
    r.final_sz  = s.sigma_z;
    r.final_rho = s.rho;
    r.final_mu  = s.mu;
    r.h_rmse = count > 0 ? (float)sqrt(sse / count) : 0;
    r.accept_rate = s.total_proposals > 0
        ? (float)s.total_accepts / s.total_proposals : 0;

    printf("  Final: σ_z=%.5f (true=%.3f)  accept=%.1f%%  RMSE=%.4f\n",
           r.final_sz, true_sz, 100.0f * r.accept_rate, r.h_rmse);
    printf("└");
    for (int p = 0; p < 75; p++) printf("─");
    printf("┘\n");

    return r;
}

// Regime change test (manual loop)
static TestResult run_regime_test(int T, int change_t, float true_rho,
                                   float true_sz1, float true_sz2, float true_mu,
                                   int n_pf, int window, float rho_corr,
                                   float prop_std_sz, int trace_interval) {
    printf("\n┌── REGIME CHANGE: σ_z %.2f→%.2f at t=%d ", true_sz1, true_sz2, change_t);
    for (int p = 0; p < 30; p++) printf("─");
    printf("┐\n");
    printf("  N_pf=%d  W=%d  ρ_c=%.3f\n", n_pf, window, rho_corr);

    CPMMHConfig cfg;
    cfg.n_pf_particles = n_pf;
    cfg.window = window;
    cfg.rho_corr = rho_corr;
    cfg.learn_sigma_z = true;
    cfg.learn_rho = false;
    cfg.learn_mu = false;
    cfg.proposal_std_sz = prop_std_sz;
    cfg.proposal_std_rho = 0;
    cfg.proposal_std_mu = 0;
    cfg.fixed_rho = true_rho;
    cfg.fixed_mu = true_mu;
    cfg.fixed_sigma_z = 0;

    CPMMHState s = cpmmh_create(cfg, true_sz1, true_rho, true_mu, 42);

    DGPState dgp;
    dgp.rho = true_rho; dgp.sigma_z = true_sz1; dgp.mu = true_mu; dgp.h = true_mu;

    printf("  %5s │ %8s  %8s  │ %8s  │ true σ_z  acc%%\n",
           "tick", "h_est", "σ_z", "true_h");
    printf("  ───── │ ────────  ────────  │ ────────  │ ────────  ────\n");

    double sse = 0;
    int skip = 100, count = 0;

    for (int t = 0; t < T; t++) {
        if (t == change_t) {
            dgp.sigma_z = true_sz2;
            printf("  ───── │ ──── σ_z JUMPS TO %.2f ─────────────────────────────\n", true_sz2);
        }

        float y = dgp_step(dgp);
        float h_est = cpmmh_step(s, y);

        if (t >= skip) {
            double err = h_est - dgp.h;
            sse += err * err;
            count++;
        }

        if (t % trace_interval == 0 || t == T - 1 ||
            (t >= change_t - 1 && t <= change_t + 1)) {
            float ar = s.total_proposals > 0
                ? 100.0f * s.total_accepts / s.total_proposals : 0;
            float true_now = (t < change_t) ? true_sz1 : true_sz2;
            printf("  %5d │ %+8.4f  %8.5f  │ %+8.4f  │ %.2f      %5.1f%%\n",
                   t, h_est, s.sigma_z, dgp.h, true_now, ar);
        }
    }

    TestResult r;
    r.final_sz = s.sigma_z;
    r.h_rmse = count > 0 ? (float)sqrt(sse / count) : 0;
    r.accept_rate = s.total_proposals > 0
        ? (float)s.total_accepts / s.total_proposals : 0;

    printf("  Final: σ_z=%.5f (true=%.2f)  accept=%.1f%%  RMSE=%.4f\n",
           r.final_sz, true_sz2, 100.0f * r.accept_rate, r.h_rmse);
    printf("└");
    for (int p = 0; p < 75; p++) printf("─");
    printf("┘\n");
    return r;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    float true_rho = 0.98f, true_sz = 0.15f, true_mu = -4.5f;
    int N_pf = 2000;           // particles per filter (2x = 4000 total)
    int W = 50;                // window size
    float rho_c = 0.99f;      // correlation
    float prop_sz = 0.015f;   // proposal std for σ_z

    printf("═══════════════════════════════════════════════════════════════════════════\n");
    printf("  CPMMH Convergence Test\n");
    printf("  True DGP: ρ=%.2f  σ_z=%.2f  μ=%.1f\n", true_rho, true_sz, true_mu);
    printf("  PF particles: %d (×2 = %d total)  Window: %d  ρ_corr: %.3f\n",
           N_pf, 2 * N_pf, W, rho_c);
    printf("═══════════════════════════════════════════════════════════════════════════\n");

    // ===== TEST 1: Learn σ_z only, moderate misspec =====
    g_rng = pcg32_seed(42);
    TestResult r1 = run_test("TEST 1: σ_z only (start=0.05, true=0.15)",
        3000, true_rho, true_sz, true_mu,
        true_rho, 0.05f, true_mu,
        true, false, false,
        N_pf, W, rho_c, prop_sz, 0, 0,
        200);

    // ===== TEST 2: Learn σ_z, severe misspec =====
    g_rng = pcg32_seed(42);
    TestResult r2 = run_test("TEST 2: σ_z only (start=0.50, true=0.15)",
        3000, true_rho, true_sz, true_mu,
        true_rho, 0.50f, true_mu,
        true, false, false,
        N_pf, W, rho_c, prop_sz, 0, 0,
        200);

    // ===== TEST 3: Learn σ_z + μ =====
    g_rng = pcg32_seed(42);
    TestResult r3 = run_test("TEST 3: σ_z + μ jointly",
        5000, true_rho, true_sz, true_mu,
        true_rho, 0.05f, -3.0f,
        true, false, true,
        N_pf, W, rho_c, prop_sz, 0, 0.15f,
        300);

    // ===== TEST 4: Learn σ_z + ρ =====
    g_rng = pcg32_seed(42);
    TestResult r4 = run_test("TEST 4: σ_z + ρ jointly",
        5000, true_rho, true_sz, true_mu,
        0.85f, 0.05f, true_mu,
        true, true, false,
        N_pf, W, rho_c, prop_sz, 0.008f, 0,
        300);

    // ===== TEST 5: Learn all three =====
    g_rng = pcg32_seed(42);
    TestResult r5 = run_test("TEST 5: σ_z + ρ + μ (all three)",
        8000, true_rho, true_sz, true_mu,
        0.85f, 0.05f, -3.0f,
        true, true, true,
        N_pf, W, rho_c, prop_sz, 0.008f, 0.15f,
        500);

    // ===== TEST 6: Regime change =====
    g_rng = pcg32_seed(42);
    TestResult r6 = run_regime_test(5000, 2000, true_rho,
        true_sz, 0.40f, true_mu,
        N_pf, W, rho_c, prop_sz, 200);

    // ===== TEST 7: Window size sweep =====
    printf("\n┌── TEST 7: Window size sweep ──────────────────────────────────────────┐\n");
    printf("  %5s │ %8s │ %6s │ %8s │ proposals/3000 ticks\n",
           "W", "σ_z_est", "acc%", "RMSE");
    printf("  ───── │ ──────── │ ────── │ ──────── │ ────────────────────\n");
    int windows[] = {10, 25, 50, 100, 200, 500};
    for (int wi = 0; wi < 6; wi++) {
        g_rng = pcg32_seed(42);

        CPMMHConfig cfg;
        cfg.n_pf_particles = N_pf;
        cfg.window = windows[wi];
        cfg.rho_corr = rho_c;
        cfg.learn_sigma_z = true; cfg.learn_rho = false; cfg.learn_mu = false;
        cfg.proposal_std_sz = prop_sz;
        cfg.proposal_std_rho = 0; cfg.proposal_std_mu = 0;
        cfg.fixed_rho = true_rho; cfg.fixed_mu = true_mu; cfg.fixed_sigma_z = 0;

        CPMMHState s = cpmmh_create(cfg, 0.05f, true_rho, true_mu, 42);

        DGPState dgp;
        dgp.rho = true_rho; dgp.sigma_z = true_sz; dgp.mu = true_mu; dgp.h = true_mu;

        std::vector<float> returns(3000), true_h(3000);
        for (int t = 0; t < 3000; t++) { returns[t] = dgp_step(dgp); true_h[t] = dgp.h; }

        double sse = 0; int cnt = 0;
        for (int t = 0; t < 3000; t++) {
            float he = cpmmh_step(s, returns[t]);
            if (t >= 100) { double e = he - true_h[t]; sse += e*e; cnt++; }
        }
        float rmse = sqrtf((float)(sse / cnt));
        float ar = s.total_proposals > 0 ? 100.0f * s.total_accepts / s.total_proposals : 0;
        printf("  %5d │ %8.5f │ %5.1f%% │ %8.4f │ %d\n",
               windows[wi], s.sigma_z, ar, rmse, s.total_proposals);
    }
    printf("└");
    for (int p = 0; p < 75; p++) printf("─");
    printf("┘\n");

    // ===== TEST 8: Correlation sweep =====
    printf("\n┌── TEST 8: Correlation ρ_c sweep ──────────────────────────────────────┐\n");
    printf("  %6s │ %8s │ %6s │ %8s\n", "ρ_c", "σ_z_est", "acc%", "RMSE");
    printf("  ────── │ ──────── │ ────── │ ────────\n");
    float rho_cs[] = {0.0f, 0.5f, 0.8f, 0.9f, 0.95f, 0.99f, 0.999f};
    for (int ri = 0; ri < 7; ri++) {
        g_rng = pcg32_seed(42);

        CPMMHConfig cfg;
        cfg.n_pf_particles = N_pf;
        cfg.window = W;
        cfg.rho_corr = rho_cs[ri];
        cfg.learn_sigma_z = true; cfg.learn_rho = false; cfg.learn_mu = false;
        cfg.proposal_std_sz = prop_sz;
        cfg.proposal_std_rho = 0; cfg.proposal_std_mu = 0;
        cfg.fixed_rho = true_rho; cfg.fixed_mu = true_mu; cfg.fixed_sigma_z = 0;

        CPMMHState s = cpmmh_create(cfg, 0.05f, true_rho, true_mu, 42);

        DGPState dgp;
        dgp.rho = true_rho; dgp.sigma_z = true_sz; dgp.mu = true_mu; dgp.h = true_mu;

        std::vector<float> returns(3000), true_h(3000);
        for (int t = 0; t < 3000; t++) { returns[t] = dgp_step(dgp); true_h[t] = dgp.h; }

        double sse = 0; int cnt = 0;
        for (int t = 0; t < 3000; t++) {
            float he = cpmmh_step(s, returns[t]);
            if (t >= 100) { double e = he - true_h[t]; sse += e*e; cnt++; }
        }
        float rmse = sqrtf((float)(sse / cnt));
        float ar = s.total_proposals > 0 ? 100.0f * s.total_accepts / s.total_proposals : 0;
        printf("  %6.3f │ %8.5f │ %5.1f%% │ %8.4f\n", rho_cs[ri], s.sigma_z, ar, rmse);
    }
    printf("└");
    for (int p = 0; p < 75; p++) printf("─");
    printf("┘\n");

    // ===== ORACLE COMPARISON =====
    printf("\n┌── ORACLE COMPARISON ──────────────────────────────────────────────────┐\n");
    g_rng = pcg32_seed(42);
    TestResult oracle = run_test("Oracle (fixed true params)",
        3000, true_rho, true_sz, true_mu,
        true_rho, true_sz, true_mu,
        false, false, false,
        N_pf, W, rho_c, 0, 0, 0,
        99999);
    printf("  Oracle RMSE = %.4f  (N_pf=%d, same budget)\n", oracle.h_rmse, N_pf);
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
