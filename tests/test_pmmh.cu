// =============================================================================
// Particle Marginal Metropolis-Hastings (PMMH) — Convergence Test
//
// Simple online PMMH: two independent BPFs, MH accept/reject every W ticks.
//
// Design:
//   - Main filter runs with current θ
//   - Shadow filter runs with proposed θ' ~ N(θ, σ_prop²)
//   - Both accumulate log p̂(y_{t-W+1:t} | θ) over window
//   - Accept/reject based on likelihood ratio
//   - On accept: shadow params become current, shadow particles become main
//   - On reject: shadow resets to main particles with new proposal
//
// Tuning lessons from CPMMH attempt:
//   - Proposal std ~0.05 (not 0.015 — need acceptance ~30-50%)
//   - Window W=10-20 (not 50 — need many proposals)
//   - More particles = lower likelihood variance = cleaner MH decisions
//
// Build: nvcc -O3 test_pmmh.cu -o test_pmmh
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

// =============================================================================
// PRNG
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
// Minimal BPF (for PMMH inner filter)
// =============================================================================

struct BPF {
    int N;
    std::vector<float> h, h2, w, log_w, cdf;
    float h_est;
    double cum_ll;  // accumulated log-likelihood over window
    PCG32 rng;
};

static BPF bpf_create(int N, int seed) {
    BPF f;
    f.N = N;
    f.h.resize(N); f.h2.resize(N); f.w.resize(N);
    f.log_w.resize(N); f.cdf.resize(N);
    f.h_est = 0; f.cum_ll = 0;
    f.rng = pcg32_seed(seed);
    return f;
}

static void bpf_init(BPF& f, float mu, float sigma_stat) {
    for (int i = 0; i < f.N; i++) {
        f.h[i] = mu + sigma_stat * pcg32_randn(f.rng);
        f.w[i] = 1.0f / f.N;
    }
    f.cum_ll = 0;
}

static void bpf_copy_particles(BPF& dst, const BPF& src) {
    dst.h = src.h;
    for (int i = 0; i < dst.N; i++) dst.w[i] = 1.0f / dst.N;
    dst.cum_ll = 0;
}

static float bpf_step(BPF& f, float y_t, float rho, float sigma_z, float mu) {
    int N = f.N;

    // Propagate
    for (int i = 0; i < N; i++) {
        float z = pcg32_randn(f.rng);
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

    float step_ll = logf(sum_w / N) + max_lw;
    f.cum_ll += step_ll;

    // Normalize
    for (int i = 0; i < N; i++) f.w[i] /= sum_w;

    // Weighted mean
    float hm = 0;
    for (int i = 0; i < N; i++) hm += f.w[i] * f.h[i];
    f.h_est = hm;

    // Systematic resampling
    f.cdf[0] = f.w[0];
    for (int i = 1; i < N; i++) f.cdf[i] = f.cdf[i-1] + f.w[i];

    float u = pcg32_float(f.rng) / N;
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
// PMMH State
// =============================================================================

struct PMMHConfig {
    int n_pf;                // particles per filter
    int window;              // ticks between MH decisions
    bool learn_sigma_z, learn_rho, learn_mu;
    float prop_std_sz;       // random walk proposal σ for σ_z
    float prop_std_rho;
    float prop_std_mu;
    float fixed_rho, fixed_mu, fixed_sigma_z;
};

struct PMMHState {
    float sigma_z, rho, mu;                    // current accepted
    float prop_sigma_z, prop_rho, prop_mu;     // current proposal

    BPF main_filter;
    BPF shadow_filter;

    int window_tick;
    int n_proposals, n_accepts;

    PMMHConfig cfg;
};

static void pmmh_propose(PMMHState& s) {
    s.prop_sigma_z = s.sigma_z;
    s.prop_rho = s.rho;
    s.prop_mu = s.mu;

    if (s.cfg.learn_sigma_z) {
        s.prop_sigma_z += s.cfg.prop_std_sz * randn();
        s.prop_sigma_z = fmaxf(0.005f, fminf(1.0f, s.prop_sigma_z));
    }
    if (s.cfg.learn_rho) {
        s.prop_rho += s.cfg.prop_std_rho * randn();
        s.prop_rho = fmaxf(0.5f, fminf(0.999f, s.prop_rho));
    }
    if (s.cfg.learn_mu) {
        s.prop_mu += s.cfg.prop_std_mu * randn();
        s.prop_mu = fmaxf(-10.0f, fminf(0.0f, s.prop_mu));
    }

    // Shadow starts from main's particle cloud
    bpf_copy_particles(s.shadow_filter, s.main_filter);
    s.main_filter.cum_ll = 0;
    s.shadow_filter.cum_ll = 0;
    s.window_tick = 0;
}

static PMMHState pmmh_create(PMMHConfig cfg, float init_sz, float init_rho,
                              float init_mu, int seed) {
    PMMHState s;
    s.cfg = cfg;
    s.sigma_z = cfg.learn_sigma_z ? init_sz : cfg.fixed_sigma_z;
    s.rho     = cfg.learn_rho ? init_rho : cfg.fixed_rho;
    s.mu      = cfg.learn_mu ? init_mu : cfg.fixed_mu;

    s.main_filter   = bpf_create(cfg.n_pf, seed);
    s.shadow_filter = bpf_create(cfg.n_pf, seed + 9973);

    float sigma_stat = s.sigma_z / sqrtf(1.0f - s.rho * s.rho + 1e-8f);
    bpf_init(s.main_filter, s.mu, sigma_stat);

    s.window_tick = 0;
    s.n_proposals = 0;
    s.n_accepts = 0;

    s.prop_sigma_z = s.sigma_z;
    s.prop_rho = s.rho;
    s.prop_mu = s.mu;

    return s;
}

static float pmmh_step(PMMHState& s, float y_t) {
    if (s.window_tick == 0) {
        if (s.n_proposals == 0) {
            pmmh_propose(s);
        }
        s.n_proposals++;
    }

    // Run both filters
    bpf_step(s.main_filter, y_t, s.rho, s.sigma_z, s.mu);
    bpf_step(s.shadow_filter, y_t, s.prop_rho, s.prop_sigma_z, s.prop_mu);

    s.window_tick++;

    // MH decision at end of window
    if (s.window_tick >= s.cfg.window) {
        float log_alpha = (float)(s.shadow_filter.cum_ll - s.main_filter.cum_ll);
        float u = randf();
        bool accept = (logf(u + 1e-30f) < log_alpha);

        if (accept) {
            s.sigma_z = s.prop_sigma_z;
            s.rho = s.prop_rho;
            s.mu = s.prop_mu;
            // Shadow becomes main
            std::swap(s.main_filter.h, s.shadow_filter.h);
            std::swap(s.main_filter.rng, s.shadow_filter.rng);
            s.n_accepts++;
        }

        pmmh_propose(s);
    }

    return s.main_filter.h_est;
}

// =============================================================================
// Test runners
// =============================================================================

struct TestResult {
    float final_sz, final_rho, final_mu;
    float h_rmse;
    float accept_rate;
};

static TestResult run_test(const char* name, int T,
                            float true_rho, float true_sz, float true_mu,
                            float init_rho, float init_sz, float init_mu,
                            bool learn_sz, bool learn_rho, bool learn_mu,
                            int n_pf, int window,
                            float ps_sz, float ps_rho, float ps_mu,
                            int trace_every, bool print_trace = true) {
    if (print_trace) {
        printf("\n┌── %s ", name);
        for (int p = (int)strlen(name); p < 68; p++) printf("─");
        printf("┐\n");
        printf("  True: σ_z=%.3f  ρ=%.3f  μ=%.1f  │  Init: σ_z=%.3f  ρ=%.3f  μ=%.1f\n",
               true_sz, true_rho, true_mu, init_sz, init_rho, init_mu);
        printf("  N_pf=%d  W=%d  prop_std(σ_z)=%.3f\n", n_pf, window, ps_sz);
    }

    PMMHConfig cfg;
    cfg.n_pf = n_pf; cfg.window = window;
    cfg.learn_sigma_z = learn_sz; cfg.learn_rho = learn_rho; cfg.learn_mu = learn_mu;
    cfg.prop_std_sz = ps_sz; cfg.prop_std_rho = ps_rho; cfg.prop_std_mu = ps_mu;
    cfg.fixed_rho = init_rho; cfg.fixed_mu = init_mu; cfg.fixed_sigma_z = init_sz;

    PMMHState s = pmmh_create(cfg, init_sz, init_rho, init_mu, 42);

    DGPState dgp;
    dgp.rho = true_rho; dgp.sigma_z = true_sz; dgp.mu = true_mu; dgp.h = true_mu;

    std::vector<float> returns(T), true_h(T);
    for (int t = 0; t < T; t++) { returns[t] = dgp_step(dgp); true_h[t] = dgp.h; }

    if (print_trace) {
        printf("  %5s │ %8s", "tick", "h_est");
        if (learn_sz)  printf("  %8s", "σ_z");
        if (learn_rho) printf("  %8s", "ρ");
        if (learn_mu)  printf("  %8s", "μ");
        printf("  │ %8s  %5s  %5s\n", "true_h", "acc%", "prop#");
        printf("  ───── │ ────────");
        if (learn_sz)  printf("  ────────");
        if (learn_rho) printf("  ────────");
        if (learn_mu)  printf("  ────────");
        printf("  │ ────────  ─────  ─────\n");
    }

    double sse = 0; int skip = 100, count = 0;

    for (int t = 0; t < T; t++) {
        float h_est = pmmh_step(s, returns[t]);
        if (t >= skip) { double e = h_est - true_h[t]; sse += e*e; count++; }

        if (print_trace && (t % trace_every == 0 || t == T - 1)) {
            float ar = s.n_proposals > 0 ? 100.0f * s.n_accepts / s.n_proposals : 0;
            printf("  %5d │ %+8.4f", t, h_est);
            if (learn_sz)  printf("  %8.5f", s.sigma_z);
            if (learn_rho) printf("  %8.5f", s.rho);
            if (learn_mu)  printf("  %+8.4f", s.mu);
            printf("  │ %+8.4f  %4.1f%%  %5d\n", true_h[t], ar, s.n_proposals);
        }
    }

    TestResult r;
    r.final_sz = s.sigma_z; r.final_rho = s.rho; r.final_mu = s.mu;
    r.h_rmse = count > 0 ? (float)sqrt(sse / count) : 0;
    r.accept_rate = s.n_proposals > 0 ? (float)s.n_accepts / s.n_proposals : 0;

    if (print_trace) {
        printf("  Final: σ_z=%.5f", r.final_sz);
        if (learn_rho) printf("  ρ=%.5f", r.final_rho);
        if (learn_mu) printf("  μ=%.4f", r.final_mu);
        printf("  │  accept=%.1f%%  RMSE=%.4f\n", 100*r.accept_rate, r.h_rmse);
        printf("└");
        for (int p = 0; p < 75; p++) printf("─");
        printf("┘\n");
    }
    return r;
}

// Regime change variant
static TestResult run_regime_test(int T, int change_t,
                                   float true_rho, float sz1, float sz2, float true_mu,
                                   int n_pf, int window, float ps_sz, int trace_every) {
    printf("\n┌── REGIME CHANGE: σ_z %.2f→%.2f at t=%d ", sz1, sz2, change_t);
    for (int p = 0; p < 32; p++) printf("─");
    printf("┐\n");
    printf("  N_pf=%d  W=%d  prop_std=%.3f\n", n_pf, window, ps_sz);

    PMMHConfig cfg;
    cfg.n_pf = n_pf; cfg.window = window;
    cfg.learn_sigma_z = true; cfg.learn_rho = false; cfg.learn_mu = false;
    cfg.prop_std_sz = ps_sz; cfg.prop_std_rho = 0; cfg.prop_std_mu = 0;
    cfg.fixed_rho = true_rho; cfg.fixed_mu = true_mu; cfg.fixed_sigma_z = 0;

    PMMHState s = pmmh_create(cfg, sz1, true_rho, true_mu, 42);

    DGPState dgp;
    dgp.rho = true_rho; dgp.sigma_z = sz1; dgp.mu = true_mu; dgp.h = true_mu;

    printf("  %5s │ %8s  %8s  │ %8s  │ true_σ_z  acc%%\n",
           "tick", "h_est", "σ_z", "true_h");
    printf("  ───── │ ────────  ────────  │ ────────  │ ────────  ────\n");

    double sse = 0; int skip = 100, count = 0;

    for (int t = 0; t < T; t++) {
        if (t == change_t) {
            dgp.sigma_z = sz2;
            printf("  ───── │ ──── σ_z JUMPS TO %.2f ─────────────────────────────\n", sz2);
        }

        float y = dgp_step(dgp);
        float h_est = pmmh_step(s, y);

        if (t >= skip) { double e = h_est - dgp.h; sse += e*e; count++; }

        if (t % trace_every == 0 || t == T - 1 ||
            (t >= change_t - 1 && t <= change_t + 1)) {
            float ar = s.n_proposals > 0 ? 100.0f * s.n_accepts / s.n_proposals : 0;
            printf("  %5d │ %+8.4f  %8.5f  │ %+8.4f  │ %.2f      %4.1f%%\n",
                   t, h_est, s.sigma_z, dgp.h,
                   (t < change_t) ? sz1 : sz2, ar);
        }
    }

    TestResult r;
    r.final_sz = s.sigma_z;
    r.h_rmse = count > 0 ? (float)sqrt(sse / count) : 0;
    r.accept_rate = s.n_proposals > 0 ? (float)s.n_accepts / s.n_proposals : 0;

    printf("  Final: σ_z=%.5f (true=%.2f)  accept=%.1f%%  RMSE=%.4f\n",
           r.final_sz, sz2, 100*r.accept_rate, r.h_rmse);
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

    // Tuning: bigger proposals, shorter windows than CPMMH attempt
    int N_pf = 2000;
    int W = 15;
    float ps_sz = 0.05f;
    float ps_rho = 0.015f;
    float ps_mu = 0.30f;

    printf("═══════════════════════════════════════════════════════════════════════════\n");
    printf("  PMMH Convergence Test\n");
    printf("  True DGP: ρ=%.2f  σ_z=%.2f  μ=%.1f\n", true_rho, true_sz, true_mu);
    printf("  N_pf=%d (×2=%d total)  W=%d  prop_std(σ_z)=%.3f\n",
           N_pf, 2*N_pf, W, ps_sz);
    printf("═══════════════════════════════════════════════════════════════════════════\n");

    // ===== TEST 1: σ_z only, moderate =====
    g_rng = pcg32_seed(42);
    TestResult r1 = run_test("TEST 1: σ_z only (0.05→0.15)",
        3000, true_rho, true_sz, true_mu,
        true_rho, 0.05f, true_mu,
        true, false, false,
        N_pf, W, ps_sz, 0, 0, 200);

    // ===== TEST 2: σ_z only, severe =====
    g_rng = pcg32_seed(42);
    TestResult r2 = run_test("TEST 2: σ_z only (0.50→0.15)",
        3000, true_rho, true_sz, true_mu,
        true_rho, 0.50f, true_mu,
        true, false, false,
        N_pf, W, ps_sz, 0, 0, 200);

    // ===== TEST 3: σ_z + μ =====
    g_rng = pcg32_seed(42);
    TestResult r3 = run_test("TEST 3: σ_z + μ (0.05/-3.0 → 0.15/-4.5)",
        5000, true_rho, true_sz, true_mu,
        true_rho, 0.05f, -3.0f,
        true, false, true,
        N_pf, W, ps_sz, 0, ps_mu, 300);

    // ===== TEST 4: σ_z + ρ =====
    g_rng = pcg32_seed(42);
    TestResult r4 = run_test("TEST 4: σ_z + ρ (0.05/0.85 → 0.15/0.98)",
        5000, true_rho, true_sz, true_mu,
        0.85f, 0.05f, true_mu,
        true, true, false,
        N_pf, W, ps_sz, ps_rho, 0, 300);

    // ===== TEST 5: all three =====
    g_rng = pcg32_seed(42);
    TestResult r5 = run_test("TEST 5: σ_z + ρ + μ (all three)",
        8000, true_rho, true_sz, true_mu,
        0.85f, 0.05f, -3.0f,
        true, true, true,
        N_pf, W, ps_sz, ps_rho, ps_mu, 500);

    // ===== TEST 6: Regime change =====
    g_rng = pcg32_seed(42);
    TestResult r6 = run_regime_test(5000, 2000,
        true_rho, true_sz, 0.40f, true_mu,
        N_pf, W, ps_sz, 200);

    // ===== TEST 7: Tuning grid — window × proposal_std =====
    printf("\n┌── TEST 7: Tuning grid (W × prop_std) ─────────────────────────────────┐\n");
    printf("  %3s  %6s │ %8s │ %5s │ %8s │ %5s\n",
           "W", "p_std", "σ_z_est", "acc%", "RMSE", "prop#");
    printf("  ─── ────── │ ──────── │ ───── │ ──────── │ ─────\n");

    int ws[]     = {5, 10, 15, 25, 50};
    float ps[]   = {0.02f, 0.05f, 0.10f, 0.15f};

    for (int wi = 0; wi < 5; wi++) {
        for (int pi = 0; pi < 4; pi++) {
            g_rng = pcg32_seed(42);
            TestResult r = run_test("", 3000,
                true_rho, true_sz, true_mu,
                true_rho, 0.05f, true_mu,
                true, false, false,
                N_pf, ws[wi], ps[pi], 0, 0,
                99999, false);
            printf("  %3d  %6.3f │ %8.5f │ %4.1f%% │ %8.4f │ %5d\n",
                   ws[wi], ps[pi], r.final_sz, 100*r.accept_rate, r.h_rmse,
                   (int)(3000.0f / ws[wi]));
        }
        printf("  ─── ────── │ ──────── │ ───── │ ──────── │ ─────\n");
    }
    printf("└");
    for (int p = 0; p < 75; p++) printf("─");
    printf("┘\n");

    // ===== TEST 8: Particle count sweep =====
    printf("\n┌── TEST 8: Particle count sweep (W=%d, prop_std=%.3f) ─────────────────┐\n", W, ps_sz);
    printf("  %6s │ %8s │ %5s │ %8s\n", "N_pf", "σ_z_est", "acc%", "RMSE");
    printf("  ────── │ ──────── │ ───── │ ────────\n");

    int npfs[] = {500, 1000, 2000, 5000, 10000};
    for (int ni = 0; ni < 5; ni++) {
        g_rng = pcg32_seed(42);
        TestResult r = run_test("", 3000,
            true_rho, true_sz, true_mu,
            true_rho, 0.05f, true_mu,
            true, false, false,
            npfs[ni], W, ps_sz, 0, 0,
            99999, false);
        printf("  %6d │ %8.5f │ %4.1f%% │ %8.4f\n",
               npfs[ni], r.final_sz, 100*r.accept_rate, r.h_rmse);
    }
    printf("└");
    for (int p = 0; p < 75; p++) printf("─");
    printf("┘\n");

    // ===== ORACLE =====
    printf("\n┌── ORACLE COMPARISON ──────────────────────────────────────────────────┐\n");
    g_rng = pcg32_seed(42);
    TestResult oracle = run_test("Oracle", 3000,
        true_rho, true_sz, true_mu,
        true_rho, true_sz, true_mu,
        false, false, false,
        N_pf, W, 0, 0, 0, 99999, false);
    printf("  Oracle RMSE = %.4f  (N_pf=%d)\n", oracle.h_rmse, N_pf);
    printf("  ──────────────────────────────────────────────────────────────────────\n");
    printf("  Test 1 (σ_z from 0.05):  RMSE=%.4f  (vs oracle: %+.1f%%)\n",
           r1.h_rmse, 100*(r1.h_rmse/oracle.h_rmse - 1));
    printf("  Test 2 (σ_z from 0.50):  RMSE=%.4f  (vs oracle: %+.1f%%)\n",
           r2.h_rmse, 100*(r2.h_rmse/oracle.h_rmse - 1));
    printf("  Test 3 (σ_z + μ):        RMSE=%.4f  (vs oracle: %+.1f%%)\n",
           r3.h_rmse, 100*(r3.h_rmse/oracle.h_rmse - 1));
    printf("  Test 4 (σ_z + ρ):        RMSE=%.4f  (vs oracle: %+.1f%%)\n",
           r4.h_rmse, 100*(r4.h_rmse/oracle.h_rmse - 1));
    printf("  Test 5 (all three):      RMSE=%.4f  (vs oracle: %+.1f%%)\n",
           r5.h_rmse, 100*(r5.h_rmse/oracle.h_rmse - 1));
    printf("═══════════════════════════════════════════════════════════════════════════\n");

    return 0;
}
