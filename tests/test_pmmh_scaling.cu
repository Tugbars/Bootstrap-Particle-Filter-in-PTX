// =============================================================================
// PMMH Particle Scaling Test
//
// The CPU test proved: at N_pf=2000, likelihood estimates are too noisy for
// MH to make meaningful decisions. This test cranks N_pf up to GPU-scale
// to find where PMMH actually converges.
//
// Still CPU (will be slow at high N) but proves the algorithm works before
// porting to GPU.
//
// Tests:
//   1. Particle sweep: 2K → 5K → 10K → 25K → 50K → 100K
//      All learning σ_z from 0.05→0.15 with W=15
//   2. Best N from sweep: regime change test
//   3. Best N: learn all three params
//   4. Window sweep at best N (find optimal W at high particle count)
//
// Build: nvcc -O3 test_pmmh_scaling.cu -o test_pmmh_scaling
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <ctime>

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
// BPF
// =============================================================================

struct BPF {
    int N;
    float* h;
    float* h2;
    float* w;
    float* log_w;
    float* cdf;
    float h_est;
    double cum_ll;
    PCG32 rng;
};

static BPF bpf_create(int N, int seed) {
    BPF f;
    f.N = N;
    f.h     = (float*)malloc(N * sizeof(float));
    f.h2    = (float*)malloc(N * sizeof(float));
    f.w     = (float*)malloc(N * sizeof(float));
    f.log_w = (float*)malloc(N * sizeof(float));
    f.cdf   = (float*)malloc(N * sizeof(float));
    f.h_est = 0; f.cum_ll = 0;
    f.rng = pcg32_seed(seed);
    return f;
}

static void bpf_destroy(BPF& f) {
    free(f.h); free(f.h2); free(f.w); free(f.log_w); free(f.cdf);
}

static void bpf_init(BPF& f, float mu, float sigma_stat) {
    for (int i = 0; i < f.N; i++) {
        f.h[i] = mu + sigma_stat * pcg32_randn(f.rng);
        f.w[i] = 1.0f / f.N;
    }
    f.cum_ll = 0;
}

static void bpf_copy_particles(BPF& dst, const BPF& src) {
    memcpy(dst.h, src.h, dst.N * sizeof(float));
    for (int i = 0; i < dst.N; i++) dst.w[i] = 1.0f / dst.N;
    dst.cum_ll = 0;
}

static float bpf_step(BPF& f, float y_t, float rho, float sigma_z, float mu) {
    int N = f.N;

    for (int i = 0; i < N; i++) {
        float z = pcg32_randn(f.rng);
        f.h[i] = mu + rho * (f.h[i] - mu) + sigma_z * z;
    }

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

    for (int i = 0; i < N; i++) f.w[i] /= sum_w;

    float hm = 0;
    for (int i = 0; i < N; i++) hm += f.w[i] * f.h[i];
    f.h_est = hm;

    f.cdf[0] = f.w[0];
    for (int i = 1; i < N; i++) f.cdf[i] = f.cdf[i-1] + f.w[i];

    float u = pcg32_float(f.rng) / N;
    int j = 0;
    for (int i = 0; i < N; i++) {
        float target = u + (float)i / N;
        while (j < N - 1 && f.cdf[j] < target) j++;
        f.h2[i] = f.h[j];
    }
    // swap
    float* tmp = f.h; f.h = f.h2; f.h2 = tmp;
    for (int i = 0; i < N; i++) f.w[i] = 1.0f / N;

    return step_ll;
}

// =============================================================================
// PMMH
// =============================================================================

struct PMMHConfig {
    int n_pf, window;
    bool learn_sigma_z, learn_rho, learn_mu;
    float prop_std_sz, prop_std_rho, prop_std_mu;
    float fixed_rho, fixed_mu, fixed_sigma_z;
};

struct PMMHState {
    float sigma_z, rho, mu;
    float prop_sigma_z, prop_rho, prop_mu;
    BPF main_f, shadow_f;
    int window_tick, n_proposals, n_accepts;
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

    bpf_copy_particles(s.shadow_f, s.main_f);
    s.main_f.cum_ll = 0;
    s.shadow_f.cum_ll = 0;
    // Fresh shadow RNG
    s.shadow_f.rng = pcg32_seed((int)(s.sigma_z * 1e6f) + s.n_proposals * 137 + 9973);
    s.window_tick = 0;
}

static PMMHState pmmh_create(PMMHConfig cfg, float init_sz, float init_rho,
                              float init_mu, int seed) {
    PMMHState s;
    s.cfg = cfg;
    s.sigma_z = cfg.learn_sigma_z ? init_sz : cfg.fixed_sigma_z;
    s.rho     = cfg.learn_rho ? init_rho : cfg.fixed_rho;
    s.mu      = cfg.learn_mu ? init_mu : cfg.fixed_mu;

    s.main_f   = bpf_create(cfg.n_pf, seed);
    s.shadow_f = bpf_create(cfg.n_pf, seed + 9973);

    float sigma_stat = s.sigma_z / sqrtf(1.0f - s.rho * s.rho + 1e-8f);
    bpf_init(s.main_f, s.mu, sigma_stat);

    s.window_tick = 0;
    s.n_proposals = 0;
    s.n_accepts = 0;
    return s;
}

static void pmmh_destroy(PMMHState& s) {
    bpf_destroy(s.main_f);
    bpf_destroy(s.shadow_f);
}

static float pmmh_step(PMMHState& s, float y_t) {
    if (s.window_tick == 0) {
        if (s.n_proposals == 0) pmmh_propose(s);
        s.n_proposals++;
    }

    bpf_step(s.main_f, y_t, s.rho, s.sigma_z, s.mu);
    bpf_step(s.shadow_f, y_t, s.prop_rho, s.prop_sigma_z, s.prop_mu);
    s.window_tick++;

    if (s.window_tick >= s.cfg.window) {
        float log_alpha = (float)(s.shadow_f.cum_ll - s.main_f.cum_ll);
        float u = randf();
        bool accept = (logf(u + 1e-30f) < log_alpha);

        if (accept) {
            s.sigma_z = s.prop_sigma_z;
            s.rho = s.prop_rho;
            s.mu = s.prop_mu;
            // Swap particle arrays
            float* tmp;
            tmp = s.main_f.h;  s.main_f.h  = s.shadow_f.h;  s.shadow_f.h  = tmp;
            tmp = s.main_f.h2; s.main_f.h2 = s.shadow_f.h2; s.shadow_f.h2 = tmp;
            PCG32 rtmp = s.main_f.rng; s.main_f.rng = s.shadow_f.rng; s.shadow_f.rng = rtmp;
            s.n_accepts++;
        }
        pmmh_propose(s);
    }

    return s.main_f.h_est;
}

// =============================================================================
// Pre-generate data (shared across all particle counts)
// =============================================================================

static void generate_data(float* returns, float* true_h, int T,
                           float rho, float sz, float mu) {
    DGPState d;
    d.rho = rho; d.sigma_z = sz; d.mu = mu; d.h = mu;
    for (int t = 0; t < T; t++) {
        returns[t] = dgp_step(d);
        true_h[t] = d.h;
    }
}

// =============================================================================
// Main
// =============================================================================

int main() {
    float true_rho = 0.98f, true_sz = 0.15f, true_mu = -4.5f;
    int T = 3000;
    int W = 15;
    float ps_sz = 0.05f;

    printf("═══════════════════════════════════════════════════════════════════════════\n");
    printf("  PMMH Particle Scaling Test\n");
    printf("  True DGP: ρ=%.2f  σ_z=%.2f  μ=%.1f  │  T=%d  W=%d  prop_std=%.3f\n",
           true_rho, true_sz, true_mu, T, W, ps_sz);
    printf("═══════════════════════════════════════════════════════════════════════════\n");

    // =========================================================================
    // TEST 1: Particle count sweep (σ_z only, 0.05→0.15)
    // =========================================================================
    printf("\n┌── TEST 1: Particle count sweep (σ_z from 0.05→0.15) ─────────────────┐\n");
    printf("  %7s │ %8s │ %5s │ %8s │ %5s │ %6s\n",
           "N_pf", "σ_z_est", "acc%", "RMSE", "prop#", "sec");
    printf("  ─────── │ ──────── │ ───── │ ──────── │ ───── │ ──────\n");

    int npfs[] = {2000, 5000, 10000, 25000, 50000, 100000};
    int n_npfs = 6;

    // Pre-generate data
    g_rng = pcg32_seed(42);
    float* returns = (float*)malloc(T * sizeof(float));
    float* true_h  = (float*)malloc(T * sizeof(float));
    generate_data(returns, true_h, T, true_rho, true_sz, true_mu);

    int best_npf = 2000;
    float best_rmse = 999.0f;

    for (int ni = 0; ni < n_npfs; ni++) {
        g_rng = pcg32_seed(123);  // same proposal sequence

        PMMHConfig cfg;
        cfg.n_pf = npfs[ni]; cfg.window = W;
        cfg.learn_sigma_z = true; cfg.learn_rho = false; cfg.learn_mu = false;
        cfg.prop_std_sz = ps_sz; cfg.prop_std_rho = 0; cfg.prop_std_mu = 0;
        cfg.fixed_rho = true_rho; cfg.fixed_mu = true_mu; cfg.fixed_sigma_z = 0;

        PMMHState s = pmmh_create(cfg, 0.05f, true_rho, true_mu, 42);

        clock_t t0 = clock();
        double sse = 0; int cnt = 0;
        for (int t = 0; t < T; t++) {
            float h_est = pmmh_step(s, returns[t]);
            if (t >= 100) { double e = h_est - true_h[t]; sse += e*e; cnt++; }
        }
        clock_t t1 = clock();
        float elapsed = (float)(t1 - t0) / CLOCKS_PER_SEC;

        float rmse = sqrtf((float)(sse / cnt));
        float ar = s.n_proposals > 0 ? 100.0f * s.n_accepts / s.n_proposals : 0;

        printf("  %7d │ %8.5f │ %4.1f%% │ %8.4f │ %5d │ %5.1fs\n",
               npfs[ni], s.sigma_z, ar, rmse, s.n_proposals, elapsed);

        if (rmse < best_rmse) { best_rmse = rmse; best_npf = npfs[ni]; }
        pmmh_destroy(s);

        // Skip extremely slow runs
        if (elapsed > 120.0f) {
            printf("  (skipping larger N — too slow on CPU)\n");
            break;
        }
    }
    printf("  Best: N_pf=%d  RMSE=%.4f\n", best_npf, best_rmse);
    printf("└");
    for (int p = 0; p < 75; p++) printf("─");
    printf("┘\n");

    // =========================================================================
    // TEST 2: Window sweep at high particle count
    // =========================================================================
    int N_hi = std::min(best_npf, 25000);  // cap for CPU speed
    printf("\n┌── TEST 2: Window sweep at N_pf=%d ──────────────────────────────────┐\n", N_hi);
    printf("  %5s │ %8s │ %5s │ %8s │ %5s\n",
           "W", "σ_z_est", "acc%", "RMSE", "prop#");
    printf("  ───── │ ──────── │ ───── │ ──────── │ ─────\n");

    int ws[] = {5, 10, 15, 20, 30, 50, 100};
    for (int wi = 0; wi < 7; wi++) {
        g_rng = pcg32_seed(123);

        PMMHConfig cfg;
        cfg.n_pf = N_hi; cfg.window = ws[wi];
        cfg.learn_sigma_z = true; cfg.learn_rho = false; cfg.learn_mu = false;
        cfg.prop_std_sz = ps_sz; cfg.prop_std_rho = 0; cfg.prop_std_mu = 0;
        cfg.fixed_rho = true_rho; cfg.fixed_mu = true_mu; cfg.fixed_sigma_z = 0;

        PMMHState s = pmmh_create(cfg, 0.05f, true_rho, true_mu, 42);

        double sse = 0; int cnt = 0;
        for (int t = 0; t < T; t++) {
            float h_est = pmmh_step(s, returns[t]);
            if (t >= 100) { double e = h_est - true_h[t]; sse += e*e; cnt++; }
        }
        float rmse = sqrtf((float)(sse / cnt));
        float ar = s.n_proposals > 0 ? 100.0f * s.n_accepts / s.n_proposals : 0;
        printf("  %5d │ %8.5f │ %4.1f%% │ %8.4f │ %5d\n",
               ws[wi], s.sigma_z, ar, rmse, s.n_proposals);
        pmmh_destroy(s);
    }
    printf("└");
    for (int p = 0; p < 75; p++) printf("─");
    printf("┘\n");

    // =========================================================================
    // TEST 3: Proposal std sweep at high N
    // =========================================================================
    printf("\n┌── TEST 3: Proposal std sweep at N_pf=%d, W=%d ─────────────────────┐\n", N_hi, W);
    printf("  %6s │ %8s │ %5s │ %8s\n", "p_std", "σ_z_est", "acc%", "RMSE");
    printf("  ────── │ ──────── │ ───── │ ────────\n");

    float pss[] = {0.01f, 0.02f, 0.03f, 0.05f, 0.08f, 0.12f, 0.20f};
    for (int pi = 0; pi < 7; pi++) {
        g_rng = pcg32_seed(123);

        PMMHConfig cfg;
        cfg.n_pf = N_hi; cfg.window = W;
        cfg.learn_sigma_z = true; cfg.learn_rho = false; cfg.learn_mu = false;
        cfg.prop_std_sz = pss[pi]; cfg.prop_std_rho = 0; cfg.prop_std_mu = 0;
        cfg.fixed_rho = true_rho; cfg.fixed_mu = true_mu; cfg.fixed_sigma_z = 0;

        PMMHState s = pmmh_create(cfg, 0.05f, true_rho, true_mu, 42);

        double sse = 0; int cnt = 0;
        for (int t = 0; t < T; t++) {
            float h_est = pmmh_step(s, returns[t]);
            if (t >= 100) { double e = h_est - true_h[t]; sse += e*e; cnt++; }
        }
        float rmse = sqrtf((float)(sse / cnt));
        float ar = s.n_proposals > 0 ? 100.0f * s.n_accepts / s.n_proposals : 0;
        printf("  %6.3f │ %8.5f │ %4.1f%% │ %8.4f\n", pss[pi], s.sigma_z, ar, rmse);
        pmmh_destroy(s);
    }
    printf("└");
    for (int p = 0; p < 75; p++) printf("─");
    printf("┘\n");

    // =========================================================================
    // TEST 4: Regime change at high N (the real test)
    // =========================================================================
    printf("\n┌── TEST 4: Regime change at N_pf=%d ─────────────────────────────────┐\n", N_hi);
    printf("  σ_z: 0.15 → 0.40 at t=2000, T=5000\n\n");
    {
        g_rng = pcg32_seed(42);

        PMMHConfig cfg;
        cfg.n_pf = N_hi; cfg.window = W;
        cfg.learn_sigma_z = true; cfg.learn_rho = false; cfg.learn_mu = false;
        cfg.prop_std_sz = ps_sz; cfg.prop_std_rho = 0; cfg.prop_std_mu = 0;
        cfg.fixed_rho = true_rho; cfg.fixed_mu = true_mu; cfg.fixed_sigma_z = 0;

        PMMHState s = pmmh_create(cfg, true_sz, true_rho, true_mu, 42);

        DGPState dgp;
        dgp.rho = true_rho; dgp.sigma_z = true_sz; dgp.mu = true_mu; dgp.h = true_mu;

        int T2 = 5000;
        int change_t = 2000;

        printf("  %5s │ %8s  %8s  │ %8s  │ true_σ_z  acc%%\n",
               "tick", "h_est", "σ_z", "true_h");
        printf("  ───── │ ────────  ────────  │ ────────  │ ────────  ────\n");

        for (int t = 0; t < T2; t++) {
            if (t == change_t) {
                dgp.sigma_z = 0.40f;
                printf("  ───── │ ──── σ_z JUMPS TO 0.40 ─────────────────────────────\n");
            }

            float y = dgp_step(dgp);
            float h_est = pmmh_step(s, y);

            if (t % 200 == 0 || t == T2 - 1 ||
                (t >= change_t - 1 && t <= change_t + 1)) {
                float ar = s.n_proposals > 0 ? 100.0f * s.n_accepts / s.n_proposals : 0;
                printf("  %5d │ %+8.4f  %8.5f  │ %+8.4f  │ %.2f      %4.1f%%\n",
                       t, h_est, s.sigma_z, dgp.h,
                       (t < change_t) ? 0.15f : 0.40f, ar);
            }
        }
        float ar = s.n_proposals > 0 ? 100.0f * s.n_accepts / s.n_proposals : 0;
        printf("\n  Final: σ_z=%.5f (true=0.40)  accept=%.1f%%\n", s.sigma_z, ar);
        pmmh_destroy(s);
    }
    printf("└");
    for (int p = 0; p < 75; p++) printf("─");
    printf("┘\n");

    // =========================================================================
    // TEST 5: Learn all three at high N
    // =========================================================================
    printf("\n┌── TEST 5: All three params at N_pf=%d ──────────────────────────────┐\n", N_hi);
    printf("  Init: σ_z=0.05  ρ=0.85  μ=-3.0  →  True: σ_z=0.15  ρ=0.98  μ=-4.5\n\n");
    {
        g_rng = pcg32_seed(42);

        // Re-generate data
        DGPState dgp;
        dgp.rho = true_rho; dgp.sigma_z = true_sz; dgp.mu = true_mu; dgp.h = true_mu;
        int T3 = 8000;
        float* ret3 = (float*)malloc(T3 * sizeof(float));
        float* th3  = (float*)malloc(T3 * sizeof(float));
        for (int t = 0; t < T3; t++) { ret3[t] = dgp_step(dgp); th3[t] = dgp.h; }

        g_rng = pcg32_seed(123);

        PMMHConfig cfg;
        cfg.n_pf = N_hi; cfg.window = W;
        cfg.learn_sigma_z = true; cfg.learn_rho = true; cfg.learn_mu = true;
        cfg.prop_std_sz = 0.05f; cfg.prop_std_rho = 0.015f; cfg.prop_std_mu = 0.30f;
        cfg.fixed_rho = 0; cfg.fixed_mu = 0; cfg.fixed_sigma_z = 0;

        PMMHState s = pmmh_create(cfg, 0.05f, 0.85f, -3.0f, 42);

        printf("  %5s │ %8s  %8s  %8s  %8s │ %8s  %5s\n",
               "tick", "h_est", "σ_z", "ρ", "μ", "true_h", "acc%");
        printf("  ───── │ ────────  ────────  ────────  ──────── │ ────────  ─────\n");

        double sse = 0; int cnt = 0;
        for (int t = 0; t < T3; t++) {
            float h_est = pmmh_step(s, ret3[t]);
            if (t >= 200) { double e = h_est - th3[t]; sse += e*e; cnt++; }

            if (t % 500 == 0 || t == T3 - 1) {
                float ar = s.n_proposals > 0 ? 100.0f * s.n_accepts / s.n_proposals : 0;
                printf("  %5d │ %+8.4f  %8.5f  %8.5f  %+8.4f │ %+8.4f  %4.1f%%\n",
                       t, h_est, s.sigma_z, s.rho, s.mu, th3[t], ar);
            }
        }
        float rmse = sqrtf((float)(sse / cnt));
        float ar = s.n_proposals > 0 ? 100.0f * s.n_accepts / s.n_proposals : 0;
        printf("\n  Final: σ_z=%.5f (true=%.3f)  ρ=%.5f (true=%.3f)  μ=%.4f (true=%.1f)\n",
               s.sigma_z, true_sz, s.rho, true_rho, s.mu, true_mu);
        printf("  RMSE=%.4f  accept=%.1f%%\n", rmse, ar);

        pmmh_destroy(s);
        free(ret3); free(th3);
    }
    printf("└");
    for (int p = 0; p < 75; p++) printf("─");
    printf("┘\n");

    // =========================================================================
    // ORACLE
    // =========================================================================
    printf("\n┌── ORACLE (N_pf=%d, fixed true params) ──────────────────────────────┐\n", N_hi);
    {
        g_rng = pcg32_seed(123);

        PMMHConfig cfg;
        cfg.n_pf = N_hi; cfg.window = W;
        cfg.learn_sigma_z = false; cfg.learn_rho = false; cfg.learn_mu = false;
        cfg.prop_std_sz = 0; cfg.prop_std_rho = 0; cfg.prop_std_mu = 0;
        cfg.fixed_rho = true_rho; cfg.fixed_mu = true_mu; cfg.fixed_sigma_z = true_sz;

        PMMHState s = pmmh_create(cfg, true_sz, true_rho, true_mu, 42);

        double sse = 0; int cnt = 0;
        for (int t = 0; t < T; t++) {
            float h_est = pmmh_step(s, returns[t]);
            if (t >= 100) { double e = h_est - true_h[t]; sse += e*e; cnt++; }
        }
        float rmse = sqrtf((float)(sse / cnt));
        printf("  Oracle RMSE = %.4f\n", rmse);
        pmmh_destroy(s);
    }
    printf("└");
    for (int p = 0; p < 75; p++) printf("─");
    printf("┘\n");

    free(returns); free(true_h);

    printf("═══════════════════════════════════════════════════════════════════════════\n");
    return 0;
}
