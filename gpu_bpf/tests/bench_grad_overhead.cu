/**
 * @file bench_grad_overhead.cu
 * @brief Benchmark: gradient kernel overhead breakdown
 *
 * Measures per-tick latency for BPF with mu+rho learning enabled.
 * Breaks down: total tick, vanilla tick, gradient overhead.
 * Establishes baseline before fusing kernel 14 + reductions.
 *
 * Usage: ./bench_grad [N]    (default N=4096)
 *
 * Build:
 *   nvcc -O3 bench_grad_overhead.cu gpu_bpf_ptx_full.cu -o bench_grad -lcuda -lcurand
 */

#include "gpu_bpf_full.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#ifdef _WIN32
#include <windows.h>
static double now_us(void) {
    static LARGE_INTEGER freq = {0};
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart * 1e6;
}
#else
#include <time.h>
static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3;
}
#endif

static int cmp_float(const void* a, const void* b) {
    float fa = *(const float*)a, fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

typedef struct {
    float median;
    float p99;
    float mean;
} Stats;

static Stats compute_stats(float* t, int n) {
    qsort(t, n, sizeof(float), cmp_float);
    Stats s;
    s.median = t[n / 2];
    s.p99    = t[(int)(n * 0.99)];
    double sum = 0;
    for (int i = 0; i < n; i++) sum += t[i];
    s.mean = (float)(sum / n);
    return s;
}

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

static float* generate_observations(int T, double mu, double rho,
                                     double sigma_z, double nu_obs,
                                     unsigned long long seed) {
    float* y = (float*)malloc(T * sizeof(float));
    unsigned long long rng = seed;
    double h = mu + (sigma_z / sqrt(1.0 - rho * rho)) * host_randn(&rng);
    for (int t = 0; t < T; t++) {
        if (t > 0)
            h = mu + rho * (h - mu) + sigma_z * host_randn(&rng);
        y[t] = (float)(exp(h / 2.0) * host_sample_t(&rng, nu_obs));
    }
    return y;
}

// ═════════════════════════════════════════════════════════════════
// Bench helper
// ═════════════════════════════════════════════════════════════════

static Stats bench_one(int N, int learn_mode, int learn_rho,
                       const float* y, int T, int warmup, int measure) {
    GpuBpfState* s = gpu_bpf_create(N, 0.98f, 0.15f, -1.0f, 0.0f, 5.0f, 42);
    if (learn_mode > 0) {
        gpu_bpf_enable_mu_learning(s, learn_mode, 50, 0.1f, 10.0f, 0.667f);
        if (learn_rho)
            gpu_bpf_enable_rho_learning(s, 1);
    }

    for (int t = 0; t < warmup; t++)
        gpu_bpf_step(s, y[t % T]);
    cudaDeviceSynchronize();

    float* times = (float*)malloc(measure * sizeof(float));
    for (int i = 0; i < measure; i++) {
        cudaDeviceSynchronize();
        double t0 = now_us();
        gpu_bpf_step(s, y[(warmup + i) % T]);
        double t1 = now_us();
        times[i] = (float)(t1 - t0);
    }

    Stats st = compute_stats(times, measure);
    free(times);
    gpu_bpf_destroy(s);
    return st;
}

// ═════════════════════════════════════════════════════════════════
// Main
// ═════════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    int N = 4096;
    if (argc > 1) N = atoi(argv[1]);

    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  BPF Gradient Overhead — Baseline Before Kernel Fusion\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("  GPU:       %s  (SM_%d%d, %d SMs)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("  Particles: %d\n", N);
    printf("  Warmup:    500 ticks\n");
    printf("  Measure:   5000 ticks\n");
    printf("  K=50 (gradient accumulation window)\n\n");

    int T = 10000, warmup = 500, measure = 5000;
    float* y = generate_observations(T, -1.0, 0.98, 0.15, 5.0, 42ULL);

    // ── Run 3 configs ──
    printf("  Running vanilla BPF ...\n");
    Stats van = bench_one(N, 0, 0, y, T, warmup, measure);

    printf("  Running BPF + μ learning ...\n");
    Stats mu = bench_one(N, 1, 0, y, T, warmup, measure);

    printf("  Running BPF + μ+ρ learning ...\n");
    Stats both = bench_one(N, 1, 1, y, T, warmup, measure);
    printf("\n");

    // ── Latency table ──
    printf("  ┌─────────────────────┬──────────┬──────────┬──────────┐\n");
    printf("  │ Config              │ Median   │ Mean     │ P99      │\n");
    printf("  │                     │ (μs)     │ (μs)     │ (μs)     │\n");
    printf("  ├─────────────────────┼──────────┼──────────┼──────────┤\n");
    printf("  │ Vanilla BPF         │ %7.1f  │ %7.1f  │ %7.1f  │\n",
           van.median, van.mean, van.p99);
    printf("  │ + μ learning        │ %7.1f  │ %7.1f  │ %7.1f  │\n",
           mu.median, mu.mean, mu.p99);
    printf("  │ + μ+ρ learning      │ %7.1f  │ %7.1f  │ %7.1f  │\n",
           both.median, both.mean, both.p99);
    printf("  └─────────────────────┴──────────┴──────────┴──────────┘\n\n");

    // ── Overhead breakdown ──
    float oh_mu   = mu.median   - van.median;
    float oh_both = both.median - van.median;
    float oh_rho  = both.median - mu.median;

    // μ learning  = h_prev memcpy + kernel14(4 out) + 2 reduce_sum
    // μ+ρ learning = h_prev memcpy + kernel14(4 out) + 4 reduce_sum
    // Δ(μ+ρ − μ) = 2 extra reduce_sum launches
    float per_reduce = oh_rho / 2.0f;

    printf("  ┌───────────────────────────────────────────────────────────┐\n");
    printf("  │ Overhead Breakdown (median)                              │\n");
    printf("  ├───────────────────────────────────────────────────────────┤\n");
    printf("  │                                                           │\n");
    printf("  │ μ learning overhead:       %+.1f μs  (%+.1f%% of tick)   │\n",
           oh_mu, oh_mu / van.median * 100);
    printf("  │   = h_prev D2D + kernel14 + 2× reduce_sum                │\n");
    printf("  │                                                           │\n");
    printf("  │ μ+ρ learning overhead:     %+.1f μs  (%+.1f%% of tick)   │\n",
           oh_both, oh_both / van.median * 100);
    printf("  │   = h_prev D2D + kernel14 + 4× reduce_sum                │\n");
    printf("  │                                                           │\n");
    printf("  │ Extra ρ cost (2 reduces):  %+.1f μs                      │\n", oh_rho);
    printf("  │   → ~%.1f μs per reduce_sum launch                       │\n", per_reduce);
    printf("  │                                                           │\n");
    printf("  ├───────────────────────────────────────────────────────────┤\n");
    printf("  │ Fusion opportunity                                        │\n");
    printf("  │                                                           │\n");
    printf("  │ Current:  1× kernel14  + 4× reduce_sum  = 5 launches     │\n");
    printf("  │ Fused:    1× kernel (compute+reduce)     = 1 launch       │\n");
    printf("  │ Saves:    4 launches × ~%.1f μs = ~%.1f μs/tick           │\n",
           per_reduce, per_reduce * 4);
    printf("  │ Projected overhead:  ~%.1f μs  (%+.1f%% of tick)          │\n",
           oh_both - per_reduce * 4,
           (oh_both - per_reduce * 4) / van.median * 100);
    printf("  │                                                           │\n");
    printf("  └───────────────────────────────────────────────────────────┘\n\n");

    // ── Throughput ──
    printf("  ┌─────────────────────┬──────────────────┐\n");
    printf("  │ Config              │ Throughput        │\n");
    printf("  ├─────────────────────┼──────────────────┤\n");
    printf("  │ Vanilla BPF         │ %7.0f ticks/s  │\n", 1e6 / van.median);
    printf("  │ + μ learning        │ %7.0f ticks/s  │\n", 1e6 / mu.median);
    printf("  │ + μ+ρ learning      │ %7.0f ticks/s  │\n", 1e6 / both.median);
    printf("  └─────────────────────┴──────────────────┘\n\n");

    free(y);
    return 0;
}
