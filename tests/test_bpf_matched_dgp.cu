/**
 * @file test_bpf_matched_dgp.cu
 * @brief Matched-DGP test suite for GPU Bootstrap Particle Filter (PTX)
 *
 * All scenarios generate data from the EXACT SV model:
 *   State:       h_t = mu + rho*(h_{t-1} - mu) + sigma_z * eps_t
 *   Observation: y_t = exp(h_t / 2) * eta_t
 *
 * Zero model mismatch -- any remaining error is purely from Monte Carlo
 * variance and particle degeneracy.
 *
 * Tests:
 *   1. Full comparison: KF bound vs BPF vs EWMA vs GARCH
 *   2. Particle sweep (find the efficiency knee)
 *   3. Monte Carlo variance (multi-seed bias analysis)
 *   4. Throughput benchmark
 *
 * Build:
 *   nvcc -O3 test_bpf_matched_dgp.cu gpu_bpf_ptx_full.cu -o test_bpf -lcuda -lcurand
 *
 * Run:
 *   ./test_bpf [--ticks N] [--seed S] [--particles N]
 */

#include "gpu_bpf_full.cuh"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

// =============================================================================
// Timing
// =============================================================================

static double get_time_us(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart * 1e6 / (double)freq.QuadPart;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1e6 + (double)tv.tv_usec;
#endif
}

// =============================================================================
// PCG32 RNG (DGP generation)
// =============================================================================

typedef struct { uint64_t state; uint64_t inc; } pcg32_dgp_t;

static inline uint32_t pcg32_dgp_random(pcg32_dgp_t* rng) {
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static inline void pcg32_dgp_seed(pcg32_dgp_t* rng, uint64_t seed) {
    rng->state = 0;
    rng->inc = (seed * 12345ULL) | 1;
    pcg32_dgp_random(rng);
    rng->state += seed * 67890ULL;
    pcg32_dgp_random(rng);
    pcg32_dgp_random(rng);
}

static inline double pcg32_dgp_double(pcg32_dgp_t* rng) {
    return (pcg32_dgp_random(rng) >> 11) * (1.0 / 2097152.0);
}

static inline double pcg32_dgp_gaussian(pcg32_dgp_t* rng) {
    double u1, u2;
    do { u1 = pcg32_dgp_double(rng); } while (u1 < 1e-10);
    u2 = pcg32_dgp_double(rng);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

static inline double pcg32_dgp_student_t(pcg32_dgp_t* rng, double nu) {
    if (nu > 100.0) return pcg32_dgp_gaussian(rng);
    double z = pcg32_dgp_gaussian(rng);
    double chi_sq = 0.0;
    int inu = (int)nu;
    for (int i = 0; i < inu; i++) {
        double g = pcg32_dgp_gaussian(rng);
        chi_sq += g * g;
    }
    return z / sqrt(chi_sq / nu);
}

// =============================================================================
// Test Data
// =============================================================================

typedef struct {
    double* true_h;
    double* true_vol;
    double* returns;
    int     n_ticks;
    int     scenario_id;
    const char* scenario_name;
    const char* scenario_desc;
    double  dgp_rho;
    double  dgp_sigma_z;
    double  dgp_mu;
    double  dgp_nu_state;
    double  dgp_nu_obs;
} MatchedTestData;

static MatchedTestData* alloc_matched_data(int n) {
    MatchedTestData* d = (MatchedTestData*)calloc(1, sizeof(MatchedTestData));
    d->n_ticks  = n;
    d->true_h   = (double*)malloc(n * sizeof(double));
    d->true_vol = (double*)malloc(n * sizeof(double));
    d->returns  = (double*)malloc(n * sizeof(double));
    return d;
}

static void free_matched_data(MatchedTestData* d) {
    if (!d) return;
    free(d->true_h);
    free(d->true_vol);
    free(d->returns);
    free(d);
}

// =============================================================================
// Core DGP
// =============================================================================

static void generate_matched_series(
    MatchedTestData* data,
    double rho, double sigma_z, double mu,
    double nu_state, double nu_obs,
    pcg32_dgp_t* rng
) {
    int n = data->n_ticks;
    data->dgp_rho      = rho;
    data->dgp_sigma_z  = sigma_z;
    data->dgp_mu       = mu;
    data->dgp_nu_state = nu_state;
    data->dgp_nu_obs   = nu_obs;

    double var_stat = (sigma_z * sigma_z) / fmax(1.0 - rho * rho, 1e-6);
    double h = mu + sqrt(var_stat) * pcg32_dgp_gaussian(rng);

    for (int t = 0; t < n; t++) {
        if (t > 0) {
            double eps = (nu_state > 0.0)
                ? pcg32_dgp_student_t(rng, nu_state)
                : pcg32_dgp_gaussian(rng);
            h = mu + rho * (h - mu) + sigma_z * eps;
        }
        if (h < -12.0) h = -12.0;
        if (h >   4.0) h =  4.0;

        data->true_h[t]   = h;
        data->true_vol[t]  = exp(h / 2.0);

        double eta = (nu_obs > 0.0)
            ? pcg32_dgp_student_t(rng, nu_obs)
            : pcg32_dgp_gaussian(rng);
        data->returns[t] = data->true_vol[t] * eta;
    }
}

// =============================================================================
// Scenarios
// =============================================================================

static MatchedTestData* gen_baseline(int n, int seed) {
    MatchedTestData* d = alloc_matched_data(n);
    d->scenario_id   = 1;
    d->scenario_name = "Baseline (G/t5)";
    d->scenario_desc = "rho=0.97 sz=0.15 mu=-4.5 Gauss-state t(5)-obs";
    pcg32_dgp_t rng; pcg32_dgp_seed(&rng, seed);
    generate_matched_series(d, 0.97, 0.15, -4.5, 0.0, 5.0, &rng);
    return d;
}

static MatchedTestData* gen_student_t(int n, int seed) {
    MatchedTestData* d = alloc_matched_data(n);
    d->scenario_id   = 2;
    d->scenario_name = "Student-t (t5/t5)";
    d->scenario_desc = "rho=0.97 sz=0.15 mu=-4.5 t(5)-state t(5)-obs";
    pcg32_dgp_t rng; pcg32_dgp_seed(&rng, seed);
    generate_matched_series(d, 0.97, 0.15, -4.5, 5.0, 5.0, &rng);
    return d;
}

static MatchedTestData* gen_high_persist(int n, int seed) {
    MatchedTestData* d = alloc_matched_data(n);
    d->scenario_id   = 3;
    d->scenario_name = "High Persist";
    d->scenario_desc = "rho=0.995 sz=0.15 mu=-4.5 t(5)/t(5)";
    pcg32_dgp_t rng; pcg32_dgp_seed(&rng, seed);
    generate_matched_series(d, 0.995, 0.15, -4.5, 5.0, 5.0, &rng);
    return d;
}

static MatchedTestData* gen_high_volvol(int n, int seed) {
    MatchedTestData* d = alloc_matched_data(n);
    d->scenario_id   = 4;
    d->scenario_name = "High Vol-of-Vol";
    d->scenario_desc = "rho=0.97 sz=0.30 mu=-4.5 t(5)/t(5)";
    pcg32_dgp_t rng; pcg32_dgp_seed(&rng, seed);
    generate_matched_series(d, 0.97, 0.30, -4.5, 5.0, 5.0, &rng);
    return d;
}

static MatchedTestData* gen_low_vol(int n, int seed) {
    MatchedTestData* d = alloc_matched_data(n);
    d->scenario_id   = 5;
    d->scenario_name = "Low Vol";
    d->scenario_desc = "rho=0.97 sz=0.15 mu=-6.0 t(5)/t(5)";
    pcg32_dgp_t rng; pcg32_dgp_seed(&rng, seed);
    generate_matched_series(d, 0.97, 0.15, -6.0, 5.0, 5.0, &rng);
    return d;
}

static MatchedTestData* gen_high_vol(int n, int seed) {
    MatchedTestData* d = alloc_matched_data(n);
    d->scenario_id   = 6;
    d->scenario_name = "High Vol";
    d->scenario_desc = "rho=0.97 sz=0.15 mu=-2.0 t(5)/t(5)";
    pcg32_dgp_t rng; pcg32_dgp_seed(&rng, seed);
    generate_matched_series(d, 0.97, 0.15, -2.0, 5.0, 5.0, &rng);
    return d;
}

static MatchedTestData* gen_fast_revert(int n, int seed) {
    MatchedTestData* d = alloc_matched_data(n);
    d->scenario_id   = 7;
    d->scenario_name = "Fast Revert";
    d->scenario_desc = "rho=0.90 sz=0.15 mu=-4.5 t(5)/t(5)";
    pcg32_dgp_t rng; pcg32_dgp_seed(&rng, seed);
    generate_matched_series(d, 0.90, 0.15, -4.5, 5.0, 5.0, &rng);
    return d;
}

static MatchedTestData* gen_gaussian(int n, int seed) {
    MatchedTestData* d = alloc_matched_data(n);
    d->scenario_id   = 8;
    d->scenario_name = "Pure Gaussian";
    d->scenario_desc = "rho=0.97 sz=0.15 mu=-4.5 Gauss/Gauss";
    pcg32_dgp_t rng; pcg32_dgp_seed(&rng, seed);
    generate_matched_series(d, 0.97, 0.15, -4.5, 0.0, 0.0, &rng);
    return d;
}

static MatchedTestData* gen_fast_revert_highvol(int n, int seed) {
    MatchedTestData* d = alloc_matched_data(n);
    d->scenario_id   = 9;
    d->scenario_name = "FastRev+HighVoV";
    d->scenario_desc = "rho=0.85 sz=0.50 mu=-4.5 t(5)/t(5) -- stress test";
    pcg32_dgp_t rng; pcg32_dgp_seed(&rng, seed);
    generate_matched_series(d, 0.85, 0.50, -4.5, 5.0, 5.0, &rng);
    return d;
}

typedef MatchedTestData* (*ScenarioGenFn)(int, int);

static const struct {
    ScenarioGenFn gen;
} ALL_SCENARIOS[] = {
    { gen_baseline },
    { gen_student_t },
    { gen_high_persist },
    { gen_high_volvol },
    { gen_low_vol },
    { gen_high_vol },
    { gen_fast_revert },
    { gen_gaussian },
    { gen_fast_revert_highvol },
};
static const int N_SCENARIOS = sizeof(ALL_SCENARIOS) / sizeof(ALL_SCENARIOS[0]);

// =============================================================================
// Kalman Steady-State RMSE (Analytical Lower Bound)
// =============================================================================

static double kalman_steady_state_rmse(double rho, double sigma_z, double nu_obs) {
    double Q = sigma_z * sigma_z;
    double a = rho * rho;

    double R;
    if (nu_obs <= 0 || nu_obs > 100.0) {
        R = 4.9348;
    } else {
        double tg;
        if      (nu_obs < 3.5) tg = 1.50;
        else if (nu_obs < 4.5) tg = 0.80;
        else if (nu_obs < 5.5) tg = 0.49;
        else if (nu_obs < 7.5) tg = 0.30;
        else                   tg = 0.15;
        R = 4.9348 + 2.0 * tg;
    }

    double b    = Q + R * (1.0 - a);
    double c    = -Q * R;
    double disc = b * b - 4.0 * a * c;
    double P    = (-b + sqrt(disc)) / (2.0 * a);
    return sqrt(P);
}

// =============================================================================
// Classical Baselines
// =============================================================================

static double ewma_rmse(const MatchedTestData* data, double lambda) {
    int n = data->n_ticks;
    int skip = 100;
    double sum_sq = 0.0;
    int count = 0;

    double init_var = 0.0;
    int init_n = (n < 20) ? n : 20;
    for (int t = 0; t < init_n; t++)
        init_var += data->returns[t] * data->returns[t];
    init_var /= init_n;
    if (init_var < 1e-10) init_var = 1e-10;

    double var = init_var;
    for (int t = 0; t < n; t++) {
        if (t >= skip) {
            double h_hat = log(var);
            double err   = h_hat - data->true_h[t];
            sum_sq += err * err;
            count++;
        }
        double y2 = data->returns[t] * data->returns[t];
        var = lambda * var + (1.0 - lambda) * y2;
        if (var < 1e-10) var = 1e-10;
    }
    return sqrt(sum_sq / count);
}

static double garch_rmse(const MatchedTestData* data,
                         double omega, double alpha, double beta) {
    int n = data->n_ticks;
    int skip = 100;
    double sum_sq = 0.0;
    int count = 0;

    double persist = alpha + beta;
    double var = (persist < 0.999) ? omega / (1.0 - persist) : 0.01;
    if (var < 1e-10) var = 1e-10;

    for (int t = 0; t < n; t++) {
        if (t >= skip) {
            double h_hat = log(var);
            double err   = h_hat - data->true_h[t];
            sum_sq += err * err;
            count++;
        }
        double y2 = data->returns[t] * data->returns[t];
        var = omega + alpha * y2 + beta * var;
        if (var < 1e-10) var = 1e-10;
    }
    return sqrt(sum_sq / count);
}

static double garch_best_rmse(const MatchedTestData* data) {
    double best = 1e10;
    double alphas[] = {0.02, 0.05, 0.08, 0.10, 0.15, 0.20};
    double betas[]  = {0.75, 0.80, 0.85, 0.90, 0.93, 0.95};
    int na = sizeof(alphas) / sizeof(alphas[0]);
    int nb = sizeof(betas)  / sizeof(betas[0]);

    double sv = 0.0;
    for (int t = 0; t < data->n_ticks; t++)
        sv += data->returns[t] * data->returns[t];
    sv /= data->n_ticks;

    for (int ia = 0; ia < na; ia++) {
        for (int ib = 0; ib < nb; ib++) {
            double a = alphas[ia], b = betas[ib];
            if (a + b >= 0.999) continue;
            double omega = sv * (1.0 - a - b);
            double r = garch_rmse(data, omega, a, b);
            if (r < best) best = r;
        }
    }
    return best;
}

// =============================================================================
// Metrics
// =============================================================================

typedef struct {
    double rmse;
    double mae;
    double bias;
    double elapsed_ms;
} FilterMetrics;

// =============================================================================
// Run BPF
// =============================================================================

static FilterMetrics run_bpf_metrics(
    const MatchedTestData* data, int n_particles, int seed
) {
    int n = data->n_ticks;
    GpuBpfState* f = gpu_bpf_create(
        n_particles,
        (float)data->dgp_rho, (float)data->dgp_sigma_z, (float)data->dgp_mu,
        (float)data->dgp_nu_state, (float)data->dgp_nu_obs, seed);

    int skip = 100;
    double sum_sq = 0.0, sum_abs = 0.0, sum_bias = 0.0;
    int count = 0;

    double t0 = get_time_us();
    for (int t = 0; t < n; t++) {
        BpfResult r = gpu_bpf_step(f, (float)data->returns[t]);
        if (t >= skip) {
            double err = (double)r.h_mean - data->true_h[t];
            sum_sq  += err * err;
            sum_abs += fabs(err);
            sum_bias += err;
            count++;
        }
    }
    double elapsed = (get_time_us() - t0) / 1000.0;

    gpu_bpf_destroy(f);

    FilterMetrics m;
    m.rmse       = sqrt(sum_sq / count);
    m.mae        = sum_abs / count;
    m.bias       = sum_bias / count;
    m.elapsed_ms = elapsed;
    return m;
}

// =============================================================================
// Test 1: Full Comparison -- KF bound vs BPF vs EWMA vs GARCH
// =============================================================================

static void test_full_comparison(int n_ticks, int n_particles, int base_seed) {
    printf("\n");
    printf("=============================================================================================\n");
    printf("  TEST 1: FULL COMPARISON -- KF bound vs BPF(%dK) vs EWMA vs GARCH\n",
           n_particles / 1000);
    printf("  %d ticks, skip 100 warmup, matched DGP (zero model mismatch)\n", n_ticks);
    printf("=============================================================================================\n");
    printf("  %-18s | %7s | %7s %7s %7s | %7s %7s\n",
           "Scenario", "KF bnd", "BPF", "Bias", "ms", "EWMA94", "GARCH");
    printf("  ------------------+---------+-------------------------+-----------------\n");

    for (int i = 0; i < N_SCENARIOS; i++) {
        MatchedTestData* data = ALL_SCENARIOS[i].gen(n_ticks, base_seed + i);

        double kf = kalman_steady_state_rmse(data->dgp_rho, data->dgp_sigma_z, data->dgp_nu_obs);
        FilterMetrics bpf = run_bpf_metrics(data, n_particles, base_seed + 100 + i);
        double ew  = ewma_rmse(data, 0.94);
        double ga  = garch_best_rmse(data);

        printf("  %-18s | %7.4f | %7.4f %+7.4f %5.0fms | %7.4f %7.4f\n",
               data->scenario_name, kf,
               bpf.rmse, bpf.bias, bpf.elapsed_ms,
               ew, ga);

        free_matched_data(data);
    }
    printf("=============================================================================================\n");
    printf("  KF bnd = linearized Kalman steady-state (theoretical floor)\n");
}

// =============================================================================
// Test 2: Particle Sweep
// =============================================================================

static void test_particle_sweep(int n_ticks, int base_seed) {
    printf("\n");
    printf("=============================================================================================\n");
    printf("  TEST 2: PARTICLE SWEEP (Student-t scenario)\n");
    printf("=============================================================================================\n");
    printf("  %10s | %8s %8s %8s\n", "Particles", "RMSE", "Bias", "ms");
    printf("  ----------+----------------------------\n");

    MatchedTestData* data = gen_student_t(n_ticks, base_seed + 1);
    double kf = kalman_steady_state_rmse(data->dgp_rho, data->dgp_sigma_z, data->dgp_nu_obs);

    int counts[] = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
    int nc = sizeof(counts) / sizeof(counts[0]);

    for (int c = 0; c < nc; c++) {
        FilterMetrics m = run_bpf_metrics(data, counts[c], base_seed + 400);
        printf("  %10d | %8.4f %+8.4f %6.1fms\n",
               counts[c], m.rmse, m.bias, m.elapsed_ms);
    }

    printf("  ----------+----------------------------\n");
    printf("  KF bound  | %8.4f\n", kf);
    printf("=============================================================================================\n");

    free_matched_data(data);
}

// =============================================================================
// Test 3: Monte Carlo Variance (multi-seed)
// =============================================================================

static void test_mc_variance(int n_ticks, int n_particles, int base_seed) {
    int n_seeds = 10;

    printf("\n");
    printf("=============================================================================================\n");
    printf("  TEST 3: MONTE CARLO VARIANCE (%d seeds x %dK particles, Student-t scenario)\n",
           n_seeds, n_particles / 1000);
    printf("=============================================================================================\n");
    printf("  %-6s | %8s %9s\n", "Seed", "RMSE", "Bias");
    printf("  ------+--------------------\n");

    MatchedTestData* data = gen_student_t(n_ticks, base_seed);

    double rmse_sum = 0, rmse_sq = 0, bias_sum = 0;

    for (int s = 0; s < n_seeds; s++) {
        int filter_seed = base_seed + 1000 + s * 137;
        FilterMetrics m = run_bpf_metrics(data, n_particles, filter_seed);

        printf("  %-6d | %8.4f %+9.4f\n", s, m.rmse, m.bias);

        rmse_sum += m.rmse;
        rmse_sq  += m.rmse * m.rmse;
        bias_sum += m.bias;
    }

    double mean = rmse_sum / n_seeds;
    double std  = sqrt(rmse_sq / n_seeds - mean * mean);

    printf("  ------+--------------------\n");
    printf("  Mean  | %8.4f %+9.4f\n", mean, bias_sum / n_seeds);
    printf("  Std   | %8.4f\n", std);
    printf("=============================================================================================\n");

    free_matched_data(data);
}

// =============================================================================
// Test 4: Throughput Benchmark
// =============================================================================

static void test_throughput(int base_seed) {
    printf("\n");
    printf("=============================================================================================\n");
    printf("  TEST 4: THROUGHPUT BENCHMARK (ticks/sec, Student-t scenario)\n");
    printf("=============================================================================================\n");
    printf("  %10s | %10s %10s\n", "Particles", "ticks/s", "us/tick");
    printf("  ----------+-----------------------\n");

    int bench_ticks = 2000;
    MatchedTestData* data = gen_student_t(bench_ticks, base_seed);

    int counts[] = {1024, 4096, 16384, 65536};
    int nc = sizeof(counts) / sizeof(counts[0]);

    for (int c = 0; c < nc; c++) {
        GpuBpfState* f = gpu_bpf_create(
            counts[c], (float)data->dgp_rho, (float)data->dgp_sigma_z,
            (float)data->dgp_mu, (float)data->dgp_nu_state,
            (float)data->dgp_nu_obs, base_seed + 900);

        // Warmup
        for (int t = 0; t < 100; t++)
            gpu_bpf_step(f, (float)data->returns[t % bench_ticks]);
        cudaDeviceSynchronize();

        double t0 = get_time_us();
        for (int t = 0; t < bench_ticks; t++)
            gpu_bpf_step(f, (float)data->returns[t]);
        cudaDeviceSynchronize();
        double elapsed_us = get_time_us() - t0;

        gpu_bpf_destroy(f);

        double tps = bench_ticks / (elapsed_us * 1e-6);
        printf("  %10d | %10.0f %8.1fus\n",
               counts[c], tps, elapsed_us / bench_ticks);
    }
    printf("=============================================================================================\n");

    free_matched_data(data);
}

// =============================================================================
// CSV Export -- per-tick filtered output
// =============================================================================

static void export_csv(
    const char* csv_path, int n_ticks, int n_particles,
    int base_seed, int csv_scenario
) {
    FILE* fp = fopen(csv_path, "w");
    if (!fp) {
#ifdef _WIN32
        char dir[512];
        strncpy(dir, csv_path, sizeof(dir) - 1);
        dir[sizeof(dir) - 1] = '\0';
        char* last_sep = strrchr(dir, '\\');
        if (!last_sep) last_sep = strrchr(dir, '/');
        if (last_sep) {
            *last_sep = '\0';
            CreateDirectoryA(dir, NULL);
        }
        fp = fopen(csv_path, "w");
#else
        char dir[512];
        strncpy(dir, csv_path, sizeof(dir) - 1);
        dir[sizeof(dir) - 1] = '\0';
        char* last_sep = strrchr(dir, '/');
        if (last_sep) {
            *last_sep = '\0';
            char cmd[600];
            snprintf(cmd, sizeof(cmd), "mkdir -p %s", dir);
            system(cmd);
        }
        fp = fopen(csv_path, "w");
#endif
    }
    if (!fp) {
        fprintf(stderr, "ERROR: Cannot open %s for writing\n", csv_path);
        return;
    }

    fprintf(fp, "tick,scenario_id,scenario_name,"
                "true_h,true_vol,return,"
                "bpf_h,bpf_loglik,"
                "ewma_h,garch_h\n");

    for (int si = 0; si < N_SCENARIOS; si++) {
        MatchedTestData* data = ALL_SCENARIOS[si].gen(n_ticks, base_seed);

        if (csv_scenario > 0 && data->scenario_id != csv_scenario) {
            free_matched_data(data);
            continue;
        }

        printf("  CSV: exporting scenario %d/%d [%s] ...\n",
               data->scenario_id, N_SCENARIOS, data->scenario_name);

        GpuBpfState* bpf = gpu_bpf_create(
            n_particles,
            (float)data->dgp_rho, (float)data->dgp_sigma_z, (float)data->dgp_mu,
            (float)data->dgp_nu_state, (float)data->dgp_nu_obs, base_seed);

        // EWMA init
        double ewma_var = 0.0;
        {
            int init_n = (n_ticks < 20) ? n_ticks : 20;
            for (int t = 0; t < init_n; t++)
                ewma_var += data->returns[t] * data->returns[t];
            ewma_var /= init_n;
            if (ewma_var < 1e-10) ewma_var = 1e-10;
        }

        // GARCH init
        double garch_omega = 0.0, garch_alpha = 0.05, garch_beta = 0.90;
        {
            double best_rmse = 1e10;
            double alphas[] = {0.02, 0.05, 0.08, 0.10, 0.15, 0.20};
            double betas[]  = {0.75, 0.80, 0.85, 0.90, 0.93, 0.95};
            int na = sizeof(alphas) / sizeof(alphas[0]);
            int nb = sizeof(betas)  / sizeof(betas[0]);
            double sv = 0.0;
            for (int t = 0; t < n_ticks; t++)
                sv += data->returns[t] * data->returns[t];
            sv /= n_ticks;
            for (int ia = 0; ia < na; ia++) {
                for (int ib = 0; ib < nb; ib++) {
                    double a = alphas[ia], b = betas[ib];
                    if (a + b >= 0.999) continue;
                    double omega = sv * (1.0 - a - b);
                    double r = garch_rmse(data, omega, a, b);
                    if (r < best_rmse) {
                        best_rmse    = r;
                        garch_omega  = omega;
                        garch_alpha  = a;
                        garch_beta   = b;
                    }
                }
            }
        }
        double garch_var = (garch_alpha + garch_beta < 0.999)
            ? garch_omega / (1.0 - garch_alpha - garch_beta) : 0.01;
        if (garch_var < 1e-10) garch_var = 1e-10;

        // Tick loop
        for (int t = 0; t < n_ticks; t++) {
            float obs = (float)data->returns[t];
            BpfResult br = gpu_bpf_step(bpf, obs);

            double ewma_h = log(ewma_var);
            double y2 = (double)obs * (double)obs;
            ewma_var = 0.94 * ewma_var + 0.06 * y2;
            if (ewma_var < 1e-10) ewma_var = 1e-10;

            double garch_h = log(garch_var);
            garch_var = garch_omega + garch_alpha * y2 + garch_beta * garch_var;
            if (garch_var < 1e-10) garch_var = 1e-10;

            fprintf(fp, "%d,%d,\"%s\","
                        "%.8f,%.8f,%.8f,"
                        "%.8f,%.6f,"
                        "%.8f,%.8f\n",
                    t, data->scenario_id, data->scenario_name,
                    data->true_h[t], data->true_vol[t], data->returns[t],
                    (double)br.h_mean, (double)br.log_lik,
                    ewma_h, garch_h);
        }

        gpu_bpf_destroy(bpf);
        free_matched_data(data);
    }

    fclose(fp);
    printf("  CSV written: %s\n", csv_path);
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    int n_ticks       = 5000;
    int n_particles   = 50000;
    int mc_particles  = 10000;
    int base_seed     = 42;
    const char* csv_path = NULL;
    int csv_scenario     = 0;
    int csv_only         = 0;

    for (int i = 1; i < argc; i++) {
        if      (strcmp(argv[i], "--ticks") == 0 && i+1 < argc)
            n_ticks = atoi(argv[++i]);
        else if (strcmp(argv[i], "--particles") == 0 && i+1 < argc)
            n_particles = atoi(argv[++i]);
        else if (strcmp(argv[i], "--mc-particles") == 0 && i+1 < argc)
            mc_particles = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i+1 < argc)
            base_seed = atoi(argv[++i]);
        else if (strcmp(argv[i], "--csv") == 0 && i+1 < argc)
            csv_path = argv[++i];
        else if (strcmp(argv[i], "--csv-scenario") == 0 && i+1 < argc)
            csv_scenario = atoi(argv[++i]);
        else if (strcmp(argv[i], "--csv-only") == 0)
            csv_only = 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("=============================================================================================\n");
    printf("  GPU BPF -- MATCHED-DGP TEST SUITE (PTX)\n");
    printf("  Device: %s (SM %d.%d, %d SMs)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("  Zero model mismatch -- all filters use TRUE DGP parameters\n");
    printf("=============================================================================================\n");
    printf("  Config: %d ticks, %dK particles, MC=%dK\n",
           n_ticks, n_particles / 1000, mc_particles / 1000);
    if (csv_path)
        printf("  CSV output: %s (scenario=%s)\n",
               csv_path, csv_scenario ? "filtered" : "all");

    if (csv_path) {
        export_csv(csv_path, n_ticks, n_particles, base_seed, csv_scenario);
        if (csv_only) {
            printf("\n--csv-only: skipping test suite.\n");
            return 0;
        }
    }

    test_full_comparison(n_ticks, n_particles, base_seed);
    test_particle_sweep(n_ticks, base_seed);
    test_mc_variance(n_ticks, mc_particles, base_seed);
    test_throughput(base_seed);

    printf("\nAll tests complete.\n");
    return 0;
}
