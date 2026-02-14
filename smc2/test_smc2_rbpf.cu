/**
 * @file test_smc2_rbpf.cu
 * @brief Test suite for SMC² + RBPF 4-parameter learner
 *
 * Self-contained DGP matching the RBPF generative model exactly.
 * Learned: ρ, σ_z, μ_base, σ_base  (4 params)
 * Fixed:   μ_scale, μ_rate, σ_scale, σ_rate, θ(z) curve
 * Observations: log(y²) for OCSN likelihood.
 *
 * Build:
 *   nvcc -O2 -arch=sm_120 -o test_smc2_rbpf test_smc2_rbpf.cu smc2_rbpf_batch.cu \
 *        -lcurand --expt-relaxed-constexpr
 */

#include "smc2_rbpf_batch.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>

/*═══════════════════════════════════════════════════════════════════════════
 * Particle counts — change these two to tune all tests at once
 *═══════════════════════════════════════════════════════════════════════════*/

#ifndef N_THETA
#define N_THETA 1024
#endif

#ifndef N_INNER
#define N_INNER 512
#endif

/*═══════════════════════════════════════════════════════════════════════════
 * HOST RNG (xoshiro256**)
 *═══════════════════════════════════════════════════════════════════════════*/

static uint64_t s_rng[4];
static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static uint64_t xoshiro_next(void) {
    uint64_t r = rotl(s_rng[1] * 5, 7) * 9;
    uint64_t t = s_rng[1] << 17;
    s_rng[2] ^= s_rng[0]; s_rng[3] ^= s_rng[1];
    s_rng[1] ^= s_rng[2]; s_rng[0] ^= s_rng[3];
    s_rng[2] ^= t; s_rng[3] = rotl(s_rng[3], 45);
    return r;
}

static void seed_rng(uint64_t seed) {
    for (int i = 0; i < 4; i++) {
        seed += 0x9E3779B97F4A7C15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        s_rng[i] = z ^ (z >> 31);
    }
}

static double uniform01(void) {
    return (double)(xoshiro_next() >> 11) * (1.0 / 9007199254740992.0);
}

static double randn(void) {
    double u1 = uniform01(), u2 = uniform01();
    while (u1 < 1e-15) u1 = uniform01();
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

/*═══════════════════════════════════════════════════════════════════════════
 * DGP: Regime-Switching SV (matches RBPF model exactly)
 *═══════════════════════════════════════════════════════════════════════════*/

struct RSVParams {
    double rho, sigma_z, mu_base, sigma_base;           /* learned */
    double mu_scale, mu_rate, sigma_scale, sigma_rate;   /* fixed shapes */
    double theta_base, theta_scale, theta_rate;          /* fixed θ(z) */
};

static double eval_curve_h(double base, double scale, double rate, double z) {
    return base + scale * (1.0 - exp(-rate * z));
}

static double zt_to_z(double zt) { return 1.5 * (1.0 + tanh(zt)); }

static RSVParams make_truth(double rho, double sigma_z, double mu_base, double sigma_base) {
    RSVParams p;
    p.rho = rho; p.sigma_z = sigma_z; p.mu_base = mu_base; p.sigma_base = sigma_base;
    p.mu_scale = 0.5;    p.mu_rate = 1.0;      /* must match SMC² defaults */
    p.sigma_scale = 0.1; p.sigma_rate = 1.0;
    p.theta_base = 0.02; p.theta_scale = 0.08; p.theta_rate = 1.5;
    return p;
}

/**
 * @brief Simulate from regime-switching SV and return log(y²) for OCSN
 */
static void simulate_rsv(const RSVParams* p, float* y_obs, float* h_true,
                          float* z_true, int T) {
    double omr2 = fmax(1.0 - p->rho * p->rho, 1e-6);
    double zt_std = p->sigma_z / sqrt(omr2);
    double zt = zt_std * randn();
    double z = zt_to_z(zt);
    double th = eval_curve_h(p->theta_base, p->theta_scale, p->theta_rate, z);
    double mu = eval_curve_h(p->mu_base, p->mu_scale, p->mu_rate, z);
    double sh = eval_curve_h(p->sigma_base, p->sigma_scale, p->sigma_rate, z);
    double phi = 1.0 - th;
    double hv = (sh * sh) / fmax(1.0 - phi * phi, 1e-6);
    double h = mu + sqrt(hv) * randn();

    for (int t = 0; t < T; t++) {
        zt = p->rho * zt + p->sigma_z * randn();
        z = zt_to_z(zt);
        z_true[t] = (float)z;
        th = eval_curve_h(p->theta_base, p->theta_scale, p->theta_rate, z);
        mu = eval_curve_h(p->mu_base, p->mu_scale, p->mu_rate, z);
        sh = eval_curve_h(p->sigma_base, p->sigma_scale, p->sigma_rate, z);
        phi = 1.0 - th;
        h = phi * h + th * mu + sh * randn();
        h_true[t] = (float)h;
        double eps = randn();
        double y_raw = exp(h / 2.0) * eps;
        double ysq = y_raw * y_raw;
        if (ysq < 1e-30) ysq = 1e-30;
        y_obs[t] = (float)log(ysq);
    }
}

/*═══════════════════════════════════════════════════════════════════════════
 * Helpers
 *═══════════════════════════════════════════════════════════════════════════*/

static const char* param_names[N_PARAMS] = {"rho", "sigma_z", "mu_base", "sigma_base"};

static void get_true_arr(const RSVParams* p, float* out) {
    out[0] = (float)p->rho;      out[1] = (float)p->sigma_z;
    out[2] = (float)p->mu_base;  out[3] = (float)p->sigma_base;
}

static void print_data_stats(const float* y, int T) {
    float sum = 0.0f, sum_sq = 0.0f, y_min = y[0], y_max = y[0];
    for (int t = 0; t < T; t++) {
        sum += y[t]; sum_sq += y[t] * y[t];
        if (y[t] < y_min) y_min = y[t];
        if (y[t] > y_max) y_max = y[t];
    }
    float mean = sum / T;
    float std_dev = sqrtf(sum_sq / T - mean * mean);
    printf("  Data stats: mean=%.3f, std=%.3f, min=%.3f, max=%.3f\n", mean, std_dev, y_min, y_max);
    printf("  Expected log χ²(1): mean≈-1.27, var≈4.93\n");
}

static void print_recovery_table(const float* truth, const float* mean,
                                  const float* std_dev, int* n_ok, int* n_15pct) {
    printf("\n%-12s  %8s  %8s  %8s  %7s  %7s  %s\n",
           "Parameter", "True", "Est", "Std", "Err%", "z-score", "Status");
    printf("─────────────────────────────────────────────────────────────────────────\n");
    
    *n_ok = 0;
    *n_15pct = 0;
    for (int i = 0; i < N_PARAMS; i++) {
        float err = mean[i] - truth[i];
        float z = fabsf(err) / fmaxf(std_dev[i], 1e-6f);
        float pct = (fabsf(truth[i]) > 0.01f) ? 100.0f * err / truth[i] : err * 100.0f;
        const char* tag = (z <= 2.0f) ? "OK" : (z <= 3.0f) ? "WARN" : "MISS";
        if (z <= 2.0f) (*n_ok)++;
        if (fabsf(pct) <= 15.0f) (*n_15pct)++;
        printf("%-12s  %8.4f  %8.4f  %8.4f  %+6.1f%%  %7.2f  [%s]\n",
               param_names[i], truth[i], mean[i], std_dev[i], pct, z, tag);
    }
    printf("─────────────────────────────────────────────────────────────────────────\n");
}

/*═══════════════════════════════════════════════════════════════════════════
 * Test: Basic CUDA Smoke Test
 *═══════════════════════════════════════════════════════════════════════════*/

void test_basic(void) {
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("Test: Basic CUDA Operations\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    printf("  CUDA devices: %d\n", dev_count);
    
    if (dev_count == 0) { printf("  ERROR: No CUDA devices!\n"); return; }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("  Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("  Shared memory: %zu KB\n", prop.sharedMemPerBlock / 1024);
    
    SMC2StateCUDA* state = smc2_cuda_alloc(N_THETA, N_INNER);
    printf("  Allocated N_theta=%d, N_inner=%d\n", state->N_theta, state->N_inner);
    
    smc2_cuda_init_from_prior(state);
    printf("  Initialized from prior.\n");
    
    float theta_mean[N_PARAMS], theta_std[N_PARAMS];
    smc2_cuda_get_theta_mean(state, theta_mean);
    smc2_cuda_get_theta_std(state, theta_std);
    
    printf("  Prior samples:\n");
    printf("    rho        = %.4f ± %.4f\n", theta_mean[0], theta_std[0]);
    printf("    sigma_z    = %.4f ± %.4f\n", theta_mean[1], theta_std[1]);
    printf("    mu_base    = %.4f ± %.4f\n", theta_mean[2], theta_std[2]);
    printf("    sigma_base = %.4f ± %.4f\n", theta_mean[3], theta_std[3]);
    
    smc2_cuda_free(state);
    printf("  PASSED\n");
}

/*═══════════════════════════════════════════════════════════════════════════
 * Test: Prior-Data Agreement
 *═══════════════════════════════════════════════════════════════════════════*/

void test_prior_data_agreement(void) {
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("Test: Prior-Data Agreement\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    
    SMC2StateCUDA* state = smc2_cuda_alloc(N_THETA, N_INNER);
    
    printf("\nDefault prior (4 learned params):\n");
    printf("  rho        = %.3f ± %.3f\n", state->prior.rho_mean, state->prior.rho_std);
    printf("  sigma_z    = %.3f ± %.3f\n", state->prior.sigma_z_mean, state->prior.sigma_z_std);
    printf("  mu_base    = %.3f ± %.3f\n", state->prior.mu_base_mean, state->prior.mu_base_std);
    printf("  sigma_base = %.3f ± %.3f\n", state->prior.sigma_base_mean, state->prior.sigma_base_std);
    
    printf("\nDefault bounds:\n");
    printf("  rho        ∈ [%.3f, %.3f]\n", state->bounds.rho_min, state->bounds.rho_max);
    printf("  sigma_z    ∈ [%.3f, %.3f]\n", state->bounds.sigma_z_min, state->bounds.sigma_z_max);
    printf("  mu_base    ∈ [%.3f, %.3f]\n", state->bounds.mu_base_min, state->bounds.mu_base_max);
    printf("  sigma_base ∈ [%.3f, %.3f]\n", state->bounds.sigma_base_min, state->bounds.sigma_base_max);
    
    printf("\nFixed curve shapes (constant memory):\n");
    printf("  mu_scale=%.2f  mu_rate=%.2f  sigma_scale=%.2f  sigma_rate=%.2f\n",
           state->fixed_curves.mu_scale, state->fixed_curves.mu_rate,
           state->fixed_curves.sigma_scale, state->fixed_curves.sigma_rate);
    
    printf("\nTest truth values:\n");
    printf("  rho=0.95, sigma_z=0.10, mu_base=-1.0, sigma_base=0.15\n");
    printf("  → Check that truth is within bounds and near prior means!\n");
    
    smc2_cuda_free(state);
}

/*═══════════════════════════════════════════════════════════════════════════
 * Test: Parameter Learning with CPMMH
 *═══════════════════════════════════════════════════════════════════════════*/

void test_parameter_learning(void) {
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("Test: Parameter Learning with CPMMH Rejuvenation\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    
    RSVParams truth = make_truth(0.95, 0.10, -1.0, 0.15);
    int T = 1200;
    
    float* y = (float*)malloc(T * sizeof(float));
    float* h = (float*)malloc(T * sizeof(float));
    float* z = (float*)malloc(T * sizeof(float));
    simulate_rsv(&truth, y, h, z, T);
    
    printf("\nTRUE PARAMETERS (4 learned):\n");
    printf("  rho=%.3f, sigma_z=%.3f, mu_base=%.3f, sigma_base=%.3f\n",
           truth.rho, truth.sigma_z, truth.mu_base, truth.sigma_base);
    printf("  Fixed shapes: mu_scale=%.2f, mu_rate=%.2f, sigma_scale=%.2f, sigma_rate=%.2f\n",
           truth.mu_scale, truth.mu_rate, truth.sigma_scale, truth.sigma_rate);
    
    printf("\nGenerated T=%d observations\n", T);
    print_data_stats(y, T);
    
    /* h_true stats */
    float h_sum = 0.0f, h_sq = 0.0f;
    for (int t = 0; t < T; t++) { h_sum += h[t]; h_sq += h[t] * h[t]; }
    printf("  True h: mean=%.3f, std=%.3f\n", h_sum / T, sqrtf(h_sq / T - (h_sum / T) * (h_sum / T)));
    
    SMC2StateCUDA* state = smc2_cuda_alloc(N_THETA, N_INNER);
    smc2_cuda_set_seed(state, 12345);
    smc2_cuda_set_noise_capacity(state, T + 128);
    smc2_cuda_set_fixed_lag(state, 200);
    smc2_cuda_init_from_prior(state);
    
    printf("\nRunning SMC² (N_theta=%d, N_inner=%d, K_rejuv=%d, L=200)...\n",
           state->N_theta, state->N_inner, state->K_rejuv);
    
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    
    cudaEventRecord(ev0);
    for (int t = 0; t < T; t++) {
        float ess = smc2_cuda_update(state, y[t]);
        if ((t + 1) % 200 == 0 || t == T - 1) {
            float m[N_PARAMS];
            smc2_cuda_get_theta_mean(state, m);
            printf("  t=%4d: ESS=%5.1f, resamp=%2d, accept=%5.1f%%, "
                   "ρ=%.3f, σz=%.3f, μb=%.2f, σb=%.3f\n",
                   t + 1, ess, state->n_resamples,
                   state->n_rejuv_total > 0 ?
                   100.0f * state->n_rejuv_accepts / state->n_rejuv_total : 0.0f,
                   m[0], m[1], m[2], m[3]);
        }
    }
    cudaEventRecord(ev1); cudaEventSynchronize(ev1);
    float ms; cudaEventElapsedTime(&ms, ev0, ev1);
    
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("RESULTS\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Time: %.1f ms (%.2f ms/obs)\n", ms, ms / T);
    printf("  Resamples: %d\n", state->n_resamples);
    printf("  Rejuvenation: %d/%d accepted (%.1f%%)\n",
           state->n_rejuv_accepts, state->n_rejuv_total,
           state->n_rejuv_total > 0 ?
           100.0f * state->n_rejuv_accepts / state->n_rejuv_total : 0.0f);
    
    float mean[N_PARAMS], sd[N_PARAMS], tv[N_PARAMS];
    smc2_cuda_get_theta_mean(state, mean);
    smc2_cuda_get_theta_std(state, sd);
    get_true_arr(&truth, tv);
    
    printf("\nESTIMATED (mean ± std):\n");
    for (int i = 0; i < N_PARAMS; i++)
        printf("  %-12s = %.4f ± %.4f\n", param_names[i], mean[i], sd[i]);
    
    int n_ok, n_15;
    print_recovery_table(tv, mean, sd, &n_ok, &n_15);
    printf("OVERALL: %d/%d within 2σ, %d/%d within 15%% relative error\n",
           n_ok, N_PARAMS, n_15, N_PARAMS);
    printf("%s\n", n_ok >= 3 ? "PASSED" : "NEEDS INVESTIGATION");
    
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    smc2_cuda_free(state);
    free(y); free(h); free(z);
}

/*═══════════════════════════════════════════════════════════════════════════
 * Test: Fixed-Lag PMMH for Long Sequences
 *═══════════════════════════════════════════════════════════════════════════*/

void test_fixed_lag(void) {
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("Test: Fixed-Lag PMMH — Accuracy at Large T\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("\nGoal: Show that fixed-lag maintains accuracy at large T where\n");
    printf("      full-history PMMH degrades due to O(T) variance growth.\n\n");
    
    RSVParams truth = make_truth(0.95, 0.10, -1.0, 0.15);
    int T_values[] = {1000, 2000, 5000};
    int n_T = 3;
    
    for (int ti = 0; ti < n_T; ti++) {
        int T = T_values[ti];
        
        printf("─────────────────────────────────────────────────────────────────────────\n");
        printf("T = %d\n", T);
        printf("─────────────────────────────────────────────────────────────────────────\n");
        
        float* y = (float*)malloc(T * sizeof(float));
        float* h = (float*)malloc(T * sizeof(float));
        float* z = (float*)malloc(T * sizeof(float));
        simulate_rsv(&truth, y, h, z, T);
        
        printf("  %-6s  %8s  %10s  %10s  %10s  %10s  %8s  %5s\n",
               "Lag", "Time(ms)", "rho", "sigma_z", "mu_base", "sigma_base", "Accept%", "Resmp");
        
        int lags[] = {0, 200};
        for (int li = 0; li < 2; li++) {
            int L = lags[li];
            
            SMC2StateCUDA* s = smc2_cuda_alloc(N_THETA, N_INNER);
            smc2_cuda_set_seed(s, 12345);
            smc2_cuda_set_noise_capacity(s, T + 128);
            smc2_cuda_set_fixed_lag(s, L);
            smc2_cuda_init_from_prior(s);
            
            cudaEvent_t ev0, ev1;
            cudaEventCreate(&ev0); cudaEventCreate(&ev1);
            cudaEventRecord(ev0);
            for (int t = 0; t < T; t++) smc2_cuda_update(s, y[t]);
            cudaEventRecord(ev1); cudaEventSynchronize(ev1);
            float ms; cudaEventElapsedTime(&ms, ev0, ev1);
            
            float mean[N_PARAMS], sd[N_PARAMS];
            smc2_cuda_get_theta_mean(s, mean);
            smc2_cuda_get_theta_std(s, sd);
            float acc = s->n_rejuv_total > 0 ?
                100.0f * s->n_rejuv_accepts / s->n_rejuv_total : 0.0f;
            
            printf("  L=%3d   %8.1f  %5.4f±%4.3f  %5.4f±%4.3f  %5.2f±%4.2f  %5.4f±%4.3f  %7.1f%%  %5d\n",
                   L, ms,
                   mean[0], sd[0], mean[1], sd[1],
                   mean[2], sd[2], mean[3], sd[3],
                   acc, s->n_resamples);
            
            cudaEventDestroy(ev0); cudaEventDestroy(ev1);
            smc2_cuda_free(s);
        }
        
        free(y); free(h); free(z);
        printf("\n");
    }
    
    printf("─────────────────────────────────────────────────────────────────────────\n");
    printf("True: ρ=%.2f  σ_z=%.2f  μ_base=%.1f  σ_base=%.2f\n",
           truth.rho, truth.sigma_z, truth.mu_base, truth.sigma_base);
    printf("Expected: At large T, L=0 degrades while L=200 stays stable.\n");
}

/*═══════════════════════════════════════════════════════════════════════════
 * Test: Throughput
 *═══════════════════════════════════════════════════════════════════════════*/

void test_throughput(void) {
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("Test: Throughput with CPMMH Rejuvenation\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    
    RSVParams truth = make_truth(0.95, 0.10, -1.0, 0.15);
    int T = 500;
    
    float* y = (float*)malloc(T * sizeof(float));
    float* h = (float*)malloc(T * sizeof(float));
    float* z = (float*)malloc(T * sizeof(float));
    simulate_rsv(&truth, y, h, z, T);
    
    printf("\n  N_theta  N_inner  Time(ms)  Resamples  Accept%%   ms/obs\n");
    printf("  ─────────────────────────────────────────────────────────\n");
    
    int configs[][2] = {{N_THETA/4, N_INNER/4}, {N_THETA/2, N_INNER/2}, {N_THETA, N_INNER}, {N_THETA, N_INNER*2}};
    int n_configs = 4;
    
    for (int c = 0; c < n_configs; c++) {
        SMC2StateCUDA* s = smc2_cuda_alloc(configs[c][0], configs[c][1]);
        smc2_cuda_set_seed(s, 54321);
        smc2_cuda_set_noise_capacity(s, T + 128);
        smc2_cuda_set_fixed_lag(s, 200);
        smc2_cuda_init_from_prior(s);
        
        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0); cudaEventCreate(&ev1);
        cudaEventRecord(ev0);
        for (int t = 0; t < T; t++) smc2_cuda_update(s, y[t]);
        cudaEventRecord(ev1); cudaEventSynchronize(ev1);
        float ms; cudaEventElapsedTime(&ms, ev0, ev1);
        
        float acc = s->n_rejuv_total > 0 ?
            100.0f * s->n_rejuv_accepts / s->n_rejuv_total : 0.0f;
        
        printf("  %4d     %4d     %7.1f   %4d       %5.1f    %.3f\n",
               configs[c][0], configs[c][1], ms, s->n_resamples, acc, ms / T);
        
        cudaEventDestroy(ev0); cudaEventDestroy(ev1);
        smc2_cuda_free(s);
    }
    
    free(y); free(h); free(z);
}

/*═══════════════════════════════════════════════════════════════════════════
 * Test: High Volatility Regime
 *═══════════════════════════════════════════════════════════════════════════*/

void test_high_vol(void) {
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("Test: High Volatility Regime (μ_base=2.0, ρ=0.98)\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    
    RSVParams truth = make_truth(0.98, 0.08, 2.0, 0.20);
    int T = 1000;
    
    float* y = (float*)malloc(T * sizeof(float));
    float* h = (float*)malloc(T * sizeof(float));
    float* z = (float*)malloc(T * sizeof(float));
    simulate_rsv(&truth, y, h, z, T);
    
    printf("\nTRUE: ρ=%.2f  σ_z=%.2f  μ_base=%.1f  σ_base=%.2f\n",
           truth.rho, truth.sigma_z, truth.mu_base, truth.sigma_base);
    print_data_stats(y, T);
    
    SMC2StateCUDA* s = smc2_cuda_alloc(N_THETA, N_INNER);
    /* Widen prior for high-vol regime */
    s->prior.rho_mean = 0.96f;        s->prior.rho_std = 0.02f;
    s->prior.sigma_z_mean = 0.10f;    s->prior.sigma_z_std = 0.10f;
    s->prior.mu_base_mean = 0.0f;     s->prior.mu_base_std = 2.0f;
    s->prior.sigma_base_mean = 0.15f; s->prior.sigma_base_std = 0.10f;
    s->bounds.mu_base_min = -5.0f;    s->bounds.mu_base_max = 8.0f;
    smc2_cuda_set_fixed_lag(s, 200);
    smc2_cuda_set_seed(s, 999);
    smc2_cuda_init_from_prior(s);
    
    printf("\nRunning SMC² (N_theta=%d, N_inner=%d, K=%d, L=200)...\n",
           s->N_theta, s->N_inner, s->K_rejuv);
    
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    cudaEventRecord(ev0);
    for (int t = 0; t < T; t++) {
        float ess = smc2_cuda_update(s, y[t]);
        if ((t + 1) % 200 == 0 || t == T - 1) {
            float m[N_PARAMS];
            smc2_cuda_get_theta_mean(s, m);
            printf("  t=%4d: ESS=%5.1f, resamp=%2d, accept=%5.1f%%, "
                   "ρ=%.3f, σz=%.3f, μb=%.2f, σb=%.3f\n",
                   t + 1, ess, s->n_resamples,
                   s->n_rejuv_total > 0 ?
                   100.0f * s->n_rejuv_accepts / s->n_rejuv_total : 0.0f,
                   m[0], m[1], m[2], m[3]);
        }
    }
    cudaEventRecord(ev1); cudaEventSynchronize(ev1);
    float ms; cudaEventElapsedTime(&ms, ev0, ev1);
    
    printf("\n  Time: %.1f ms  Resamples: %d  Accept: %.1f%%\n",
           ms, s->n_resamples,
           s->n_rejuv_total > 0 ?
           100.0f * s->n_rejuv_accepts / s->n_rejuv_total : 0.0f);
    
    float mean[N_PARAMS], sd[N_PARAMS], tv[N_PARAMS];
    smc2_cuda_get_theta_mean(s, mean);
    smc2_cuda_get_theta_std(s, sd);
    get_true_arr(&truth, tv);
    
    int n_ok, n_15;
    print_recovery_table(tv, mean, sd, &n_ok, &n_15);
    printf("OVERALL: %d/%d within 2σ → %s\n", n_ok, N_PARAMS,
           n_ok >= 2 ? "PASSED" : "NEEDS INVESTIGATION");
    
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    smc2_cuda_free(s);
    free(y); free(h); free(z);
}

/*═══════════════════════════════════════════════════════════════════════════
 * Test: ρ vs σ_z Identifiability
 *═══════════════════════════════════════════════════════════════════════════*/

void test_identifiability(void) {
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("Test: ρ vs σ_z Identifiability\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    
    RSVParams a = make_truth(0.98, 0.05, -1.0, 0.15);
    RSVParams b = make_truth(0.85, 0.20, -1.0, 0.15);
    
    printf("\n  Case A: ρ=%.2f  σ_z=%.2f  (persistent, quiet)\n", a.rho, a.sigma_z);
    printf("  Case B: ρ=%.2f  σ_z=%.2f  (fast-switching, noisy)\n\n", b.rho, b.sigma_z);
    
    int T = 1500;
    float* y = (float*)malloc(T * sizeof(float));
    float* h = (float*)malloc(T * sizeof(float));
    float* z = (float*)malloc(T * sizeof(float));
    
    RSVParams cases[2] = {a, b};
    float learned_rho[2], learned_sz[2];
    
    for (int c = 0; c < 2; c++) {
        simulate_rsv(&cases[c], y, h, z, T);
        
        SMC2StateCUDA* s = smc2_cuda_alloc(N_THETA, N_INNER);
        s->prior.rho_mean = 0.90f;      s->prior.rho_std = 0.05f;
        s->prior.sigma_z_mean = 0.10f;  s->prior.sigma_z_std = 0.10f;
        s->bounds.rho_min = 0.70f;
        smc2_cuda_set_fixed_lag(s, 200);
        smc2_cuda_set_seed(s, 42 + c);
        
        smc2_cuda_init_from_prior(s);
        for (int t = 0; t < T; t++) smc2_cuda_update(s, y[t]);
        
        float mean[N_PARAMS];
        smc2_cuda_get_theta_mean(s, mean);
        learned_rho[c] = mean[0];
        learned_sz[c]  = mean[1];
        
        printf("  Case %c: learned ρ=%.4f (true %.2f)  σ_z=%.4f (true %.2f)\n",
               'A' + c, mean[0], (float)cases[c].rho, mean[1], (float)cases[c].sigma_z);
        smc2_cuda_free(s);
    }
    
    int rho_ok = (learned_rho[0] > learned_rho[1]);
    int sz_ok  = (learned_sz[0]  < learned_sz[1]);
    int pass = rho_ok && sz_ok;
    
    printf("\n  ρ_A > ρ_B?     %s  (%.4f vs %.4f)\n", rho_ok ? "YES" : "NO ", learned_rho[0], learned_rho[1]);
    printf("  σ_z_A < σ_z_B? %s  (%.4f vs %.4f)\n", sz_ok  ? "YES" : "NO ", learned_sz[0], learned_sz[1]);
    printf("  → %s\n", pass ? "PASSED" : "FAILED");
    
    free(y); free(h); free(z);
}

/*═══════════════════════════════════════════════════════════════════════════
 * Test: Per-Tick Latency Distribution
 *═══════════════════════════════════════════════════════════════════════════*/

void test_latency(void) {
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("Test: Per-Tick Latency Distribution\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    
    RSVParams truth = make_truth(0.95, 0.10, -1.0, 0.15);
    int T = 1000;
    
    float* y = (float*)malloc(T * sizeof(float));
    float* h = (float*)malloc(T * sizeof(float));
    float* z = (float*)malloc(T * sizeof(float));
    simulate_rsv(&truth, y, h, z, T);
    
    int inners[] = {64, 128, 256, 512};
    
    printf("\n  N_theta=%d  L=200  T=%d\n\n", N_THETA, T);
    printf("  %-8s  %8s  %8s  %6s  %7s\n", "N_inner", "Wall ms", "μs/tick", "Resamp", "Accept");
    printf("  ─────────────────────────────────────────────────\n");
    
    for (int ci = 0; ci < 4; ci++) {
        int ni = inners[ci];
        SMC2StateCUDA* s = smc2_cuda_alloc(N_THETA, ni);
        smc2_cuda_set_fixed_lag(s, 200);
        smc2_cuda_set_seed(s, 42);
        
        /* Warmup */
        smc2_cuda_init_from_prior(s);
        for (int t = 0; t < 50; t++) smc2_cuda_update(s, y[t]);
        
        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0); cudaEventCreate(&ev1);
        cudaEventRecord(ev0);
        smc2_cuda_init_from_prior(s);
        for (int t = 0; t < T; t++) smc2_cuda_update(s, y[t]);
        cudaEventRecord(ev1); cudaEventSynchronize(ev1);
        float ms; cudaEventElapsedTime(&ms, ev0, ev1);
        
        float acc = s->n_rejuv_total > 0 ?
            100.0f * s->n_rejuv_accepts / s->n_rejuv_total : 0.0f;
        
        printf("  %-8d  %8.1f  %8.1f  %6d  %6.1f%%\n",
               ni, ms, 1000.0f * ms / T, s->n_resamples, acc);
        
        cudaEventDestroy(ev0); cudaEventDestroy(ev1);
        smc2_cuda_free(s);
    }
    
    /* Per-tick percentiles at default N_INNER */
    printf("\n  Per-tick distribution (N_theta=%d, N_inner=%d):\n", N_THETA, N_INNER);
    {
        int T_b = 500;
        SMC2StateCUDA* s = smc2_cuda_alloc(N_THETA, N_INNER);
        smc2_cuda_set_fixed_lag(s, 200);
        smc2_cuda_set_seed(s, 42);
        
        float* us = (float*)malloc(T_b * sizeof(float));
        cudaEvent_t* e0 = (cudaEvent_t*)malloc(T_b * sizeof(cudaEvent_t));
        cudaEvent_t* e1 = (cudaEvent_t*)malloc(T_b * sizeof(cudaEvent_t));
        for (int t = 0; t < T_b; t++) { cudaEventCreate(&e0[t]); cudaEventCreate(&e1[t]); }
        
        smc2_cuda_init_from_prior(s);
        for (int t = 0; t < T_b; t++) {
            cudaEventRecord(e0[t]);
            smc2_cuda_update(s, y[t]);
            cudaEventRecord(e1[t]);
        }
        cudaDeviceSynchronize();
        
        for (int t = 0; t < T_b; t++) {
            float ms; cudaEventElapsedTime(&ms, e0[t], e1[t]);
            us[t] = ms * 1000.0f;
        }
        /* Sort */
        for (int i = 0; i < T_b - 1; i++)
            for (int j = i + 1; j < T_b; j++)
                if (us[j] < us[i]) { float tmp = us[i]; us[i] = us[j]; us[j] = tmp; }
        
        float sum = 0; for (int t = 0; t < T_b; t++) sum += us[t];
        int n_spike = 0; float thresh = 3.0f * us[T_b / 2];
        for (int t = 0; t < T_b; t++) if (us[t] > thresh) n_spike++;
        
        printf("    Min:  %8.1f μs\n", us[0]);
        printf("    P50:  %8.1f μs  (forward-only baseline)\n", us[T_b / 2]);
        printf("    Mean: %8.1f μs\n", sum / T_b);
        printf("    P90:  %8.1f μs\n", us[(int)(T_b * 0.90)]);
        printf("    P95:  %8.1f μs\n", us[(int)(T_b * 0.95)]);
        printf("    P99:  %8.1f μs  (resample + rejuvenation spikes)\n", us[(int)(T_b * 0.99)]);
        printf("    Max:  %8.1f μs\n", us[T_b - 1]);
        printf("    Spikes: %d / %d ticks > %.0f μs (3× median)\n", n_spike, T_b, thresh);
        
        for (int t = 0; t < T_b; t++) { cudaEventDestroy(e0[t]); cudaEventDestroy(e1[t]); }
        free(e0); free(e1); free(us);
        smc2_cuda_free(s);
    }
    
    free(y); free(h); free(z);
}

/*═══════════════════════════════════════════════════════════════════════════
 * Main
 *═══════════════════════════════════════════════════════════════════════════*/

void print_usage(const char* prog) {
    printf("Usage:\n");
    printf("  %s                  Run all tests\n", prog);
    printf("  %s basic            CUDA smoke test\n", prog);
    printf("  %s prior            Prior-data agreement check\n", prog);
    printf("  %s learn            Parameter learning (T=1200)\n", prog);
    printf("  %s highvol          High-vol regime (μ_base=2.0)\n", prog);
    printf("  %s ident            ρ vs σ_z identifiability\n", prog);
    printf("  %s fixedlag         Fixed-lag vs full-history\n", prog);
    printf("  %s throughput       Scaling benchmark\n", prog);
    printf("  %s latency          Per-tick latency distribution\n", prog);
}

int main(int argc, char** argv) {
    printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  SMC² RBPF — 4-Param Convergence Tests                      ║\n");
    printf("║  Learned: ρ, σ_z, μ_base, σ_base                            ║\n");
    printf("║  Fixed:   μ_scale=0.5  μ_rate=1.0  σ_scale=0.1  σ_rate=1.0  ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    
    seed_rng(12345);
    
    if (argc > 1) {
        if (strcmp(argv[1], "basic") == 0)       { test_basic(); }
        else if (strcmp(argv[1], "prior") == 0)  { test_prior_data_agreement(); }
        else if (strcmp(argv[1], "learn") == 0)  { test_parameter_learning(); }
        else if (strcmp(argv[1], "highvol") == 0) { test_high_vol(); }
        else if (strcmp(argv[1], "ident") == 0)  { test_identifiability(); }
        else if (strcmp(argv[1], "fixedlag") == 0) { test_fixed_lag(); }
        else if (strcmp(argv[1], "throughput") == 0) { test_throughput(); }
        else if (strcmp(argv[1], "latency") == 0) { test_latency(); }
        else { print_usage(argv[0]); return 1; }
        return 0;
    }
    
    test_basic();
    test_prior_data_agreement();
    test_parameter_learning();
    test_high_vol();
    test_identifiability();
    test_fixed_lag();
    test_throughput();
    test_latency();
    
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("All tests completed.\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");
    
    return 0;
}
