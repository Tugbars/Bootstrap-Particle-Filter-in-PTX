// =============================================================================
// Conditional Resampling Test — GPU BPF
//
// Compare always-resample vs ESS-triggered resampling at various thresholds.
// Measures: RMSE, resample rate, timing.
//
// Key question: how much particle diversity do we gain by skipping unnecessary
// resamples, and does it improve filtering accuracy?
//
// Build: nvcc -O3 test_cond_resample.cu gpu_bpf.cu -o test_cond_resample
//        -lcurand -arch=sm_89 (or sm_120 for RTX 5080)
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include "gpu_bpf.cuh"

// =============================================================================
// PRNG for DGP
// =============================================================================

static unsigned long long g_rng_state = 0;

static float randn_host() {
    // PCG32
    unsigned long long old = g_rng_state;
    g_rng_state = old * 6364136223846793005ULL + 1442695040888963407ULL;
    unsigned int xsh = (unsigned int)(((old >> 18u) ^ old) >> 27u);
    unsigned int rot = (unsigned int)(old >> 59u);
    unsigned int r = (xsh >> rot) | (xsh << ((-rot) & 31));
    float u1 = (float)(r >> 9) * (1.0f / 8388608.0f) + 1e-10f;
    old = g_rng_state;
    g_rng_state = old * 6364136223846793005ULL + 1442695040888963407ULL;
    xsh = (unsigned int)(((old >> 18u) ^ old) >> 27u);
    rot = (unsigned int)(old >> 59u);
    r = (xsh >> rot) | (xsh << ((-rot) & 31));
    float u2 = (float)(r >> 9) * (1.0f / 8388608.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979f * u2);
}

// =============================================================================
// DGP
// =============================================================================

struct DGP {
    float h, rho, sigma_z, mu;
};

static float dgp_step(DGP& d) {
    d.h = d.mu + d.rho * (d.h - d.mu) + d.sigma_z * randn_host();
    return expf(d.h * 0.5f) * randn_host();
}

// =============================================================================
// Run one scenario at one threshold
// =============================================================================

struct RunResult {
    float rmse;
    float spike_rmse;  // RMSE on |y|>3 ticks
    int resample_count;
    int total_ticks;
    float elapsed_ms;
};

static RunResult run_single(
    const double* returns, const double* true_h, int T,
    int n_particles, float rho, float sigma_z, float mu,
    float nu_state, float nu_obs, float ess_threshold,
    float silverman_shrink, int seed)
{
    GpuBpfState* s = gpu_bpf_create(n_particles, rho, sigma_z, mu,
                                     nu_state, nu_obs, seed);
    s->silverman_shrink = silverman_shrink;
    gpu_bpf_set_ess_threshold(s, ess_threshold);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    int skip = 100;
    double sse = 0, spike_sse = 0;
    int count = 0, spike_count = 0;

    cudaEventRecord(t0);
    for (int t = 0; t < T; t++) {
        BpfResult r = gpu_bpf_step(s, (float)returns[t]);
        if (t >= skip) {
            double err = r.h_mean - true_h[t];
            sse += err * err;
            count++;
            if (fabs(returns[t]) > 3.0) {
                spike_sse += err * err;
                spike_count++;
            }
        }
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms;
    cudaEventElapsedTime(&ms, t0, t1);

    RunResult rr;
    rr.rmse = count > 0 ? (float)sqrt(sse / count) : 0;
    rr.spike_rmse = spike_count > 0 ? (float)sqrt(spike_sse / spike_count) : 0;
    rr.resample_count = gpu_bpf_get_resample_count(s);
    rr.total_ticks = T;
    rr.elapsed_ms = ms;

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    gpu_bpf_destroy(s);
    return rr;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    float true_rho = 0.98f, true_sz = 0.15f, true_mu = -4.5f;
    int N = 300000;  // GPU-scale
    int T = 3000;

    printf("═══════════════════════════════════════════════════════════════════════════\n");
    printf("  Conditional Resampling Test — GPU BPF\n");
    printf("  DGP: ρ=%.2f  σ_z=%.2f  μ=%.1f  │  N=%dK  T=%d\n",
           true_rho, true_sz, true_mu, N/1000, T);
    printf("═══════════════════════════════════════════════════════════════════════════\n");

    // =========================================================================
    // Generate shared data for 4 scenarios
    // =========================================================================

    // Scenario 1: Standard SV (oracle params)
    double* ret1 = (double*)malloc(T * sizeof(double));
    double* th1  = (double*)malloc(T * sizeof(double));
    g_rng_state = 42;
    {
        DGP d = {true_mu, true_rho, true_sz, true_mu};
        for (int t = 0; t < T; t++) {
            ret1[t] = dgp_step(d);
            th1[t] = d.h;
        }
    }

    // Scenario 2: Wrong σ_z (misspec: using 0.08 when true=0.15)
    // Same data as scenario 1, different filter params

    // Scenario 3: Regime change (σ_z 0.15→0.40 at t=1500)
    double* ret3 = (double*)malloc(T * sizeof(double));
    double* th3  = (double*)malloc(T * sizeof(double));
    g_rng_state = 42;
    {
        DGP d = {true_mu, true_rho, true_sz, true_mu};
        for (int t = 0; t < T; t++) {
            if (t == 1500) d.sigma_z = 0.40f;
            ret3[t] = dgp_step(d);
            th3[t] = d.h;
        }
    }

    // Scenario 4: High vol (σ_z=0.40 throughout)
    double* ret4 = (double*)malloc(T * sizeof(double));
    double* th4  = (double*)malloc(T * sizeof(double));
    g_rng_state = 42;
    {
        DGP d = {true_mu, true_rho, 0.40f, true_mu};
        for (int t = 0; t < T; t++) {
            ret4[t] = dgp_step(d);
            th4[t] = d.h;
        }
    }

    // =========================================================================
    // ESS threshold sweep
    // =========================================================================

    float thresholds[] = {0.0f, 0.3f, 0.5f, 0.7f, 0.9f};
    int n_thresh = 5;
    const char* thresh_names[] = {"always", "0.3", "0.5", "0.7", "0.9"};

    struct Scenario {
        const char* name;
        const double* returns;
        const double* true_h;
        float filter_rho, filter_sz, filter_mu;
        float nu_state, nu_obs;
    };

    Scenario scenarios[] = {
        {"Oracle params",     ret1, th1, true_rho, true_sz, true_mu, 0, 0},
        {"Wrong σ_z (0.08)",  ret1, th1, true_rho, 0.08f,   true_mu, 0, 0},
        {"Regime 0.15→0.40",  ret3, th3, true_rho, true_sz, true_mu, 0, 0},
        {"High vol (σ_z=0.4)",ret4, th4, true_rho, 0.40f,   true_mu, 0, 0},
    };
    int n_scenarios = 4;

    for (int si = 0; si < n_scenarios; si++) {
        Scenario& sc = scenarios[si];
        printf("\n┌── %s ", sc.name);
        for (int p = (int)strlen(sc.name); p < 65; p++) printf("─");
        printf("┐\n");
        printf("  %6s │ %8s  %8s │ %6s │ %7s │ %6s\n",
               "thresh", "RMSE", "spkRMSE", "resamp", "rate", "ms");
        printf("  ────── │ ────────  ──────── │ ────── │ ─────── │ ──────\n");

        float base_rmse = 0;
        for (int ti = 0; ti < n_thresh; ti++) {
            RunResult r = run_single(
                sc.returns, sc.true_h, T,
                N, sc.filter_rho, sc.filter_sz, sc.filter_mu,
                sc.nu_state, sc.nu_obs, thresholds[ti],
                0.5f,  // silverman
                42);

            if (ti == 0) base_rmse = r.rmse;
            float pct = r.resample_count * 100.0f / r.total_ticks;

            printf("  %6s │ %8.4f  %8.4f │ %5d  │ %5.1f%%  │ %5.1f\n",
                   thresh_names[ti], r.rmse, r.spike_rmse,
                   r.resample_count, pct, r.elapsed_ms);
        }
        printf("└");
        for (int p = 0; p < 75; p++) printf("─");
        printf("┘\n");
    }

    // =========================================================================
    // Particle count × threshold interaction
    // =========================================================================
    printf("\n┌── Particle count × threshold (Oracle params) ─────────────────────────┐\n");
    printf("  %7s │", "N");
    for (int ti = 0; ti < n_thresh; ti++) printf(" %8s", thresh_names[ti]);
    printf("\n  ─────── │");
    for (int ti = 0; ti < n_thresh; ti++) printf(" ────────");
    printf("\n");

    int npfs[] = {50000, 100000, 300000, 500000};
    int n_npfs = 4;

    for (int ni = 0; ni < n_npfs; ni++) {
        printf("  %5dK │", npfs[ni]/1000);
        for (int ti = 0; ti < n_thresh; ti++) {
            RunResult r = run_single(ret1, th1, T,
                npfs[ni], true_rho, true_sz, true_mu, 0, 0,
                thresholds[ti], 0.5f, 42);
            printf(" %8.4f", r.rmse);
        }
        printf("\n");
    }
    printf("└");
    for (int p = 0; p < 75; p++) printf("─");
    printf("┘\n");

    // =========================================================================
    // Silverman interaction: does conditional resample reduce need for jitter?
    // =========================================================================
    printf("\n┌── Silverman × Threshold interaction (300K, Oracle) ───────────────────┐\n");
    printf("  %6s  %4s │ %8s  %6s\n",
           "thresh", "silv", "RMSE", "resamp");
    printf("  ──────  ──── │ ────────  ──────\n");

    float silvs[] = {0.0f, 0.3f, 0.5f, 0.7f};
    float ess_ts[] = {0.0f, 0.5f};
    const char* silv_names[] = {"off", "0.3", "0.5", "0.7"};
    const char* ess_names[] = {"always", "0.5"};

    for (int ei = 0; ei < 2; ei++) {
        for (int si = 0; si < 4; si++) {
            RunResult r = run_single(ret1, th1, T,
                N, true_rho, true_sz, true_mu, 0, 0,
                ess_ts[ei], silvs[si], 42);
            float pct = r.resample_count * 100.0f / r.total_ticks;
            printf("  %6s  %4s │ %8.4f  %5.1f%%\n",
                   ess_names[ei], silv_names[si], r.rmse, pct);
        }
        printf("  ──────  ──── │ ────────  ──────\n");
    }
    printf("└");
    for (int p = 0; p < 75; p++) printf("─");
    printf("┘\n");

    printf("═══════════════════════════════════════════════════════════════════════════\n");

    free(ret1); free(th1);
    free(ret3); free(th3);
    free(ret4); free(th4);
    return 0;
}
