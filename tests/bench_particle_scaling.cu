/**
 * @file bench_particle_scaling.cu
 * @brief Benchmark: BPF latency vs particle count (50K - 500K)
 *
 * Measures per-tick latency for the full PTX BPF path at various particle
 * counts. Uses synthetic SV returns so no data files needed.
 *
 * Build (from project root):
 *   Add to CMakeLists.txt or build standalone:
 *   nvcc -O3 tests/bench_particle_scaling.cu gpu_bpf/gpu_bpf_ptx_full.cu
 *        -o build/bench_particle_scaling -lcuda -lcurand -DUSE_PTX_FULL=1
 *        -I. -Igpu_bpf --gpu-architecture=sm_120 --expt-relaxed-constexpr
 *
 * Run from build/ directory (needs bpf_kernels_full.cubin or .ptx)
 */

#ifdef USE_PTX_FULL
#include "gpu_bpf_full.cuh"
#else
#include "gpu_bpf.cuh"
#endif

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Simple LCG for generating synthetic returns
static double lcg_normal(unsigned long long* state) {
    // Box-Muller from two LCG uniforms
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u1 = (double)(*state >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-15) u1 = 1e-15;
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u2 = (double)(*state >> 11) / (double)(1ULL << 53);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

int main() {
    // Print device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (SM %d.%d, %d SMs)\n\n", prop.name,
           prop.major, prop.minor, prop.multiProcessorCount);

    // DGP parameters (Baseline: Gaussian state, Student-t(5) obs)
    const float rho = 0.98f, sigma_z = 0.15f, mu = -1.0f;
    const float nu_state = 0.0f, nu_obs = 5.0f;
    const int seed = 42;

    // Generate synthetic returns
    const int n_ticks = 2000;
    const int warmup_ticks = 200;  // warmup before timing
    double* returns = (double*)malloc(n_ticks * sizeof(double));
    {
        unsigned long long rng = 123456789ULL;
        double h = mu;
        for (int t = 0; t < n_ticks; t++) {
            h = mu + rho * (h - mu) + sigma_z * lcg_normal(&rng);
            double vol = exp(h / 2.0);
            returns[t] = vol * lcg_normal(&rng);
        }
    }

    // Particle counts to benchmark
    const int counts[] = { 50000, 100000, 200000, 300000, 400000, 500000, 1000000};
    const int n_counts = sizeof(counts) / sizeof(counts[0]);

    printf("%-12s %10s %10s %10s %12s\n",
           "Particles", "Total(ms)", "Ticks", "us/tick", "Throughput");
    printf("%-12s %10s %10s %10s %12s\n",
           "----------", "--------", "-----", "-------", "----------");

    for (int ci = 0; ci < n_counts; ci++) {
        int np = counts[ci];

        // Create filter
        GpuBpfState* f = gpu_bpf_create(np, rho, sigma_z, mu,
                                         nu_state, nu_obs, seed);

        // Warmup (untimed)
        for (int t = 0; t < warmup_ticks; t++) {
            gpu_bpf_step(f, (float)returns[t % n_ticks]);
        }
        cudaDeviceSynchronize();

        // Timed run
        const int bench_ticks = n_ticks - warmup_ticks;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        for (int t = 0; t < bench_ticks; t++) {
            gpu_bpf_step(f, (float)returns[(warmup_ticks + t) % n_ticks]);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        double us_per_tick = (double)ms * 1000.0 / bench_ticks;
        double ticks_per_sec = bench_ticks / ((double)ms / 1000.0);

        printf("%10dK %9.1fms %10d %8.1fus %10.0f t/s\n",
               np / 1000, ms, bench_ticks, us_per_tick, ticks_per_sec);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        gpu_bpf_destroy(f);
    }

    printf("\nDGP: rho=%.2f sigma_z=%.2f mu=%.1f nu_state=%.0f nu_obs=%.0f\n",
           rho, sigma_z, mu, nu_state, nu_obs);

    free(returns);
    return 0;
}
