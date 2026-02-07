/**
 * @file test_icdf_diag.cu  
 * @brief Diagnostic: test PTX ICDF by generating N normals and printing stats
 *
 * Build: nvcc -O3 test_icdf_diag.cu -o test_icdf_diag -lcuda
 * Run:   copy bpf_kernels_full.ptx (or .cubin) to cwd, then run
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main() {
    cudaFree(0);  // init runtime context

    // Load PTX module
    CUmodule mod;
    CUresult err;

    FILE* f = fopen("bpf_kernels_full.cubin", "rb");
    if (!f) f = fopen("bpf_kernels_full.ptx", "rb");
    if (!f) { fprintf(stderr, "Cannot find PTX/cubin\n"); return 1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc(sz + 1);
    fread(buf, 1, sz, f);
    buf[sz] = '\0';
    fclose(f);
    
    err = cuModuleLoadData(&mod, buf);
    free(buf);
    if (err != CUDA_SUCCESS) {
        const char* msg; cuGetErrorString(err, &msg);
        fprintf(stderr, "cuModuleLoadData failed: %s\n", msg);
        return 1;
    }

    CUfunction fn_init_rng, fn_init_particles, fn_propagate_weight, fn_gen_noise;
    cuModuleGetFunction(&fn_init_rng, mod, "bpf_init_rng");
    cuModuleGetFunction(&fn_init_particles, mod, "bpf_init_particles");
    cuModuleGetFunction(&fn_propagate_weight, mod, "bpf_propagate_weight");
    cuModuleGetFunction(&fn_gen_noise, mod, "bpf_gen_noise");
    
    printf("All kernels loaded.\n");

    const int N = 100000;
    const int B = 256;
    const int G = (N + B - 1) / B;

    // Allocate
    uint64_t* d_rng;
    float *d_h, *d_log_w, *d_noise;
    cudaMalloc(&d_rng,   N * 2 * sizeof(uint64_t));
    cudaMalloc(&d_h,     N * sizeof(float));
    cudaMalloc(&d_log_w, N * sizeof(float));
    cudaMalloc(&d_noise, N * sizeof(float));

    // Init RNG
    {
        CUdeviceptr drng = (CUdeviceptr)(uintptr_t)d_rng;
        unsigned long long seed = 42ULL;
        int n = N;
        void* params[] = { &drng, &seed, &n };
        cuLaunchKernel(fn_init_rng, G, 1, 1, B, 1, 1, 0, 0, params, NULL);
    }

    // Test 1: init_particles with mu=0, std=1 -> should give N(0,1)
    {
        CUdeviceptr dh = (CUdeviceptr)(uintptr_t)d_h;
        CUdeviceptr drng = (CUdeviceptr)(uintptr_t)d_rng;
        float mu = 0.0f, std = 1.0f;
        int n = N;
        void* params[] = { &dh, &drng, &mu, &std, &n };
        cuLaunchKernel(fn_init_particles, G, 1, 1, B, 1, 1, 0, 0, params, NULL);
    }
    cudaDeviceSynchronize();

    float* h_h = (float*)malloc(N * sizeof(float));
    cudaMemcpy(h_h, d_h, N * sizeof(float), cudaMemcpyDeviceToHost);

    double sum = 0, sum2 = 0, min_v = 1e30, max_v = -1e30;
    int nan_count = 0, inf_count = 0;
    for (int i = 0; i < N; i++) {
        if (isnan(h_h[i])) { nan_count++; continue; }
        if (isinf(h_h[i])) { inf_count++; continue; }
        sum += h_h[i];
        sum2 += (double)h_h[i] * h_h[i];
        if (h_h[i] < min_v) min_v = h_h[i];
        if (h_h[i] > max_v) max_v = h_h[i];
    }
    int valid = N - nan_count - inf_count;
    double mean = sum / valid;
    double var = sum2 / valid - mean * mean;
    printf("\n=== TEST 1: init_particles (mu=0, std=1) -> N(0,1) ===\n");
    printf("  N=%d  valid=%d  NaN=%d  Inf=%d\n", N, valid, nan_count, inf_count);
    printf("  mean=%.6f (expect 0)  std=%.6f (expect 1)\n", mean, sqrt(var));
    printf("  min=%.4f  max=%.4f\n", min_v, max_v);
    printf("  First 10: ");
    for (int i = 0; i < 10; i++) printf("%.4f ", h_h[i]);
    printf("\n");

    // Test 2: gen_noise -> should also give N(0,1) 
    {
        CUdeviceptr dn = (CUdeviceptr)(uintptr_t)d_noise;
        CUdeviceptr drng = (CUdeviceptr)(uintptr_t)d_rng;
        int n = N;
        void* params[] = { &dn, &drng, &n };
        cuLaunchKernel(fn_gen_noise, G, 1, 1, B, 1, 1, 0, 0, params, NULL);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_h, d_noise, N * sizeof(float), cudaMemcpyDeviceToHost);

    sum = 0; sum2 = 0; nan_count = 0; inf_count = 0;
    min_v = 1e30; max_v = -1e30;
    for (int i = 0; i < N; i++) {
        if (isnan(h_h[i])) { nan_count++; continue; }
        if (isinf(h_h[i])) { inf_count++; continue; }
        sum += h_h[i];
        sum2 += (double)h_h[i] * h_h[i];
        if (h_h[i] < min_v) min_v = h_h[i];
        if (h_h[i] > max_v) max_v = h_h[i];
    }
    valid = N - nan_count - inf_count;
    mean = sum / valid;
    var = sum2 / valid - mean * mean;
    printf("\n=== TEST 2: gen_noise -> N(0,1) ===\n");
    printf("  N=%d  valid=%d  NaN=%d  Inf=%d\n", N, valid, nan_count, inf_count);
    printf("  mean=%.6f (expect 0)  std=%.6f (expect 1)\n", mean, sqrt(var));
    printf("  min=%.4f  max=%.4f\n", min_v, max_v);

    // Test 3: propagate_weight with Gaussian state noise (nu_state=0)
    // Reset particles to h=0
    cudaMemset(d_h, 0, N * sizeof(float));
    // Re-init RNG
    {
        CUdeviceptr drng = (CUdeviceptr)(uintptr_t)d_rng;
        unsigned long long seed = 123ULL;
        int n = N;
        void* params[] = { &drng, &seed, &n };
        cuLaunchKernel(fn_init_rng, G, 1, 1, B, 1, 1, 0, 0, params, NULL);
    }
    // Run propagate_weight with do_prop=1, rho=0, sigma_z=1, mu=0
    // This should give h = 0 + 0*(0-0) + 1*z = z ~ N(0,1)
    {
        CUdeviceptr dh = (CUdeviceptr)(uintptr_t)d_h;
        CUdeviceptr dlw = (CUdeviceptr)(uintptr_t)d_log_w;
        CUdeviceptr drng = (CUdeviceptr)(uintptr_t)d_rng;
        float rho = 0.0f, sigma_z = 1.0f, mu = 0.0f;
        float nu_state = 0.0f, nu_obs = 0.0f;
        float y_t = 0.0f;
        int n = N, do_prop = 1;
        void* params[] = {
            &dh, &dlw, &drng,
            &rho, &sigma_z, &mu, &nu_state, &nu_obs,
            &y_t, &n, &do_prop
        };
        cuLaunchKernel(fn_propagate_weight, G, 1, 1, B, 1, 1, 0, 0, params, NULL);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_h, d_h, N * sizeof(float), cudaMemcpyDeviceToHost);

    sum = 0; sum2 = 0; nan_count = 0; inf_count = 0;
    min_v = 1e30; max_v = -1e30;
    for (int i = 0; i < N; i++) {
        if (isnan(h_h[i])) { nan_count++; continue; }
        if (isinf(h_h[i])) { inf_count++; continue; }
        sum += h_h[i];
        sum2 += (double)h_h[i] * h_h[i];
        if (h_h[i] < min_v) min_v = h_h[i];
        if (h_h[i] > max_v) max_v = h_h[i];
    }
    valid = N - nan_count - inf_count;
    mean = sum / valid;
    var = sum2 / valid - mean * mean;
    printf("\n=== TEST 3: propagate_weight (rho=0, sigma=1, mu=0, nu_state=0) -> N(0,1) ===\n");
    printf("  N=%d  valid=%d  NaN=%d  Inf=%d\n", N, valid, nan_count, inf_count);
    printf("  mean=%.6f (expect 0)  std=%.6f (expect 1)\n", mean, sqrt(var));
    printf("  min=%.4f  max=%.4f\n", min_v, max_v);

    // Test 4: propagate_weight with Student-t state (nu_state=5)
    cudaMemset(d_h, 0, N * sizeof(float));
    {
        CUdeviceptr drng = (CUdeviceptr)(uintptr_t)d_rng;
        unsigned long long seed = 456ULL;
        int n = N;
        void* params[] = { &drng, &seed, &n };
        cuLaunchKernel(fn_init_rng, G, 1, 1, B, 1, 1, 0, 0, params, NULL);
    }
    {
        CUdeviceptr dh = (CUdeviceptr)(uintptr_t)d_h;
        CUdeviceptr dlw = (CUdeviceptr)(uintptr_t)d_log_w;
        CUdeviceptr drng = (CUdeviceptr)(uintptr_t)d_rng;
        float rho = 0.0f, sigma_z = 1.0f, mu = 0.0f;
        float nu_state = 5.0f, nu_obs = 0.0f;
        float y_t = 0.0f;
        int n = N, do_prop = 1;
        void* params[] = {
            &dh, &dlw, &drng,
            &rho, &sigma_z, &mu, &nu_state, &nu_obs,
            &y_t, &n, &do_prop
        };
        cuLaunchKernel(fn_propagate_weight, G, 1, 1, B, 1, 1, 0, 0, params, NULL);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_h, d_h, N * sizeof(float), cudaMemcpyDeviceToHost);

    sum = 0; sum2 = 0; nan_count = 0; inf_count = 0;
    min_v = 1e30; max_v = -1e30;
    for (int i = 0; i < N; i++) {
        if (isnan(h_h[i])) { nan_count++; continue; }
        if (isinf(h_h[i])) { inf_count++; continue; }
        sum += h_h[i];
        sum2 += (double)h_h[i] * h_h[i];
        if (h_h[i] < min_v) min_v = h_h[i];
        if (h_h[i] > max_v) max_v = h_h[i];
    }
    valid = N - nan_count - inf_count;
    mean = sum / valid;
    var = sum2 / valid - mean * mean;
    // Student-t(5) has variance = 5/(5-2) = 5/3 ≈ 1.667, std ≈ 1.291
    printf("\n=== TEST 4: propagate_weight (rho=0, sigma=1, mu=0, nu_state=5) -> t(5) ===\n");
    printf("  N=%d  valid=%d  NaN=%d  Inf=%d\n", N, valid, nan_count, inf_count);
    printf("  mean=%.6f (expect 0)  std=%.6f (expect 1.291)\n", mean, sqrt(var));
    printf("  min=%.4f  max=%.4f\n", min_v, max_v);

    free(h_h);
    cudaFree(d_rng); cudaFree(d_h); cudaFree(d_log_w); cudaFree(d_noise);
    cuModuleUnload(mod);
    return 0;
}
