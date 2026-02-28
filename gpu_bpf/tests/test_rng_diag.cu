/**
 * test_rng_diag.cu - dump PCG32 state and float outputs
 * Build: nvcc -O3 test_rng_diag.cu -o test_rng_diag -lcuda
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int main() {
    cudaFree(0);
    
    CUmodule mod;
    FILE* f = fopen("bpf_kernels_full.cubin", "rb");
    if (!f) f = fopen("bpf_kernels_full.ptx", "rb");
    if (!f) { fprintf(stderr, "No PTX/cubin\n"); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc(sz+1); fread(buf, 1, sz, f); buf[sz]='\0'; fclose(f);
    cuModuleLoadData(&mod, buf); free(buf);
    
    CUfunction fn_init_rng, fn_init_particles;
    cuModuleGetFunction(&fn_init_rng, mod, "bpf_init_rng");
    cuModuleGetFunction(&fn_init_particles, mod, "bpf_init_particles");

    const int N = 16;
    uint64_t* d_rng; float* d_h;
    cudaMalloc(&d_rng, N * 2 * sizeof(uint64_t));
    cudaMalloc(&d_h, N * sizeof(float));

    // Init RNG
    {
        CUdeviceptr drng = (CUdeviceptr)(uintptr_t)d_rng;
        unsigned long long seed = 42ULL;
        int n = N;
        void* params[] = { &drng, &seed, &n };
        cuLaunchKernel(fn_init_rng, 1,1,1, N,1,1, 0,0, params, NULL);
    }
    cudaDeviceSynchronize();

    // Read RNG state
    uint64_t h_rng[32];
    cudaMemcpy(h_rng, d_rng, N * 2 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    printf("=== RNG State after init_rng ===\n");
    for (int i = 0; i < N && i < 8; i++) {
        printf("  particle %d: state=%016llx  inc=%016llx\n",
               i, (unsigned long long)h_rng[i*2], (unsigned long long)h_rng[i*2+1]);
    }

    // Init particles (mu=0, std=1)
    {
        CUdeviceptr dh = (CUdeviceptr)(uintptr_t)d_h;
        CUdeviceptr drng = (CUdeviceptr)(uintptr_t)d_rng;
        float mu = 0.0f, std = 1.0f;
        int n = N;
        void* params[] = { &dh, &drng, &mu, &std, &n };
        cuLaunchKernel(fn_init_particles, 1,1,1, N,1,1, 0,0, params, NULL);
    }
    cudaDeviceSynchronize();

    // Read particles
    float h_h[16];
    cudaMemcpy(h_h, d_h, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Read RNG state after
    uint64_t h_rng2[32];
    cudaMemcpy(h_rng2, d_rng, N * 2 * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    printf("\n=== After init_particles ===\n");
    for (int i = 0; i < N && i < 8; i++) {
        printf("  particle %d: h=%.6f  new_state=%016llx\n",
               i, h_h[i], (unsigned long long)h_rng2[i*2]);
    }

    // Now manually compute what we expect
    printf("\n=== Expected (Python reference) ===\n");
    unsigned long long MULT = 6364136223846793005ULL;
    for (int i = 0; i < 8; i++) {
        unsigned long long inc = (unsigned long long)((2*i+1) | 1);
        unsigned long long s = (42ULL * MULT + inc);
        s = (s * MULT + inc);
        // This is the stored state. init_particles uses it as old.
        unsigned long long old_state = s;
        // PCG output
        unsigned long long tmp64 = ((old_state >> 18) ^ old_state) >> 27;
        unsigned int xsh = (unsigned int)(tmp64 & 0xFFFFFFFF);
        unsigned int rot = (unsigned int)(old_state >> 59);
        unsigned int lo = xsh >> rot;
        unsigned int neg_rot = (unsigned int)((-((int)rot)) & 31);
        unsigned int hi = xsh << neg_rot;
        unsigned int output = lo | hi;
        float u = (float)(output >> 9) / 8388608.0f;
        printf("  particle %d: expected_state=%016llx  pcg_out=%08x  u=%.6f\n",
               i, (unsigned long long)s, output, u);
    }

    cudaFree(d_rng); cudaFree(d_h);
    cuModuleUnload(mod);
    return 0;
}
