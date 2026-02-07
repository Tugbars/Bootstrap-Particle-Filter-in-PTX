/**
 * @file bpf_ptx_host.cu
 * @brief Host driver for the PTX BPF kernels
 *
 * Uses CUDA Driver API (cuModuleLoadData, cuLaunchKernel) to load
 * the hand-written PTX and run the full BPF pipeline.
 *
 * Compile:
 *   nvcc -o bpf_ptx_host bpf_ptx_host.cu -lcuda -arch=sm_100
 *
 * Or load PTX at runtime from file (see load_ptx_from_file).
 *
 * Educational companion to bpf_kernels.ptx — shows how kernel launch
 * parameters map to the PTX .param declarations.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/system/cuda/execution_policy.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// =============================================================================
// PTX Module Manager
// =============================================================================

typedef struct {
    CUmodule   module;

    // Kernel function handles
    CUfunction propagate_weight;
    CUfunction set_scalar;
    CUfunction reduce_max;
    CUfunction reduce_sum;
    CUfunction exp_sub;
    CUfunction scale_wh;
    CUfunction compute_loglik;
    CUfunction resample;
} PtxKernels;

static char* load_ptx_from_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc(sz + 1);
    fread(buf, 1, sz, f);
    buf[sz] = '\0';
    fclose(f);
    return buf;
}

static PtxKernels load_kernels(const char* ptx_path) {
    PtxKernels k;
    memset(&k, 0, sizeof(k));

    char* ptx_source = load_ptx_from_file(ptx_path);
    if (!ptx_source) exit(1);

    CUresult err = cuModuleLoadData(&k.module, ptx_source);
    if (err != CUDA_SUCCESS) {
        const char* msg;
        cuGetErrorString(err, &msg);
        fprintf(stderr, "cuModuleLoadData failed: %s\n", msg);
        exit(1);
    }
    free(ptx_source);

    // Extract kernel handles — names must match .visible .entry in PTX
    cuModuleGetFunction(&k.propagate_weight, k.module, "bpf_propagate_weight");
    cuModuleGetFunction(&k.set_scalar,       k.module, "bpf_set_scalar");
    cuModuleGetFunction(&k.reduce_max,       k.module, "bpf_reduce_max");
    cuModuleGetFunction(&k.reduce_sum,       k.module, "bpf_reduce_sum");
    cuModuleGetFunction(&k.exp_sub,          k.module, "bpf_exp_sub");
    cuModuleGetFunction(&k.scale_wh,         k.module, "bpf_scale_wh");
    cuModuleGetFunction(&k.compute_loglik,   k.module, "bpf_compute_loglik");
    cuModuleGetFunction(&k.resample,         k.module, "bpf_resample");

    return k;
}

// =============================================================================
// BPF State (mirrors gpu_bpf.cuh but uses Driver API)
// =============================================================================

typedef struct {
    int   n_particles;
    int   block;
    int   grid;
    float rho, sigma_z, mu;

    // Device arrays
    CUdeviceptr d_h;
    CUdeviceptr d_h2;
    CUdeviceptr d_log_w;
    CUdeviceptr d_w;
    CUdeviceptr d_cdf;
    CUdeviceptr d_wh;
    CUdeviceptr d_scalars;   // [max_lw, sum_w, h_est, log_lik]

    // Stream
    CUstream stream;

    // Host RNG
    unsigned long long host_rng_state;
    int timestep;

    // PTX kernels
    PtxKernels kern;
} PtxBpfState;

// Host PCG32
static inline unsigned int ptx_pcg32(unsigned long long* state) {
    unsigned long long old = *state;
    *state = old * 6364136223846793005ULL + 1442695040888963407ULL;
    unsigned int xor_shifted = (unsigned int)(((old >> 18u) ^ old) >> 27u);
    unsigned int rot = (unsigned int)(old >> 59u);
    return (xor_shifted >> rot) | (xor_shifted << ((-rot) & 31));
}

static inline float ptx_pcg32_float(unsigned long long* state) {
    return (float)(ptx_pcg32(state) >> 9) * (1.0f / 8388608.0f);
}

// =============================================================================
// Helper: Launch a PTX kernel with typed params
//
// The CUDA Driver API wants void** pointing to each parameter.
// Each pointer points to the actual value (matching .param type in PTX).
// =============================================================================

static void launch_kernel(
    CUfunction func, CUstream stream,
    unsigned int gridX, unsigned int blockX,
    unsigned int smem,
    void** params
) {
    CUresult err = cuLaunchKernel(
        func,
        gridX, 1, 1,      // grid dims
        blockX, 1, 1,      // block dims
        smem,              // shared memory bytes
        stream,
        params,
        NULL               // extra (unused)
    );
    if (err != CUDA_SUCCESS) {
        const char* msg;
        cuGetErrorString(err, &msg);
        fprintf(stderr, "cuLaunchKernel failed: %s\n", msg);
    }
}

// =============================================================================
// BPF Step using PTX kernels
// =============================================================================

static void ptx_bpf_step(PtxBpfState* s, float y_t) {
    int    n    = s->n_particles;
    int    g    = s->grid;
    int    b    = s->block;
    size_t smem = b * sizeof(float);

    // ─── 1. Propagate + Weight ───
    // PTX params: (u64 h, u64 log_w, f32 y_t, s32 n)
    //
    // NOTE: Our PTX kernel is Gaussian-only and doesn't propagate (no RNG).
    // In a full implementation, you'd have a separate propagate kernel
    // or integrate a PTX PRNG. Here we demonstrate the weighting step.
    {
        void* params[] = {
            &s->d_h, &s->d_log_w, &y_t, &n
        };
        launch_kernel(s->kern.propagate_weight, s->stream, g, b, 0, params);
    }

    // ─── 2. Max of log_w → d_scalars[0] ───
    {
        CUdeviceptr ptr = s->d_scalars + 0;  // byte offset 0
        float init_val = -1e30f;
        void* p1[] = { &ptr, &init_val };
        launch_kernel(s->kern.set_scalar, s->stream, 1, 1, 0, p1);

        void* p2[] = { &s->d_log_w, &ptr, &n };
        launch_kernel(s->kern.reduce_max, s->stream, g, b, smem, p2);
    }

    // ─── 3. w = exp(log_w - max) ───
    {
        CUdeviceptr d_max = s->d_scalars + 0;
        void* params[] = { &s->d_w, &s->d_log_w, &d_max, &n };
        launch_kernel(s->kern.exp_sub, s->stream, g, b, 0, params);
    }

    // ─── 4. Sum of w → d_scalars[1] ───
    {
        CUdeviceptr ptr = s->d_scalars + sizeof(float);  // byte offset 4
        float init_val = 0.0f;
        void* p1[] = { &ptr, &init_val };
        launch_kernel(s->kern.set_scalar, s->stream, 1, 1, 0, p1);

        void* p2[] = { &s->d_w, &ptr, &n };
        launch_kernel(s->kern.reduce_sum, s->stream, g, b, smem, p2);
    }

    // ─── 5. Normalize + w*h ───
    {
        CUdeviceptr d_sum = s->d_scalars + sizeof(float);
        void* params[] = { &s->d_w, &s->d_wh, &s->d_h, &d_sum, &n };
        launch_kernel(s->kern.scale_wh, s->stream, g, b, 0, params);
    }

    // ─── 6. h_est = sum(w*h) → d_scalars[2] ───
    {
        CUdeviceptr ptr = s->d_scalars + 2 * sizeof(float);
        float init_val = 0.0f;
        void* p1[] = { &ptr, &init_val };
        launch_kernel(s->kern.set_scalar, s->stream, 1, 1, 0, p1);

        void* p2[] = { &s->d_wh, &ptr, &n };
        launch_kernel(s->kern.reduce_sum, s->stream, g, b, smem, p2);
    }

    // ─── 7. Log-likelihood ───
    {
        void* params[] = { &s->d_scalars, &n };
        launch_kernel(s->kern.compute_loglik, s->stream, 1, 1, 0, params);
    }

    // ─── 8. CDF via thrust (still using thrust for prefix scan) ───
    // In a full PTX implementation you'd write a Blelloch scan.
    // That's ~200 more lines of PTX — left as exercise.
    {
        float* w_ptr   = (float*)(uintptr_t)s->d_w;
        float* cdf_ptr = (float*)(uintptr_t)s->d_cdf;
        cudaStream_t rt_stream = (cudaStream_t)s->stream;
        thrust::inclusive_scan(
            thrust::cuda::par.on(rt_stream),
            thrust::device_ptr<float>(w_ptr),
            thrust::device_ptr<float>(w_ptr + n),
            thrust::device_ptr<float>(cdf_ptr));
    }

    // ─── 9. Resample ───
    {
        float u = ptx_pcg32_float(&s->host_rng_state) / (float)n;
        void* params[] = { &s->d_h2, &s->d_h, &s->d_cdf, &u, &n };
        launch_kernel(s->kern.resample, s->stream, g, b, 0, params);
    }

    // Swap buffers
    CUdeviceptr tmp = s->d_h;
    s->d_h  = s->d_h2;
    s->d_h2 = tmp;
    s->timestep++;
}

// =============================================================================
// Read result
// =============================================================================

typedef struct {
    float h_mean;
    float log_lik;
} PtxBpfResult;

static PtxBpfResult ptx_bpf_get_result(PtxBpfState* s) {
    float scalars[4];
    cuMemcpyDtoH(scalars, s->d_scalars, 4 * sizeof(float));
    PtxBpfResult r;
    r.h_mean  = scalars[2];
    r.log_lik = scalars[3];
    return r;
}

// =============================================================================
// Main — demo
// =============================================================================

int main(int argc, char** argv) {
    const char* ptx_path = (argc > 1) ? argv[1] : "bpf_kernels.ptx";

    // Initialize CUDA Driver API
    cuInit(0);
    CUdevice dev;
    CUcontext ctx;
    cuDeviceGet(&dev, 0);
    cuCtxCreate(&ctx, 0, dev);

    // Print device info
    char name[256];
    cuDeviceGetName(name, sizeof(name), dev);
    printf("Device: %s\n", name);

    int sm_major, sm_minor;
    cuDeviceGetAttribute(&sm_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&sm_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    printf("SM: %d.%d\n", sm_major, sm_minor);

    // Load PTX kernels
    PtxKernels kern = load_kernels(ptx_path);
    printf("PTX module loaded successfully.\n");

    // Setup BPF state
    int N = 4096;
    PtxBpfState state;
    memset(&state, 0, sizeof(state));
    state.n_particles = N;
    state.block = 256;
    state.grid  = (N + 255) / 256;
    state.rho   = 0.98f;
    state.sigma_z = 0.15f;
    state.mu    = -1.0f;
    state.host_rng_state = 42ULL * 67890ULL + 12345ULL;
    state.timestep = 0;
    state.kern  = kern;

    cuStreamCreate(&state.stream, 0);

    // Allocate device memory
    cuMemAlloc(&state.d_h,       N * sizeof(float));
    cuMemAlloc(&state.d_h2,      N * sizeof(float));
    cuMemAlloc(&state.d_log_w,   N * sizeof(float));
    cuMemAlloc(&state.d_w,       N * sizeof(float));
    cuMemAlloc(&state.d_cdf,     N * sizeof(float));
    cuMemAlloc(&state.d_wh,      N * sizeof(float));
    cuMemAlloc(&state.d_scalars, 4 * sizeof(float));

    // Initialize particles to mu (simplified — no RNG propagation in PTX)
    float* h_init = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_init[i] = state.mu;
    cuMemcpyHtoD(state.d_h, h_init, N * sizeof(float));
    free(h_init);

    // Run 100 steps with synthetic returns
    printf("\nRunning %d BPF steps with PTX kernels...\n", 100);
    printf("%-6s  %-12s  %-12s\n", "Step", "h_mean", "log_lik");
    printf("------  ------------  ------------\n");

    srand(123);
    for (int t = 0; t < 100; t++) {
        // Synthetic return: small random value
        float y_t = 0.01f * ((float)rand() / RAND_MAX - 0.5f);

        ptx_bpf_step(&state, y_t);
        cuStreamSynchronize(state.stream);

        if (t % 10 == 0) {
            PtxBpfResult r = ptx_bpf_get_result(&state);
            printf("%-6d  %12.6f  %12.6f\n", t, r.h_mean, r.log_lik);
        }
    }

    // Cleanup
    cuMemFree(state.d_h);
    cuMemFree(state.d_h2);
    cuMemFree(state.d_log_w);
    cuMemFree(state.d_w);
    cuMemFree(state.d_cdf);
    cuMemFree(state.d_wh);
    cuMemFree(state.d_scalars);
    cuStreamDestroy(state.stream);
    cuModuleUnload(kern.module);
    cuCtxDestroy(ctx);

    printf("\nDone.\n");
    return 0;
}
