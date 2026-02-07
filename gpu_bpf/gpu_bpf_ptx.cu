/**
 * @file gpu_bpf_ptx.cu
 * @brief GPU BPF using hand-written PTX for weight processing + cuRAND for RNG
 *
 * Hybrid approach:
 *   - nvcc-compiled: init_rng, init_particles, propagate_weight (need cuRAND)
 *   - Hand-written PTX: set_scalar, reduce_max, reduce_sum, exp_sub,
 *                        scale_wh, compute_loglik, resample
 *
 * Exposes the EXACT same API as gpu_bpf.cuh so the test harness can link
 * against this file instead of gpu_bpf.cu.
 *
 * Build:
 *   nvcc -O3 test_bpf_matched_dgp.cu gpu_bpf_ptx.cu -o test_bpf_ptx -lcuda -lcurand
 *
 * The PTX file (bpf_kernels.ptx) must be in the working directory at runtime.
 */

#include "gpu_bpf.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/system/cuda/execution_policy.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// =============================================================================
// PTX Module — loaded once, shared by all BPF instances
// =============================================================================

static CUmodule  g_ptx_module = NULL;

typedef struct {
    CUfunction set_scalar;
    CUfunction reduce_max;
    CUfunction reduce_sum;
    CUfunction exp_sub;
    CUfunction scale_wh;
    CUfunction compute_loglik;
    CUfunction resample;
} PtxFunctions;

static PtxFunctions g_ptx;
static int          g_ptx_loaded = 0;

static void ensure_ptx_loaded() {
    if (g_ptx_loaded) return;

    // Try several paths for the PTX file
    const char* paths[] = {
        "bpf_kernels.ptx",
        "./bpf_kernels.ptx",
        NULL
    };

    char* ptx_source = NULL;
    for (int i = 0; paths[i]; i++) {
        FILE* f = fopen(paths[i], "rb");
        if (!f) continue;
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        ptx_source = (char*)malloc(sz + 1);
        fread(ptx_source, 1, sz, f);
        ptx_source[sz] = '\0';
        fclose(f);
        fprintf(stderr, "[PTX] Loaded from: %s (%ld bytes)\n", paths[i], sz);
        break;
    }

    if (!ptx_source) {
        fprintf(stderr, "[PTX] ERROR: Cannot find bpf_kernels.ptx\n");
        exit(1);
    }

    CUresult err = cuModuleLoadData(&g_ptx_module, ptx_source);
    free(ptx_source);

    if (err != CUDA_SUCCESS) {
        const char* msg;
        cuGetErrorString(err, &msg);
        fprintf(stderr, "[PTX] cuModuleLoadData failed: %s\n", msg);
        exit(1);
    }

    // Extract kernel handles
    cuModuleGetFunction(&g_ptx.set_scalar,     g_ptx_module, "bpf_set_scalar");
    cuModuleGetFunction(&g_ptx.reduce_max,     g_ptx_module, "bpf_reduce_max");
    cuModuleGetFunction(&g_ptx.reduce_sum,     g_ptx_module, "bpf_reduce_sum");
    cuModuleGetFunction(&g_ptx.exp_sub,        g_ptx_module, "bpf_exp_sub");
    cuModuleGetFunction(&g_ptx.scale_wh,       g_ptx_module, "bpf_scale_wh");
    cuModuleGetFunction(&g_ptx.compute_loglik, g_ptx_module, "bpf_compute_loglik");
    cuModuleGetFunction(&g_ptx.resample,       g_ptx_module, "bpf_resample");

    g_ptx_loaded = 1;
    fprintf(stderr, "[PTX] All 7 kernels loaded successfully\n");
}

// =============================================================================
// Driver API launch helper
// =============================================================================

static inline void ptx_launch(
    CUfunction func, cudaStream_t stream,
    unsigned int gridX, unsigned int blockX,
    unsigned int smem, void** params
) {
    cuLaunchKernel(func,
        gridX, 1, 1,
        blockX, 1, 1,
        smem, (CUstream)stream, params, NULL);
}

// =============================================================================
// cuRAND-based kernels (compiled by nvcc) — init + propagation
// =============================================================================

static __device__ float bpf_sample_normal(curandState* st) {
    return curand_normal(st);
}

static __device__ float bpf_sample_t(curandState* st, float nu) {
    if (nu <= 0.0f) return bpf_sample_normal(st);
    float z = bpf_sample_normal(st);
    float chi2 = 0.0f;
    int nu_int = (int)nu;
    for (int k = 0; k < nu_int; k++) {
        float g = bpf_sample_normal(st);
        chi2 += g * g;
    }
    return z * rsqrtf(chi2 / nu);
}

static __device__ float bpf_log_t_pdf(float x, float nu) {
    float nu_h  = nu * 0.5f;
    float nup1h = (nu + 1.0f) * 0.5f;
    return lgammaf(nup1h) - lgammaf(nu_h)
         - 0.5f * logf(nu * 3.14159265f)
         - nup1h * logf(1.0f + (x * x) / nu);
}

__global__ void bpf_init_rng_k(curandState* states, unsigned long long seed, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) curand_init(seed, i, 0, &states[i]);
}

__global__ void bpf_init_particles_k(float* h, curandState* states,
                                      float mu, float std_stat, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = mu + std_stat * curand_normal(&states[i]);
}

__global__ void bpf_propagate_weight_k(
    float* h, float* log_w, curandState* states,
    float rho, float sigma_z, float mu,
    float nu_state, float nu_obs, float y_t,
    int n, int do_propagate
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (do_propagate) {
        float eps = bpf_sample_t(&states[i], nu_state);
        h[i] = mu + rho * (h[i] - mu) + sigma_z * eps;
    }
    float h_i = h[i];
    float eta = y_t * __expf(-h_i * 0.5f);
    log_w[i] = (nu_obs > 0.0f)
        ? bpf_log_t_pdf(eta, nu_obs) - h_i * 0.5f
        : -0.9189385f - 0.5f * eta * eta - h_i * 0.5f;
}

// =============================================================================
// Host PCG32 for systematic resampling U
// =============================================================================

static inline unsigned int host_pcg32(unsigned long long* state) {
    unsigned long long old = *state;
    *state = old * 6364136223846793005ULL + 1442695040888963407ULL;
    unsigned int xsh = (unsigned int)(((old >> 18u) ^ old) >> 27u);
    unsigned int rot = (unsigned int)(old >> 59u);
    return (xsh >> rot) | (xsh << ((-rot) & 31));
}

static inline float host_pcg32_float(unsigned long long* state) {
    return (float)(host_pcg32(state) >> 9) * (1.0f / 8388608.0f);
}

// =============================================================================
// gpu_bpf_create — allocate + init (cuRAND kernels), load PTX
// =============================================================================

GpuBpfState* gpu_bpf_create(int n_particles, float rho, float sigma_z, float mu,
                              float nu_state, float nu_obs, int seed) {
    // Ensure PTX module is loaded (idempotent)
    cuInit(0);
    ensure_ptx_loaded();

    GpuBpfState* s = (GpuBpfState*)calloc(1, sizeof(GpuBpfState));
    s->n_particles = n_particles;
    s->rho = rho;
    s->sigma_z = sigma_z;
    s->mu = mu;
    s->nu_state = nu_state;
    s->nu_obs = nu_obs;
    s->block = 256;
    s->grid = (n_particles + s->block - 1) / s->block;
    s->host_rng_state = (unsigned long long)seed * 67890ULL + 12345ULL;
    s->timestep = 0;
    s->silverman_shrink = 0.0f;  // Silverman off for now (PTX has no var/jitter)

    cudaStreamCreate(&s->stream);

    cudaMalloc(&s->d_h,       n_particles * sizeof(float));
    cudaMalloc(&s->d_h2,      n_particles * sizeof(float));
    cudaMalloc(&s->d_log_w,   n_particles * sizeof(float));
    cudaMalloc(&s->d_w,       n_particles * sizeof(float));
    cudaMalloc(&s->d_cdf,     n_particles * sizeof(float));
    cudaMalloc(&s->d_wh,      n_particles * sizeof(float));
    cudaMalloc(&s->d_rng,     n_particles * sizeof(curandState));
    cudaMalloc(&s->d_scalars, 4 * sizeof(float));
    cudaMalloc(&s->d_noise,   n_particles * sizeof(float));
    cudaMalloc(&s->d_var,     sizeof(float));

    // Init RNG + particles (cuRAND, compiled by nvcc)
    bpf_init_rng_k<<<s->grid, s->block, 0, s->stream>>>(
        s->d_rng, (unsigned long long)seed, n_particles);

    float std_stat = sqrtf((sigma_z * sigma_z) / fmaxf(1.0f - rho * rho, 1e-6f));
    bpf_init_particles_k<<<s->grid, s->block, 0, s->stream>>>(
        s->d_h, s->d_rng, mu, std_stat, n_particles);
    cudaStreamSynchronize(s->stream);

    return s;
}

// =============================================================================
// gpu_bpf_destroy
// =============================================================================

void gpu_bpf_destroy(GpuBpfState* s) {
    if (!s) return;
    cudaStreamDestroy(s->stream);
    cudaFree(s->d_h);
    cudaFree(s->d_h2);
    cudaFree(s->d_log_w);
    cudaFree(s->d_w);
    cudaFree(s->d_cdf);
    cudaFree(s->d_wh);
    cudaFree(s->d_rng);
    cudaFree(s->d_scalars);
    cudaFree(s->d_noise);
    cudaFree(s->d_var);
    free(s);
}

// =============================================================================
// gpu_bpf_step_async — HYBRID: cuRAND propagation + PTX weight processing
// =============================================================================

void gpu_bpf_step_async(GpuBpfState* s, float y_t) {
    int n = s->n_particles;
    int g = s->grid;
    int b = s->block;
    cudaStream_t st = s->stream;
    size_t smem = b * sizeof(float);

    // 1. Propagate + weight (cuRAND — nvcc compiled)
    bpf_propagate_weight_k<<<g, b, 0, st>>>(
        s->d_h, s->d_log_w, s->d_rng,
        s->rho, s->sigma_z, s->mu, s->nu_state, s->nu_obs, y_t,
        n, (s->timestep > 0) ? 1 : 0);

    // ════════════════════════════════════════════════════════════════════
    // From here on, ALL kernels are hand-written PTX via Driver API
    // ════════════════════════════════════════════════════════════════════

    // We need CUdeviceptr for the Driver API.
    // cudaMalloc returns pointers compatible with both Runtime and Driver API.
    // Cast: (CUdeviceptr)(uintptr_t)s->d_xxx
    CUdeviceptr dh    = (CUdeviceptr)(uintptr_t)s->d_h;
    CUdeviceptr dh2   = (CUdeviceptr)(uintptr_t)s->d_h2;
    CUdeviceptr dlw   = (CUdeviceptr)(uintptr_t)s->d_log_w;
    CUdeviceptr dw    = (CUdeviceptr)(uintptr_t)s->d_w;
    CUdeviceptr dcdf  = (CUdeviceptr)(uintptr_t)s->d_cdf;
    CUdeviceptr dwh   = (CUdeviceptr)(uintptr_t)s->d_wh;
    CUdeviceptr dscal = (CUdeviceptr)(uintptr_t)s->d_scalars;

    // 2. Max of log_w -> d_scalars[0]
    {
        CUdeviceptr ptr = dscal;  // scalars[0]
        float val = -1e30f;
        void* p1[] = { &ptr, &val };
        ptx_launch(g_ptx.set_scalar, st, 1, 1, 0, p1);

        void* p2[] = { &dlw, &ptr, &n };
        ptx_launch(g_ptx.reduce_max, st, g, b, smem, p2);
    }

    // 3. w = exp(log_w - max)
    {
        CUdeviceptr d_max = dscal;  // scalars[0]
        void* params[] = { &dw, &dlw, &d_max, &n };
        ptx_launch(g_ptx.exp_sub, st, g, b, 0, params);
    }

    // 4. Sum of w -> d_scalars[1]
    {
        CUdeviceptr ptr = dscal + sizeof(float);  // scalars[1]
        float val = 0.0f;
        void* p1[] = { &ptr, &val };
        ptx_launch(g_ptx.set_scalar, st, 1, 1, 0, p1);

        void* p2[] = { &dw, &ptr, &n };
        ptx_launch(g_ptx.reduce_sum, st, g, b, smem, p2);
    }

    // 5. Normalize + w*h
    {
        CUdeviceptr d_sum = dscal + sizeof(float);  // scalars[1]
        void* params[] = { &dw, &dwh, &dh, &d_sum, &n };
        ptx_launch(g_ptx.scale_wh, st, g, b, 0, params);
    }

    // 6. h_est = sum(w*h) -> d_scalars[2]
    {
        CUdeviceptr ptr = dscal + 2 * sizeof(float);  // scalars[2]
        float val = 0.0f;
        void* p1[] = { &ptr, &val };
        ptx_launch(g_ptx.set_scalar, st, 1, 1, 0, p1);

        void* p2[] = { &dwh, &ptr, &n };
        ptx_launch(g_ptx.reduce_sum, st, g, b, smem, p2);
    }

    // 7. Log-likelihood
    {
        void* params[] = { &dscal, &n };
        ptx_launch(g_ptx.compute_loglik, st, 1, 1, 0, params);
    }

    // 8. CDF (thrust prefix scan — still using thrust for now)
    {
        thrust::inclusive_scan(
            thrust::cuda::par.on(st),
            thrust::device_ptr<float>(s->d_w),
            thrust::device_ptr<float>(s->d_w + n),
            thrust::device_ptr<float>(s->d_cdf));
    }

    // 9. Resample (PTX)
    {
        float u = host_pcg32_float(&s->host_rng_state) / (float)n;
        void* params[] = { &dh2, &dh, &dcdf, &u, &n };
        ptx_launch(g_ptx.resample, st, g, b, 0, params);
    }

    // Swap particle buffers
    float* tmp = s->d_h; s->d_h = s->d_h2; s->d_h2 = tmp;

    s->timestep++;
}

// =============================================================================
// gpu_bpf_get_result / gpu_bpf_step
// =============================================================================

BpfResult gpu_bpf_get_result(GpuBpfState* s) {
    float scalars[4];
    cudaMemcpy(scalars, s->d_scalars, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    BpfResult r;
    r.h_mean  = scalars[2];
    r.log_lik = scalars[3];
    return r;
}

BpfResult gpu_bpf_step(GpuBpfState* s, float y_t) {
    gpu_bpf_step_async(s, y_t);
    cudaStreamSynchronize(s->stream);
    return gpu_bpf_get_result(s);
}

// =============================================================================
// gpu_bpf_run_rmse — batch convenience
// =============================================================================

double gpu_bpf_run_rmse(
    const double* returns, const double* true_h, int n_ticks,
    int n_particles, float rho, float sigma_z, float mu,
    float nu_state, float nu_obs, int seed
) {
    GpuBpfState* f = gpu_bpf_create(n_particles, rho, sigma_z, mu, nu_state, nu_obs, seed);
    double sse = 0.0;
    int skip = 100;
    int count = 0;
    for (int t = 0; t < n_ticks; t++) {
        BpfResult r = gpu_bpf_step(f, (float)returns[t]);
        if (t >= skip) {
            double err = (double)r.h_mean - true_h[t];
            sse += err * err;
            count++;
        }
    }
    gpu_bpf_destroy(f);
    return sqrt(sse / (double)count);
}


// =============================================================================
// nvcc-compiled utility kernels (needed by APF which doesn't use PTX)
// =============================================================================

__global__ void bpf_set_scalar_k(float* scalar, float val) { *scalar = val; }

__global__ void bpf_reduce_max_k(const float* in, float* out, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? in[i] : -1e30f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0) {
        unsigned int* addr = (unsigned int*)out;
        unsigned int old = __float_as_uint(*out);
        unsigned int assumed;
        do {
            assumed = old;
            old = atomicCAS(addr, assumed,
                __float_as_uint(fmaxf(__uint_as_float(assumed), sdata[0])));
        } while (old != assumed);
    }
}

__global__ void bpf_reduce_sum_k(const float* in, float* out, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}

__global__ void bpf_exp_sub_k(float* w, const float* log_w, const float* d_max, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) w[i] = __expf(log_w[i] - *d_max);
}

__global__ void bpf_scale_wh_k(float* w, float* wh, const float* h,
                                const float* d_sum, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float inv = 1.0f / *d_sum;
    w[i] *= inv;
    wh[i] = w[i] * h[i];
}

__global__ void bpf_compute_loglik_k(float* d_scalars, int n) {
    float ratio = fmaxf(d_scalars[1] / (float)n, 1e-30f);
    d_scalars[3] = d_scalars[0] + logf(ratio);
}

__global__ void bpf_resample_k(float* h_out, const float* h_in,
                                const float* cdf, float u_base, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float target = u_base + (float)i / (float)n;
    if (target >= 1.0f) target -= 1.0f;
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (cdf[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    h_out[i] = h_in[lo];
}

// =============================================================================
// APF Kernels (nvcc compiled, use cuRAND)
// =============================================================================

__global__ void apf_first_stage(
    const float* h, float* mu_pred, float* log_v,
    float rho, float mu, float nu_obs, float y_t, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float mp = mu + rho * (h[i] - mu);
    mu_pred[i] = mp;
    float eta = y_t * __expf(-mp * 0.5f);
    log_v[i] = (nu_obs > 0.0f)
        ? bpf_log_t_pdf(eta, nu_obs) - mp * 0.5f
        : -0.9189385f - 0.5f * eta * eta - mp * 0.5f;
}

__global__ void apf_resample_pair(
    float* h_out, float* mu_out,
    const float* h_in, const float* mu_in,
    const float* cdf, float u_base, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float target = u_base + (float)i / (float)n;
    if (target >= 1.0f) target -= 1.0f;
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (cdf[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    h_out[i] = h_in[lo];
    mu_out[i] = mu_in[lo];
}

__global__ void apf_propagate_correct(
    float* h, const float* mu_pred_resampled, float* log_w,
    curandState* states,
    float rho, float sigma_z, float mu,
    float nu_state, float nu_obs, float y_t, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float eps = bpf_sample_t(&states[i], nu_state);
    h[i] = mu + rho * (h[i] - mu) + sigma_z * eps;
    float h_new = h[i];
    float mp = mu_pred_resampled[i];
    float eta_new = y_t * __expf(-h_new * 0.5f);
    float eta_pred = y_t * __expf(-mp * 0.5f);
    float log_p_new, log_p_pred;
    if (nu_obs > 0.0f) {
        log_p_new  = bpf_log_t_pdf(eta_new, nu_obs) - h_new * 0.5f;
        log_p_pred = bpf_log_t_pdf(eta_pred, nu_obs) - mp * 0.5f;
    } else {
        log_p_new  = -0.9189385f - 0.5f * eta_new * eta_new - h_new * 0.5f;
        log_p_pred = -0.9189385f - 0.5f * eta_pred * eta_pred - mp * 0.5f;
    }
    log_w[i] = log_p_new - log_p_pred;
}

// =============================================================================
// APF Create / Destroy
// =============================================================================

GpuApfState* gpu_apf_create(int n_particles, float rho, float sigma_z, float mu,
                              float nu_state, float nu_obs, int seed) {
    GpuApfState* s = (GpuApfState*)calloc(1, sizeof(GpuApfState));
    s->n_particles = n_particles;
    s->rho = rho; s->sigma_z = sigma_z; s->mu = mu;
    s->nu_state = nu_state; s->nu_obs = nu_obs;
    s->block = 256;
    s->grid = (n_particles + s->block - 1) / s->block;
    s->host_rng_state = (unsigned long long)seed * 67890ULL + 12345ULL;
    s->timestep = 0;

    cudaStreamCreate(&s->stream);
    cudaMalloc(&s->d_h,        n_particles * sizeof(float));
    cudaMalloc(&s->d_h2,       n_particles * sizeof(float));
    cudaMalloc(&s->d_mu_pred,  n_particles * sizeof(float));
    cudaMalloc(&s->d_mu_pred2, n_particles * sizeof(float));
    cudaMalloc(&s->d_log_v,    n_particles * sizeof(float));
    cudaMalloc(&s->d_v,        n_particles * sizeof(float));
    cudaMalloc(&s->d_log_w,    n_particles * sizeof(float));
    cudaMalloc(&s->d_w,        n_particles * sizeof(float));
    cudaMalloc(&s->d_cdf,      n_particles * sizeof(float));
    cudaMalloc(&s->d_wh,       n_particles * sizeof(float));
    cudaMalloc(&s->d_rng,      n_particles * sizeof(curandState));
    cudaMalloc(&s->d_scalars,  4 * sizeof(float));

    bpf_init_rng_k<<<s->grid, s->block, 0, s->stream>>>(
        s->d_rng, (unsigned long long)seed, n_particles);
    float std_stat = sqrtf((sigma_z * sigma_z) / fmaxf(1.0f - rho * rho, 1e-6f));
    bpf_init_particles_k<<<s->grid, s->block, 0, s->stream>>>(
        s->d_h, s->d_rng, mu, std_stat, n_particles);
    cudaStreamSynchronize(s->stream);
    return s;
}

void gpu_apf_destroy(GpuApfState* s) {
    if (!s) return;
    cudaStreamDestroy(s->stream);
    cudaFree(s->d_h); cudaFree(s->d_h2);
    cudaFree(s->d_mu_pred); cudaFree(s->d_mu_pred2);
    cudaFree(s->d_log_v); cudaFree(s->d_v);
    cudaFree(s->d_log_w); cudaFree(s->d_w);
    cudaFree(s->d_cdf); cudaFree(s->d_wh);
    cudaFree(s->d_rng); cudaFree(s->d_scalars);
    free(s);
}

static BpfResult apf_read_scalars(float* d_scalars) {
    float scalars[4];
    cudaMemcpy(scalars, d_scalars, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    BpfResult r;
    r.h_mean = scalars[2]; r.log_lik = scalars[3];
    return r;
}

BpfResult gpu_apf_step(GpuApfState* s, float y_t) {
    int n = s->n_particles, g = s->grid, b = s->block;
    cudaStream_t st = s->stream;
    size_t smem = b * sizeof(float);

    if (s->timestep == 0) {
        bpf_propagate_weight_k<<<g, b, 0, st>>>(
            s->d_h, s->d_log_w, s->d_rng,
            s->rho, s->sigma_z, s->mu, s->nu_state, s->nu_obs, y_t, n, 0);
        bpf_set_scalar_k<<<1, 1, 0, st>>>(s->d_scalars + 0, -1e30f);
        bpf_reduce_max_k<<<g, b, smem, st>>>(s->d_log_w, s->d_scalars + 0, n);
        bpf_exp_sub_k<<<g, b, 0, st>>>(s->d_w, s->d_log_w, s->d_scalars + 0, n);
        bpf_set_scalar_k<<<1, 1, 0, st>>>(s->d_scalars + 1, 0.0f);
        bpf_reduce_sum_k<<<g, b, smem, st>>>(s->d_w, s->d_scalars + 1, n);
        bpf_scale_wh_k<<<g, b, 0, st>>>(s->d_w, s->d_wh, s->d_h, s->d_scalars + 1, n);
        bpf_set_scalar_k<<<1, 1, 0, st>>>(s->d_scalars + 2, 0.0f);
        bpf_reduce_sum_k<<<g, b, smem, st>>>(s->d_wh, s->d_scalars + 2, n);
        bpf_compute_loglik_k<<<1, 1, 0, st>>>(s->d_scalars, n);
        thrust::inclusive_scan(thrust::cuda::par.on(st),
            thrust::device_ptr<float>(s->d_w),
            thrust::device_ptr<float>(s->d_w + n),
            thrust::device_ptr<float>(s->d_cdf));
        float u = host_pcg32_float(&s->host_rng_state) / (float)n;
        bpf_resample_k<<<g, b, 0, st>>>(s->d_h2, s->d_h, s->d_cdf, u, n);
        float* tmp = s->d_h; s->d_h = s->d_h2; s->d_h2 = tmp;
        cudaStreamSynchronize(st);
        s->timestep++;
        return apf_read_scalars(s->d_scalars);
    }

    // 1. First stage
    apf_first_stage<<<g, b, 0, st>>>(
        s->d_h, s->d_mu_pred, s->d_log_v, s->rho, s->mu, s->nu_obs, y_t, n);
    bpf_set_scalar_k<<<1, 1, 0, st>>>(s->d_scalars + 0, -1e30f);
    bpf_reduce_max_k<<<g, b, smem, st>>>(s->d_log_v, s->d_scalars + 0, n);
    bpf_exp_sub_k<<<g, b, 0, st>>>(s->d_v, s->d_log_v, s->d_scalars + 0, n);
    bpf_set_scalar_k<<<1, 1, 0, st>>>(s->d_scalars + 1, 0.0f);
    bpf_reduce_sum_k<<<g, b, smem, st>>>(s->d_v, s->d_scalars + 1, n);
    bpf_scale_wh_k<<<g, b, 0, st>>>(s->d_v, s->d_wh, s->d_v, s->d_scalars + 1, n);
    thrust::inclusive_scan(thrust::cuda::par.on(st),
        thrust::device_ptr<float>(s->d_v),
        thrust::device_ptr<float>(s->d_v + n),
        thrust::device_ptr<float>(s->d_cdf));

    // 2. Resample pair
    float u = host_pcg32_float(&s->host_rng_state) / (float)n;
    apf_resample_pair<<<g, b, 0, st>>>(
        s->d_h2, s->d_mu_pred2, s->d_h, s->d_mu_pred, s->d_cdf, u, n);
    float* tmp = s->d_h; s->d_h = s->d_h2; s->d_h2 = tmp;

    // 3. Propagate + correct
    apf_propagate_correct<<<g, b, 0, st>>>(
        s->d_h, s->d_mu_pred2, s->d_log_w, s->d_rng,
        s->rho, s->sigma_z, s->mu, s->nu_state, s->nu_obs, y_t, n);

    // 4. Second-stage normalization
    bpf_set_scalar_k<<<1, 1, 0, st>>>(s->d_scalars + 0, -1e30f);
    bpf_reduce_max_k<<<g, b, smem, st>>>(s->d_log_w, s->d_scalars + 0, n);
    bpf_exp_sub_k<<<g, b, 0, st>>>(s->d_w, s->d_log_w, s->d_scalars + 0, n);
    bpf_set_scalar_k<<<1, 1, 0, st>>>(s->d_scalars + 1, 0.0f);
    bpf_reduce_sum_k<<<g, b, smem, st>>>(s->d_w, s->d_scalars + 1, n);
    bpf_scale_wh_k<<<g, b, 0, st>>>(s->d_w, s->d_wh, s->d_h, s->d_scalars + 1, n);
    bpf_set_scalar_k<<<1, 1, 0, st>>>(s->d_scalars + 2, 0.0f);
    bpf_reduce_sum_k<<<g, b, smem, st>>>(s->d_wh, s->d_scalars + 2, n);
    bpf_compute_loglik_k<<<1, 1, 0, st>>>(s->d_scalars, n);

    // 5. Resample for next step
    thrust::inclusive_scan(thrust::cuda::par.on(st),
        thrust::device_ptr<float>(s->d_w),
        thrust::device_ptr<float>(s->d_w + n),
        thrust::device_ptr<float>(s->d_cdf));
    u = host_pcg32_float(&s->host_rng_state) / (float)n;
    bpf_resample_k<<<g, b, 0, st>>>(s->d_h2, s->d_h, s->d_cdf, u, n);
    tmp = s->d_h; s->d_h = s->d_h2; s->d_h2 = tmp;

    cudaStreamSynchronize(st);
    s->timestep++;
    return apf_read_scalars(s->d_scalars);
}

double gpu_apf_run_rmse(
    const double* returns, const double* true_h, int n_ticks,
    int n_particles, float rho, float sigma_z, float mu,
    float nu_state, float nu_obs, int seed
) {
    GpuApfState* state = gpu_apf_create(n_particles, rho, sigma_z, mu, nu_state, nu_obs, seed);
    int skip = 100; double sum_sq = 0.0; int count = 0;
    for (int t = 0; t < n_ticks; t++) {
        BpfResult r = gpu_apf_step(state, (float)returns[t]);
        if (t >= skip) { double e = (double)r.h_mean - true_h[t]; sum_sq += e*e; count++; }
    }
    gpu_apf_destroy(state);
    return sqrt(sum_sq / count);
}

// =============================================================================
// IMM (uses gpu_bpf_* which now goes through PTX for BPF)
// =============================================================================

static double log_sum_exp(const double* x, int n) {
    double mx = -1e30;
    for (int i = 0; i < n; i++) if (x[i] > mx) mx = x[i];
    double s = 0.0;
    for (int i = 0; i < n; i++) s += exp(x[i] - mx);
    return mx + log(s);
}

GpuImmState* gpu_imm_create(
    const ImmModelParams* models, int n_models, int n_particles_per_model,
    const float* transition_matrix, int seed
) {
    if (n_models > IMM_MAX_MODELS) {
        fprintf(stderr, "IMM: n_models=%d > max=%d\n", n_models, IMM_MAX_MODELS);
        return NULL;
    }
    GpuImmState* s = (GpuImmState*)calloc(1, sizeof(GpuImmState));
    s->n_models = n_models;
    s->n_particles_per_model = n_particles_per_model;
    s->timestep = 0;
    s->filters = (GpuBpfState**)malloc(n_models * sizeof(GpuBpfState*));
    for (int k = 0; k < n_models; k++)
        s->filters[k] = gpu_bpf_create(n_particles_per_model,
            models[k].rho, models[k].sigma_z, models[k].mu,
            models[k].nu_state, models[k].nu_obs, seed + k * 7919);
    s->log_pi = (double*)malloc(n_models * sizeof(double));
    s->log_pi_pred = (double*)malloc(n_models * sizeof(double));
    double log_u = -log((double)n_models);
    for (int k = 0; k < n_models; k++) s->log_pi[k] = log_u;
    s->log_T = (double*)malloc(n_models * n_models * sizeof(double));
    if (transition_matrix) {
        for (int i = 0; i < n_models * n_models; i++)
            s->log_T[i] = log(fmax((double)transition_matrix[i], 1e-30));
    } else {
        double p_stay = 0.95;
        double p_sw = (n_models > 1) ? (1.0 - p_stay) / (n_models - 1) : 0.0;
        for (int i = 0; i < n_models; i++)
            for (int j = 0; j < n_models; j++)
                s->log_T[i * n_models + j] = log(fmax((i == j) ? p_stay : p_sw, 1e-30));
    }
    return s;
}

void gpu_imm_destroy(GpuImmState* s) {
    if (!s) return;
    for (int k = 0; k < s->n_models; k++) gpu_bpf_destroy(s->filters[k]);
    free(s->filters); free(s->log_pi); free(s->log_pi_pred); free(s->log_T); free(s);
}

ImmResult gpu_imm_step(GpuImmState* s, float y_t) {
    int K = s->n_models;
    for (int k = 0; k < K; k++) {
        double terms[IMM_MAX_MODELS];
        for (int j = 0; j < K; j++) terms[j] = s->log_T[j * K + k] + s->log_pi[j];
        s->log_pi_pred[k] = log_sum_exp(terms, K);
    }
    for (int k = 0; k < K; k++) gpu_bpf_step_async(s->filters[k], y_t);
    cudaDeviceSynchronize();
    double log_liks[IMM_MAX_MODELS]; float h_ests[IMM_MAX_MODELS];
    for (int k = 0; k < K; k++) {
        BpfResult r = gpu_bpf_get_result(s->filters[k]);
        h_ests[k] = r.h_mean; log_liks[k] = (double)r.log_lik;
    }
    double log_joint[IMM_MAX_MODELS];
    for (int k = 0; k < K; k++) log_joint[k] = s->log_pi_pred[k] + log_liks[k];
    double log_Z = log_sum_exp(log_joint, K);
    for (int k = 0; k < K; k++) s->log_pi[k] = log_joint[k] - log_Z;
    double h_mixed = 0.0; int best_k = 0; double best_lp = -1e30;
    for (int k = 0; k < K; k++) {
        h_mixed += exp(s->log_pi[k]) * (double)h_ests[k];
        if (s->log_pi[k] > best_lp) { best_lp = s->log_pi[k]; best_k = k; }
    }
    s->timestep++;
    ImmResult r;
    r.h_mean = (float)h_mixed; r.vol = expf((float)h_mixed * 0.5f);
    r.log_lik = (float)log_Z; r.best_model = best_k; r.best_prob = (float)exp(best_lp);
    return r;
}

float gpu_imm_get_prob(const GpuImmState* s, int k) {
    return (k >= 0 && k < s->n_models) ? (float)exp(s->log_pi[k]) : 0.0f;
}
void gpu_imm_get_probs(const GpuImmState* s, float* out) {
    for (int k = 0; k < s->n_models; k++) out[k] = (float)exp(s->log_pi[k]);
}

ImmModelParams* gpu_imm_build_grid(
    const float* rhos, int n_rho, const float* sigma_zs, int n_sigma,
    const float* mus, int n_mu, float nu_state, float nu_obs, int* out_n
) {
    int total = n_rho * n_sigma * n_mu;
    ImmModelParams* g = (ImmModelParams*)malloc(total * sizeof(ImmModelParams));
    int idx = 0;
    for (int r = 0; r < n_rho; r++)
        for (int s = 0; s < n_sigma; s++)
            for (int m = 0; m < n_mu; m++) {
                g[idx].rho = rhos[r]; g[idx].sigma_z = sigma_zs[s];
                g[idx].mu = mus[m]; g[idx].nu_state = nu_state;
                g[idx].nu_obs = nu_obs; idx++;
            }
    *out_n = total;
    return g;
}
