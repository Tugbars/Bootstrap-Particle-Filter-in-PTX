/**
 * @file gpu_bpf_ptx_full.cu
 * @brief GPU BPF — 100% hand-written PTX, zero cuRAND for BPF
 *
 * All 13 BPF kernels from bpf_kernels_full.ptx:
 *   1.  bpf_init_rng          — PCG32 seeding
 *   2.  bpf_init_particles    — Draw N(mu, sigma_stat^2) via ICDF
 *   3.  bpf_propagate_weight  — OU + Student-t + obs weight (fused)
 *   4.  bpf_set_scalar
 *   5.  bpf_reduce_max
 *   6.  bpf_reduce_sum
 *   7.  bpf_exp_sub
 *   8.  bpf_scale_wh
 *   9.  bpf_compute_loglik
 *  10.  bpf_resample
 *  11.  bpf_compute_var
 *  12.  bpf_gen_noise
 *  13.  bpf_silverman_jitter
 *
 * PCG32 state: 16 bytes/particle (2 x u64) vs curandState ~48 bytes/particle.
 * Normal generation: ICDF (Acklam rational approx) — 1 uniform per normal.
 *
 * APF/IMM still nvcc + cuRAND.
 *
 * Build:
 *   nvcc -O3 test_bpf_matched_dgp.cu gpu_bpf_ptx_full.cu -o test_bpf_full -lcuda -lcurand
 *
 * Runtime: needs bpf_kernels_full.cubin or bpf_kernels_full.ptx in cwd
 */

#include "gpu_bpf_full.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/system/cuda/execution_policy.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// =============================================================================
// Chi2 square-sum kernel: chi2[i] = sum(normals[i*nu .. (i+1)*nu - 1]^2)
// =============================================================================

__global__ void chi2_square_sum_k(float* __restrict__ chi2,
                                   const float* __restrict__ normals,
                                   int nu, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float sum = 0.0f;
    const float* base = normals + i * nu;
    for (int k = 0; k < nu; k++) {
        float z = base[k];
        sum += z * z;
    }
    chi2[i] = sum;
}

// =============================================================================
// PTX Module — 13 kernels
// =============================================================================

static CUmodule g_ptx_module = NULL;

typedef struct {
    CUfunction init_rng;
    CUfunction init_particles;
    CUfunction propagate_weight;
    CUfunction set_scalar;
    CUfunction reduce_max;
    CUfunction reduce_sum;
    CUfunction exp_sub;
    CUfunction scale_wh;
    CUfunction compute_loglik;
    CUfunction resample;
    CUfunction compute_var;
    CUfunction gen_noise;
    CUfunction silverman_jitter;
} PtxFunctions;

static PtxFunctions g_ptx;
static int          g_ptx_loaded = 0;

static void ensure_ptx_loaded() {
    if (g_ptx_loaded) return;
    cudaFree(0);

    const char* cubin_paths[] = { "bpf_kernels_full.cubin", NULL };
    const char* ptx_paths[]   = { "bpf_kernels_full.ptx", NULL };

    for (int i = 0; cubin_paths[i]; i++) {
        FILE* f = fopen(cubin_paths[i], "rb");
        if (!f) continue;
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        char* buf = (char*)malloc(sz);
        fread(buf, 1, sz, f);
        fclose(f);
        CUresult err = cuModuleLoadData(&g_ptx_module, buf);
        free(buf);
        if (err == CUDA_SUCCESS) {
            fprintf(stderr, "[PTX-FULL] Loaded cubin: %s (%ld bytes)\n", cubin_paths[i], sz);
            goto extract;
        }
    }

    {
        char* src = NULL;
        for (int i = 0; ptx_paths[i]; i++) {
            FILE* f = fopen(ptx_paths[i], "rb");
            if (!f) continue;
            fseek(f, 0, SEEK_END);
            long sz = ftell(f);
            fseek(f, 0, SEEK_SET);
            src = (char*)malloc(sz + 1);
            fread(src, 1, sz, f);
            src[sz] = '\0';
            fclose(f);
            fprintf(stderr, "[PTX-FULL] Loaded source: %s (%ld bytes)\n", ptx_paths[i], sz);
            break;
        }
        if (!src) {
            fprintf(stderr, "[PTX-FULL] ERROR: Cannot find bpf_kernels_full.cubin or .ptx\n");
            exit(1);
        }

        char jit_err[4096] = {0};
        CUjit_option opts[] = {
            CU_JIT_ERROR_LOG_BUFFER,
            CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
        };
        void* vals[] = {
            (void*)jit_err,
            (void*)(uintptr_t)sizeof(jit_err)
        };
        CUresult err = cuModuleLoadDataEx(&g_ptx_module, src, 2, opts, vals);
        free(src);
        if (err != CUDA_SUCCESS) {
            const char* msg;
            cuGetErrorString(err, &msg);
            fprintf(stderr, "[PTX-FULL] JIT failed: %s\n", msg);
            if (jit_err[0]) fprintf(stderr, "[PTX-FULL] %s\n", jit_err);
            exit(1);
        }
    }

extract:
    cuModuleGetFunction(&g_ptx.init_rng,         g_ptx_module, "bpf_init_rng");
    cuModuleGetFunction(&g_ptx.init_particles,    g_ptx_module, "bpf_init_particles");
    cuModuleGetFunction(&g_ptx.propagate_weight,  g_ptx_module, "bpf_propagate_weight");
    cuModuleGetFunction(&g_ptx.set_scalar,        g_ptx_module, "bpf_set_scalar");
    cuModuleGetFunction(&g_ptx.reduce_max,        g_ptx_module, "bpf_reduce_max");
    cuModuleGetFunction(&g_ptx.reduce_sum,        g_ptx_module, "bpf_reduce_sum");
    cuModuleGetFunction(&g_ptx.exp_sub,           g_ptx_module, "bpf_exp_sub");
    cuModuleGetFunction(&g_ptx.scale_wh,          g_ptx_module, "bpf_scale_wh");
    cuModuleGetFunction(&g_ptx.compute_loglik,    g_ptx_module, "bpf_compute_loglik");
    cuModuleGetFunction(&g_ptx.resample,          g_ptx_module, "bpf_resample");
    cuModuleGetFunction(&g_ptx.compute_var,       g_ptx_module, "bpf_compute_var");
    cuModuleGetFunction(&g_ptx.gen_noise,         g_ptx_module, "bpf_gen_noise");
    cuModuleGetFunction(&g_ptx.silverman_jitter,  g_ptx_module, "bpf_silverman_jitter");

    g_ptx_loaded = 1;
    fprintf(stderr, "[PTX-FULL] All 13 kernels loaded\n");
}

// =============================================================================
// Driver API launch helper
// =============================================================================

static inline void ptx_launch(
    CUfunction func, cudaStream_t stream,
    unsigned int gridX, unsigned int blockX,
    unsigned int smem, void** params
) {
    CUresult err = cuLaunchKernel(func,
        gridX, 1, 1, blockX, 1, 1,
        smem, (CUstream)stream, params, NULL);
    if (err != CUDA_SUCCESS) {
        const char* msg;
        cuGetErrorString(err, &msg);
        fprintf(stderr, "[PTX-FULL] Launch failed: %s\n", msg);
    }
}

// =============================================================================
// Host PCG32 (for systematic resampling U)
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
// gpu_bpf_create — ALL PTX
// =============================================================================

GpuBpfState* gpu_bpf_create(int n_particles, float rho, float sigma_z, float mu,
                              float nu_state, float nu_obs, int seed) {
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
    s->silverman_shrink = 0.0f;

    cudaStreamCreate(&s->stream);

    cudaMalloc(&s->d_h,       n_particles * sizeof(float));
    cudaMalloc(&s->d_h2,      n_particles * sizeof(float));
    cudaMalloc(&s->d_log_w,   n_particles * sizeof(float));
    cudaMalloc(&s->d_w,       n_particles * sizeof(float));
    cudaMalloc(&s->d_cdf,     n_particles * sizeof(float));
    cudaMalloc(&s->d_wh,      n_particles * sizeof(float));
    // PCG32: 2 x u64 per particle = 16 bytes each
    cudaMalloc(&s->d_rng,     n_particles * 2 * sizeof(uint64_t));
    cudaMalloc(&s->d_scalars, 4 * sizeof(float));
    cudaMalloc(&s->d_noise,   n_particles * sizeof(float));
    cudaMalloc(&s->d_var,     sizeof(float));

    // Chi2 pre-generation for Student-t state noise
    s->nu_int = (nu_state > 0.0f) ? (int)(nu_state + 0.5f) : 0;
    s->d_chi2 = NULL;
    s->d_chi2_normals = NULL;
    s->curand_gen = NULL;
    if (s->nu_int > 0) {
        cudaMalloc(&s->d_chi2, n_particles * sizeof(float));
        cudaMalloc(&s->d_chi2_normals, (size_t)n_particles * s->nu_int * sizeof(float));
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long)seed * 31337ULL);
        curandSetStream(gen, s->stream);
        s->curand_gen = (void*)gen;
    }

    // Precompute Student-t observation constant: lgamma((nu+1)/2) - lgamma(nu/2) - 0.5*log(nu*pi)
    if (nu_obs > 0.0f) {
        s->C_obs = (float)(lgamma((nu_obs + 1.0) / 2.0) - lgamma(nu_obs / 2.0)
                           - 0.5 * log(nu_obs * 3.14159265358979323846));
    } else {
        s->C_obs = 0.0f;
    }

    int g = s->grid, b = s->block;
    cudaStream_t st = s->stream;

    // PTX: init_rng(u64 rng, u64 seed, s32 n)
    {
        CUdeviceptr drng = (CUdeviceptr)(uintptr_t)s->d_rng;
        unsigned long long seed64 = (unsigned long long)seed;
        int n = n_particles;
        void* params[] = { &drng, &seed64, &n };
        ptx_launch(g_ptx.init_rng, st, g, b, 0, params);
    }

    // PTX: init_particles(u64 h, u64 rng, f32 mu, f32 std, s32 n)
    {
        CUdeviceptr dh   = (CUdeviceptr)(uintptr_t)s->d_h;
        CUdeviceptr drng = (CUdeviceptr)(uintptr_t)s->d_rng;
        float std_stat = sqrtf((sigma_z * sigma_z) / fmaxf(1.0f - rho * rho, 1e-6f));
        int n = n_particles;
        void* params[] = { &dh, &drng, &mu, &std_stat, &n };
        ptx_launch(g_ptx.init_particles, st, g, b, 0, params);
    }

    cudaStreamSynchronize(st);
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
    if (s->d_chi2) cudaFree(s->d_chi2);
    if (s->d_chi2_normals) cudaFree(s->d_chi2_normals);
    if (s->curand_gen) curandDestroyGenerator((curandGenerator_t)s->curand_gen);
    free(s);
}

// =============================================================================
// gpu_bpf_step_async — ALL PTX (13 kernels)
// =============================================================================

void gpu_bpf_step_async(GpuBpfState* s, float y_t) {
    int n = s->n_particles;
    int g = s->grid;
    int b = s->block;
    cudaStream_t st = s->stream;
    size_t smem = b * sizeof(float);

    CUdeviceptr dh    = (CUdeviceptr)(uintptr_t)s->d_h;
    CUdeviceptr dh2   = (CUdeviceptr)(uintptr_t)s->d_h2;
    CUdeviceptr dlw   = (CUdeviceptr)(uintptr_t)s->d_log_w;
    CUdeviceptr dw    = (CUdeviceptr)(uintptr_t)s->d_w;
    CUdeviceptr dcdf  = (CUdeviceptr)(uintptr_t)s->d_cdf;
    CUdeviceptr dwh   = (CUdeviceptr)(uintptr_t)s->d_wh;
    CUdeviceptr drng  = (CUdeviceptr)(uintptr_t)s->d_rng;
    CUdeviceptr dscal = (CUdeviceptr)(uintptr_t)s->d_scalars;
    CUdeviceptr dnoise= (CUdeviceptr)(uintptr_t)s->d_noise;
    CUdeviceptr dvar  = (CUdeviceptr)(uintptr_t)s->d_var;

    // 1. Generate chi2 variates if Student-t state noise
    CUdeviceptr dchi2 = (CUdeviceptr)(uintptr_t)s->d_chi2;
    if (s->nu_int > 0 && s->timestep > 0) {
        curandGenerator_t gen = (curandGenerator_t)s->curand_gen;
        curandGenerateNormal(gen, s->d_chi2_normals,
                             (size_t)n * s->nu_int, 0.0f, 1.0f);
        int chi_g = (n + b - 1) / b;
        chi2_square_sum_k<<<chi_g, b, 0, st>>>(s->d_chi2, s->d_chi2_normals,
                                                  s->nu_int, n);
    }

    // 2. Propagate + weight (PTX: fused PCG32 + ICDF + OU + obs weight)
    //    propagate_weight(u64 h, u64 log_w, u64 rng, u64 chi2,
    //                     f32 rho, f32 sigma_z, f32 mu, f32 nu_state, f32 nu_obs,
    //                     f32 C_obs, f32 y_t, s32 n, s32 do_prop)
    {
        int do_prop = (s->timestep > 0) ? 1 : 0;
        void* params[] = {
            &dh, &dlw, &drng, &dchi2,
            &s->rho, &s->sigma_z, &s->mu, &s->nu_state, &s->nu_obs,
            &s->C_obs, &y_t, &n, &do_prop
        };
        ptx_launch(g_ptx.propagate_weight, st, g, b, 0, params);
    }

    // 2. Max of log_w -> d_scalars[0]
    {
        CUdeviceptr ptr = dscal;
        float val = -1e30f;
        void* p1[] = { &ptr, &val };
        ptx_launch(g_ptx.set_scalar, st, 1, 1, 0, p1);
        void* p2[] = { &dlw, &ptr, &n };
        ptx_launch(g_ptx.reduce_max, st, g, b, smem, p2);
    }

    // 3. w = exp(log_w - max)
    {
        CUdeviceptr d_max = dscal;
        void* params[] = { &dw, &dlw, &d_max, &n };
        ptx_launch(g_ptx.exp_sub, st, g, b, 0, params);
    }

    // 4. Sum of w -> d_scalars[1]
    {
        CUdeviceptr ptr = dscal + sizeof(float);
        float val = 0.0f;
        void* p1[] = { &ptr, &val };
        ptx_launch(g_ptx.set_scalar, st, 1, 1, 0, p1);
        void* p2[] = { &dw, &ptr, &n };
        ptx_launch(g_ptx.reduce_sum, st, g, b, smem, p2);
    }

    // 5. Normalize + w*h
    {
        CUdeviceptr d_sum = dscal + sizeof(float);
        void* params[] = { &dw, &dwh, &dh, &d_sum, &n };
        ptx_launch(g_ptx.scale_wh, st, g, b, 0, params);
    }

    // 6. h_est = sum(w*h) -> d_scalars[2]
    {
        CUdeviceptr ptr = dscal + 2 * sizeof(float);
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

    // 8. CDF (thrust prefix scan)
    {
        thrust::inclusive_scan(
            thrust::cuda::par.on(st),
            thrust::device_ptr<float>(s->d_w),
            thrust::device_ptr<float>(s->d_w + n),
            thrust::device_ptr<float>(s->d_cdf));
    }

    // 9. Resample
    {
        float u = host_pcg32_float(&s->host_rng_state) / (float)n;
        void* params[] = { &dh2, &dh, &dcdf, &u, &n };
        ptx_launch(g_ptx.resample, st, g, b, 0, params);
    }

    // Swap buffers
    float* tmp = s->d_h; s->d_h = s->d_h2; s->d_h2 = tmp;

    // 10. Silverman jitter (PTX: var + noise + jitter)
    if (s->silverman_shrink > 0.0f) {
        // Need h_mean from step 6
        cudaStreamSynchronize(st);
        float h_mean;
        cudaMemcpy(&h_mean, s->d_scalars + 2, sizeof(float), cudaMemcpyDeviceToHost);

        // Recompute dh after swap
        dh = (CUdeviceptr)(uintptr_t)s->d_h;

        // 10a. Variance
        {
            float zero = 0.0f;
            void* p1[] = { &dvar, &zero };
            ptx_launch(g_ptx.set_scalar, st, 1, 1, 0, p1);
            void* p2[] = { &dh, &dvar, &h_mean, &n };
            ptx_launch(g_ptx.compute_var, st, g, b, smem, p2);
        }

        cudaStreamSynchronize(st);
        float var_sum;
        cudaMemcpy(&var_sum, s->d_var, sizeof(float), cudaMemcpyDeviceToHost);
        float sigma_hat = sqrtf(var_sum / (float)n);
        float bw = s->silverman_shrink * sigma_hat * 1.05922f * powf((float)n, -0.2f);

        // 10b. Generate noise (PTX: PCG32 + ICDF)
        {
            CUdeviceptr drng2 = (CUdeviceptr)(uintptr_t)s->d_rng;
            void* params[] = { &dnoise, &drng2, &n };
            ptx_launch(g_ptx.gen_noise, st, g, b, 0, params);
        }

        // 10c. Jitter
        {
            void* params[] = { &dh, &dnoise, &bw, &n, &s->mu };
            ptx_launch(g_ptx.silverman_jitter, st, g, b, 0, params);
        }
    }

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
// gpu_bpf_run_rmse
// =============================================================================

double gpu_bpf_run_rmse(
    const double* returns, const double* true_h, int n_ticks,
    int n_particles, float rho, float sigma_z, float mu,
    float nu_state, float nu_obs, int seed
) {
    GpuBpfState* f = gpu_bpf_create(n_particles, rho, sigma_z, mu, nu_state, nu_obs, seed);
    double sse = 0.0;
    int skip = 100, count = 0;
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
// APF — nvcc compiled (cuRAND), same as original
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
    float nu_h = nu * 0.5f, nup1h = (nu + 1.0f) * 0.5f;
    return lgammaf(nup1h) - lgammaf(nu_h)
         - 0.5f * logf(nu * 3.14159265f)
         - nup1h * logf(1.0f + (x * x) / nu);
}

__global__ void apf_init_rng_k(curandState* states, unsigned long long seed, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) curand_init(seed, i, 0, &states[i]);
}

__global__ void apf_init_particles_k(float* h, curandState* states,
                                      float mu, float std_stat, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = mu + std_stat * curand_normal(&states[i]);
}

// APF utility kernels (nvcc compiled)
__global__ void apf_set_scalar_k(float* scalar, float val) { *scalar = val; }

__global__ void apf_reduce_max_k(const float* in, float* out, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x, i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? in[i] : -1e30f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0) {
        unsigned int* addr = (unsigned int*)out;
        unsigned int old = __float_as_uint(*out), assumed;
        do { assumed = old;
             old = atomicCAS(addr, assumed,
                 __float_as_uint(fmaxf(__uint_as_float(assumed), sdata[0])));
        } while (old != assumed);
    }
}

__global__ void apf_reduce_sum_k(const float* in, float* out, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x, i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}

__global__ void apf_exp_sub_k(float* w, const float* log_w, const float* d_max, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) w[i] = __expf(log_w[i] - *d_max);
}

__global__ void apf_scale_wh_k(float* w, float* wh, const float* h,
                                const float* d_sum, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float inv = 1.0f / *d_sum;
    w[i] *= inv;
    wh[i] = w[i] * h[i];
}

__global__ void apf_compute_loglik_k(float* d_scalars, int n) {
    float ratio = fmaxf(d_scalars[1] / (float)n, 1e-30f);
    d_scalars[3] = d_scalars[0] + logf(ratio);
}

__global__ void apf_resample_k(float* h_out, const float* h_in,
                                const float* cdf, float u_base, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float target = u_base + (float)i / (float)n;
    if (target >= 1.0f) target -= 1.0f;
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (cdf[mid] < target) lo = mid + 1; else hi = mid;
    }
    h_out[i] = h_in[lo];
}

__global__ void apf_propagate_weight_k(
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
        if (cdf[mid] < target) lo = mid + 1; else hi = mid;
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
    float h_new = h[i], mp = mu_pred_resampled[i];
    float eta_new = y_t * __expf(-h_new * 0.5f);
    float eta_pred = y_t * __expf(-mp * 0.5f);
    float lp_new, lp_pred;
    if (nu_obs > 0.0f) {
        lp_new  = bpf_log_t_pdf(eta_new, nu_obs)  - h_new * 0.5f;
        lp_pred = bpf_log_t_pdf(eta_pred, nu_obs) - mp * 0.5f;
    } else {
        lp_new  = -0.9189385f - 0.5f * eta_new * eta_new  - h_new * 0.5f;
        lp_pred = -0.9189385f - 0.5f * eta_pred * eta_pred - mp * 0.5f;
    }
    log_w[i] = lp_new - lp_pred;
}

// =============================================================================
// APF Create / Destroy / Step
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

    apf_init_rng_k<<<s->grid, s->block, 0, s->stream>>>(
        s->d_rng, (unsigned long long)seed, n_particles);
    float std_stat = sqrtf((sigma_z * sigma_z) / fmaxf(1.0f - rho * rho, 1e-6f));
    apf_init_particles_k<<<s->grid, s->block, 0, s->stream>>>(
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
    float sc[4];
    cudaMemcpy(sc, d_scalars, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    BpfResult r; r.h_mean = sc[2]; r.log_lik = sc[3]; return r;
}

BpfResult gpu_apf_step(GpuApfState* s, float y_t) {
    int n = s->n_particles, g = s->grid, b = s->block;
    cudaStream_t st = s->stream;
    size_t smem = b * sizeof(float);

    if (s->timestep == 0) {
        apf_propagate_weight_k<<<g, b, 0, st>>>(
            s->d_h, s->d_log_w, s->d_rng,
            s->rho, s->sigma_z, s->mu, s->nu_state, s->nu_obs, y_t, n, 0);
        apf_set_scalar_k<<<1, 1, 0, st>>>(s->d_scalars + 0, -1e30f);
        apf_reduce_max_k<<<g, b, smem, st>>>(s->d_log_w, s->d_scalars + 0, n);
        apf_exp_sub_k<<<g, b, 0, st>>>(s->d_w, s->d_log_w, s->d_scalars + 0, n);
        apf_set_scalar_k<<<1, 1, 0, st>>>(s->d_scalars + 1, 0.0f);
        apf_reduce_sum_k<<<g, b, smem, st>>>(s->d_w, s->d_scalars + 1, n);
        apf_scale_wh_k<<<g, b, 0, st>>>(s->d_w, s->d_wh, s->d_h, s->d_scalars + 1, n);
        apf_set_scalar_k<<<1, 1, 0, st>>>(s->d_scalars + 2, 0.0f);
        apf_reduce_sum_k<<<g, b, smem, st>>>(s->d_wh, s->d_scalars + 2, n);
        apf_compute_loglik_k<<<1, 1, 0, st>>>(s->d_scalars, n);
        thrust::inclusive_scan(thrust::cuda::par.on(st),
            thrust::device_ptr<float>(s->d_w),
            thrust::device_ptr<float>(s->d_w + n),
            thrust::device_ptr<float>(s->d_cdf));
        float u = host_pcg32_float(&s->host_rng_state) / (float)n;
        apf_resample_k<<<g, b, 0, st>>>(s->d_h2, s->d_h, s->d_cdf, u, n);
        float* tmp = s->d_h; s->d_h = s->d_h2; s->d_h2 = tmp;
        cudaStreamSynchronize(st);
        s->timestep++;
        return apf_read_scalars(s->d_scalars);
    }

    apf_first_stage<<<g, b, 0, st>>>(
        s->d_h, s->d_mu_pred, s->d_log_v, s->rho, s->mu, s->nu_obs, y_t, n);
    apf_set_scalar_k<<<1, 1, 0, st>>>(s->d_scalars + 0, -1e30f);
    apf_reduce_max_k<<<g, b, smem, st>>>(s->d_log_v, s->d_scalars + 0, n);
    apf_exp_sub_k<<<g, b, 0, st>>>(s->d_v, s->d_log_v, s->d_scalars + 0, n);
    apf_set_scalar_k<<<1, 1, 0, st>>>(s->d_scalars + 1, 0.0f);
    apf_reduce_sum_k<<<g, b, smem, st>>>(s->d_v, s->d_scalars + 1, n);
    apf_scale_wh_k<<<g, b, 0, st>>>(s->d_v, s->d_wh, s->d_v, s->d_scalars + 1, n);
    thrust::inclusive_scan(thrust::cuda::par.on(st),
        thrust::device_ptr<float>(s->d_v),
        thrust::device_ptr<float>(s->d_v + n),
        thrust::device_ptr<float>(s->d_cdf));

    float u = host_pcg32_float(&s->host_rng_state) / (float)n;
    apf_resample_pair<<<g, b, 0, st>>>(
        s->d_h2, s->d_mu_pred2, s->d_h, s->d_mu_pred, s->d_cdf, u, n);
    float* tmp = s->d_h; s->d_h = s->d_h2; s->d_h2 = tmp;

    apf_propagate_correct<<<g, b, 0, st>>>(
        s->d_h, s->d_mu_pred2, s->d_log_w, s->d_rng,
        s->rho, s->sigma_z, s->mu, s->nu_state, s->nu_obs, y_t, n);

    apf_set_scalar_k<<<1, 1, 0, st>>>(s->d_scalars + 0, -1e30f);
    apf_reduce_max_k<<<g, b, smem, st>>>(s->d_log_w, s->d_scalars + 0, n);
    apf_exp_sub_k<<<g, b, 0, st>>>(s->d_w, s->d_log_w, s->d_scalars + 0, n);
    apf_set_scalar_k<<<1, 1, 0, st>>>(s->d_scalars + 1, 0.0f);
    apf_reduce_sum_k<<<g, b, smem, st>>>(s->d_w, s->d_scalars + 1, n);
    apf_scale_wh_k<<<g, b, 0, st>>>(s->d_w, s->d_wh, s->d_h, s->d_scalars + 1, n);
    apf_set_scalar_k<<<1, 1, 0, st>>>(s->d_scalars + 2, 0.0f);
    apf_reduce_sum_k<<<g, b, smem, st>>>(s->d_wh, s->d_scalars + 2, n);
    apf_compute_loglik_k<<<1, 1, 0, st>>>(s->d_scalars, n);

    thrust::inclusive_scan(thrust::cuda::par.on(st),
        thrust::device_ptr<float>(s->d_w),
        thrust::device_ptr<float>(s->d_w + n),
        thrust::device_ptr<float>(s->d_cdf));
    u = host_pcg32_float(&s->host_rng_state) / (float)n;
    apf_resample_k<<<g, b, 0, st>>>(s->d_h2, s->d_h, s->d_cdf, u, n);
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
    GpuApfState* st = gpu_apf_create(n_particles, rho, sigma_z, mu, nu_state, nu_obs, seed);
    int skip = 100; double sse = 0.0; int count = 0;
    for (int t = 0; t < n_ticks; t++) {
        BpfResult r = gpu_apf_step(st, (float)returns[t]);
        if (t >= skip) { double e = (double)r.h_mean - true_h[t]; sse += e*e; count++; }
    }
    gpu_apf_destroy(st);
    return sqrt(sse / (double)count);
}

// =============================================================================
// IMM (uses gpu_bpf_* which is now full PTX for BPF)
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
        double p_stay = 0.95, p_sw = (n_models > 1) ? (1.0 - p_stay) / (n_models - 1) : 0.0;
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
