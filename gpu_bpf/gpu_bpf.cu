/**
 * @file gpu_bpf.cu
 * @brief GPU Bootstrap PF + IMM with CUDA stream parallelism
 *
 * Custom reduction kernels replace thrust for max/sum to enable fully async
 * execution across K streams. Only thrust::inclusive_scan remains (device→device).
 */

#include "gpu_bpf.cuh"
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/system/cuda/execution_policy.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

// =============================================================================
// Device helpers
// =============================================================================

__device__ static float bpf_sample_t(curandState* s, float nu) {
    if (nu <= 0.0f || nu > 100.0f) return curand_normal(s);
    float z = curand_normal(s);
    float chi2 = 0.0f;
    for (int k = 0; k < (int)nu; k++) {
        float g = curand_normal(s);
        chi2 += g * g;
    }
    return z * rsqrtf(chi2 / nu);
}

__device__ static float bpf_log_t_pdf(float x, float nu) {
    return lgammaf((nu + 1.0f) / 2.0f) - lgammaf(nu / 2.0f)
         - 0.5f * logf(nu * 3.14159265f)
         - (nu + 1.0f) / 2.0f * logf(1.0f + x * x / nu);
}

// Atomic float max via CAS
__device__ static float atomicMaxf(float* addr, float val) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

// =============================================================================
// Kernels: BPF core
// =============================================================================

__global__ void bpf_init_rng(curandState* states, unsigned long long seed, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) curand_init(seed, i, 0, &states[i]);
}

__global__ void bpf_init_particles(float* h, curandState* states,
                                    float mu, float std_stat, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = mu + std_stat * curand_normal(&states[i]);
}

__global__ void bpf_propagate_weight(
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
// Kernels: Async reductions (no host sync)
// =============================================================================

// Initialize a device scalar
__global__ void bpf_set_scalar(float* scalar, float val) {
    *scalar = val;
}

// Block-reduce max → atomicMax to output scalar
__global__ void bpf_reduce_max(const float* in, float* out, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? in[i] : -1e30f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0) atomicMaxf(out, sdata[0]);
}

// Block-reduce sum → atomicAdd to output scalar
__global__ void bpf_reduce_sum(const float* in, float* out, int n) {
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

// exp(log_w - max_lw), reading max from device scalar
__global__ void bpf_exp_sub_dev(float* w, const float* log_w, const float* d_max, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) w[i] = __expf(log_w[i] - *d_max);
}

// Normalize weights and compute w*h, reading sum from device scalar
__global__ void bpf_scale_wh_dev(float* w, float* wh, const float* h,
                                  const float* d_sum, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float inv_sum = 1.0f / *d_sum;
        w[i] *= inv_sum;
        wh[i] = w[i] * h[i];
    }
}

// Compute log-likelihood from max_lw and sum_w: log(sum_w/N) + max_lw
// d_scalars[0]=max_lw, d_scalars[1]=sum_w → d_scalars[3]=log_lik
__global__ void bpf_compute_loglik(float* d_scalars, int n) {
    d_scalars[3] = d_scalars[0] + logf(fmaxf(d_scalars[1] / (float)n, 1e-30f));
}

// Systematic resampling
__global__ void bpf_resample(float* h_out, const float* h_in,
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
// Host-side PCG32
// =============================================================================

static inline unsigned int bpf_pcg32(unsigned long long* state) {
    unsigned long long old = *state;
    *state = old * 6364136223846793005ULL + 1442695040888963407ULL;
    unsigned int xor_shifted = (unsigned int)(((old >> 18u) ^ old) >> 27u);
    unsigned int rot = (unsigned int)(old >> 59u);
    return (xor_shifted >> rot) | (xor_shifted << ((-rot) & 31));
}

static inline float bpf_pcg32_float(unsigned long long* state) {
    return (float)(bpf_pcg32(state) >> 9) * (1.0f / 8388608.0f);
}

// =============================================================================
// BPF Create / Destroy
// =============================================================================

GpuBpfState* gpu_bpf_create(int n_particles, float rho, float sigma_z, float mu,
                              float nu_state, float nu_obs, int seed) {
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

    cudaStreamCreate(&s->stream);

    cudaMalloc(&s->d_h,       n_particles * sizeof(float));
    cudaMalloc(&s->d_h2,      n_particles * sizeof(float));
    cudaMalloc(&s->d_log_w,   n_particles * sizeof(float));
    cudaMalloc(&s->d_w,       n_particles * sizeof(float));
    cudaMalloc(&s->d_cdf,     n_particles * sizeof(float));
    cudaMalloc(&s->d_wh,      n_particles * sizeof(float));
    cudaMalloc(&s->d_rng,     n_particles * sizeof(curandState));
    cudaMalloc(&s->d_scalars, 4 * sizeof(float));  // max_lw, sum_w, h_est, log_lik

    bpf_init_rng<<<s->grid, s->block, 0, s->stream>>>(
        s->d_rng, (unsigned long long)seed, n_particles);

    float std_stat = sqrtf((sigma_z * sigma_z) / fmaxf(1.0f - rho * rho, 1e-6f));
    bpf_init_particles<<<s->grid, s->block, 0, s->stream>>>(
        s->d_h, s->d_rng, mu, std_stat, n_particles);
    cudaStreamSynchronize(s->stream);

    return s;
}

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
    free(s);
}

// =============================================================================
// BPF Async Step — all work on stream, no host sync
// =============================================================================

void gpu_bpf_step_async(GpuBpfState* s, float y_t) {
    int n = s->n_particles;
    int g = s->grid;
    int b = s->block;
    cudaStream_t st = s->stream;
    size_t smem = b * sizeof(float);

    // 1. Propagate + weight
    bpf_propagate_weight<<<g, b, 0, st>>>(
        s->d_h, s->d_log_w, s->d_rng,
        s->rho, s->sigma_z, s->mu, s->nu_state, s->nu_obs, y_t,
        n, (s->timestep > 0) ? 1 : 0);

    // 2. Max of log_w → d_scalars[0]
    bpf_set_scalar<<<1, 1, 0, st>>>(s->d_scalars + 0, -1e30f);
    bpf_reduce_max<<<g, b, smem, st>>>(s->d_log_w, s->d_scalars + 0, n);

    // 3. w = exp(log_w - max_lw)
    bpf_exp_sub_dev<<<g, b, 0, st>>>(s->d_w, s->d_log_w, s->d_scalars + 0, n);

    // 4. sum_w → d_scalars[1]
    bpf_set_scalar<<<1, 1, 0, st>>>(s->d_scalars + 1, 0.0f);
    bpf_reduce_sum<<<g, b, smem, st>>>(s->d_w, s->d_scalars + 1, n);

    // 5. Normalize + w*h
    bpf_scale_wh_dev<<<g, b, 0, st>>>(s->d_w, s->d_wh, s->d_h, s->d_scalars + 1, n);

    // 6. h_est = sum(w*h) → d_scalars[2]
    bpf_set_scalar<<<1, 1, 0, st>>>(s->d_scalars + 2, 0.0f);
    bpf_reduce_sum<<<g, b, smem, st>>>(s->d_wh, s->d_scalars + 2, n);

    // 7. log_lik → d_scalars[3]
    bpf_compute_loglik<<<1, 1, 0, st>>>(s->d_scalars, n);

    // 8. Inclusive scan for CDF (thrust on stream — device→device, no host sync)
    thrust::inclusive_scan(
        thrust::cuda::par.on(st),
        thrust::device_ptr<float>(s->d_w),
        thrust::device_ptr<float>(s->d_w + n),
        thrust::device_ptr<float>(s->d_cdf));

    // 9. Resample
    float u = bpf_pcg32_float(&s->host_rng_state) / (float)n;
    bpf_resample<<<g, b, 0, st>>>(s->d_h2, s->d_h, s->d_cdf, u, n);

    // Swap particle buffers
    float* tmp = s->d_h; s->d_h = s->d_h2; s->d_h2 = tmp;
    s->timestep++;
}

// Read result after cudaDeviceSynchronize() or cudaStreamSynchronize()
BpfResult gpu_bpf_get_result(GpuBpfState* s) {
    float scalars[4];
    cudaMemcpy(scalars, s->d_scalars, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    BpfResult r;
    r.h_mean  = scalars[2];
    r.log_lik = scalars[3];
    return r;
}

// =============================================================================
// BPF Synchronous Step (convenience wrapper)
// =============================================================================

BpfResult gpu_bpf_step(GpuBpfState* s, float y_t) {
    gpu_bpf_step_async(s, y_t);
    cudaStreamSynchronize(s->stream);
    return gpu_bpf_get_result(s);
}

// =============================================================================
// BPF Batch RMSE
// =============================================================================

double gpu_bpf_run_rmse(
    const double* returns, const double* true_h, int n_ticks,
    int n_particles,
    float rho, float sigma_z, float mu,
    float nu_state, float nu_obs, int seed
) {
    GpuBpfState* state = gpu_bpf_create(n_particles, rho, sigma_z, mu,
                                         nu_state, nu_obs, seed);
    int skip = 100;
    double sum_sq = 0.0;
    int count = 0;

    for (int t = 0; t < n_ticks; t++) {
        BpfResult r = gpu_bpf_step(state, (float)returns[t]);
        if (t >= skip) {
            double err = (double)r.h_mean - true_h[t];
            sum_sq += err * err;
            count++;
        }
    }

    gpu_bpf_destroy(state);
    return sqrt(sum_sq / count);
}

// =============================================================================
// APF Kernels
// =============================================================================

// Compute predictive mean and first-stage log weight
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

// Resample both h and mu_pred together
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

// Propagate resampled particles + compute second-stage correction weight
__global__ void apf_propagate_correct(
    float* h, const float* mu_pred_resampled, float* log_w,
    curandState* states,
    float rho, float sigma_z, float mu,
    float nu_state, float nu_obs, float y_t, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    // Propagate: h_new = mu + rho*(h_resampled - mu) + sigma_z * eps
    float eps = bpf_sample_t(&states[i], nu_state);
    h[i] = mu + rho * (h[i] - mu) + sigma_z * eps;
    
    float h_new = h[i];
    float mp = mu_pred_resampled[i];
    
    // Second-stage weight: log p(y|h_new) - log p(y|mu_pred)
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
    s->rho = rho;
    s->sigma_z = sigma_z;
    s->mu = mu;
    s->nu_state = nu_state;
    s->nu_obs = nu_obs;
    s->block = 256;
    s->grid = (n_particles + s->block - 1) / s->block;
    s->host_rng_state = (unsigned long long)seed * 67890ULL + 12345ULL;
    s->timestep = 0;

    cudaStreamCreate(&s->stream);

    cudaMalloc(&s->d_h,         n_particles * sizeof(float));
    cudaMalloc(&s->d_h2,        n_particles * sizeof(float));
    cudaMalloc(&s->d_mu_pred,   n_particles * sizeof(float));
    cudaMalloc(&s->d_mu_pred2,  n_particles * sizeof(float));
    cudaMalloc(&s->d_log_v,     n_particles * sizeof(float));
    cudaMalloc(&s->d_v,         n_particles * sizeof(float));
    cudaMalloc(&s->d_log_w,     n_particles * sizeof(float));
    cudaMalloc(&s->d_w,         n_particles * sizeof(float));
    cudaMalloc(&s->d_cdf,       n_particles * sizeof(float));
    cudaMalloc(&s->d_wh,        n_particles * sizeof(float));
    cudaMalloc(&s->d_rng,       n_particles * sizeof(curandState));
    cudaMalloc(&s->d_scalars,   4 * sizeof(float));

    bpf_init_rng<<<s->grid, s->block, 0, s->stream>>>(
        s->d_rng, (unsigned long long)seed, n_particles);
    float std_stat = sqrtf((sigma_z * sigma_z) / fmaxf(1.0f - rho * rho, 1e-6f));
    bpf_init_particles<<<s->grid, s->block, 0, s->stream>>>(
        s->d_h, s->d_rng, mu, std_stat, n_particles);
    cudaStreamSynchronize(s->stream);

    return s;
}

void gpu_apf_destroy(GpuApfState* s) {
    if (!s) return;
    cudaStreamDestroy(s->stream);
    cudaFree(s->d_h);
    cudaFree(s->d_h2);
    cudaFree(s->d_mu_pred);
    cudaFree(s->d_mu_pred2);
    cudaFree(s->d_log_v);
    cudaFree(s->d_v);
    cudaFree(s->d_log_w);
    cudaFree(s->d_w);
    cudaFree(s->d_cdf);
    cudaFree(s->d_wh);
    cudaFree(s->d_rng);
    cudaFree(s->d_scalars);
    free(s);
}

// Helper: read BpfResult from any d_scalars pointer
static BpfResult apf_read_scalars(float* d_scalars) {
    float scalars[4];
    cudaMemcpy(scalars, d_scalars, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    BpfResult r;
    r.h_mean = scalars[2];
    r.log_lik = scalars[3];
    return r;
}

// =============================================================================
// APF Step
//   1. First-stage weights from predictive means
//   2. Resample by first-stage
//   3. Propagate + second-stage correction
//   4. Weighted mean with second-stage weights
// =============================================================================

BpfResult gpu_apf_step(GpuApfState* s, float y_t) {
    int n = s->n_particles;
    int g = s->grid;
    int b = s->block;
    cudaStream_t st = s->stream;
    size_t smem = b * sizeof(float);

    if (s->timestep == 0) {
        // First tick: no propagation yet, just weight like BPF
        bpf_propagate_weight<<<g, b, 0, st>>>(
            s->d_h, s->d_log_w, s->d_rng,
            s->rho, s->sigma_z, s->mu, s->nu_state, s->nu_obs, y_t,
            n, 0);

        // Max → exp → sum → normalize → weighted mean
        bpf_set_scalar<<<1, 1, 0, st>>>(s->d_scalars + 0, -1e30f);
        bpf_reduce_max<<<g, b, smem, st>>>(s->d_log_w, s->d_scalars + 0, n);
        bpf_exp_sub_dev<<<g, b, 0, st>>>(s->d_w, s->d_log_w, s->d_scalars + 0, n);
        bpf_set_scalar<<<1, 1, 0, st>>>(s->d_scalars + 1, 0.0f);
        bpf_reduce_sum<<<g, b, smem, st>>>(s->d_w, s->d_scalars + 1, n);
        bpf_scale_wh_dev<<<g, b, 0, st>>>(s->d_w, s->d_wh, s->d_h, s->d_scalars + 1, n);
        bpf_set_scalar<<<1, 1, 0, st>>>(s->d_scalars + 2, 0.0f);
        bpf_reduce_sum<<<g, b, smem, st>>>(s->d_wh, s->d_scalars + 2, n);
        bpf_compute_loglik<<<1, 1, 0, st>>>(s->d_scalars, n);

        // Resample
        thrust::inclusive_scan(
            thrust::cuda::par.on(st),
            thrust::device_ptr<float>(s->d_w),
            thrust::device_ptr<float>(s->d_w + n),
            thrust::device_ptr<float>(s->d_cdf));
        float u = bpf_pcg32_float(&s->host_rng_state) / (float)n;
        bpf_resample<<<g, b, 0, st>>>(s->d_h2, s->d_h, s->d_cdf, u, n);
        float* tmp = s->d_h; s->d_h = s->d_h2; s->d_h2 = tmp;

        cudaStreamSynchronize(st);
        s->timestep++;
        return apf_read_scalars(s->d_scalars);
    }

    // ─── 1. First-stage: weight by predictive mean likelihood ───
    apf_first_stage<<<g, b, 0, st>>>(
        s->d_h, s->d_mu_pred, s->d_log_v,
        s->rho, s->mu, s->nu_obs, y_t, n);

    // Normalize first-stage weights for resampling
    bpf_set_scalar<<<1, 1, 0, st>>>(s->d_scalars + 0, -1e30f);
    bpf_reduce_max<<<g, b, smem, st>>>(s->d_log_v, s->d_scalars + 0, n);
    bpf_exp_sub_dev<<<g, b, 0, st>>>(s->d_v, s->d_log_v, s->d_scalars + 0, n);
    bpf_set_scalar<<<1, 1, 0, st>>>(s->d_scalars + 1, 0.0f);
    bpf_reduce_sum<<<g, b, smem, st>>>(s->d_v, s->d_scalars + 1, n);
    bpf_scale_wh_dev<<<g, b, 0, st>>>(
        s->d_v, s->d_wh, s->d_v,  // wh unused, just normalizing v
        s->d_scalars + 1, n);

    // CDF for first-stage resampling
    thrust::inclusive_scan(
        thrust::cuda::par.on(st),
        thrust::device_ptr<float>(s->d_v),
        thrust::device_ptr<float>(s->d_v + n),
        thrust::device_ptr<float>(s->d_cdf));

    // ─── 2. Resample h and mu_pred together ───
    float u = bpf_pcg32_float(&s->host_rng_state) / (float)n;
    apf_resample_pair<<<g, b, 0, st>>>(
        s->d_h2, s->d_mu_pred2,
        s->d_h, s->d_mu_pred,
        s->d_cdf, u, n);

    // Swap: h2 → h (resampled particles)
    float* tmp = s->d_h; s->d_h = s->d_h2; s->d_h2 = tmp;

    // ─── 3. Propagate + second-stage correction weight ───
    apf_propagate_correct<<<g, b, 0, st>>>(
        s->d_h, s->d_mu_pred2, s->d_log_w, s->d_rng,
        s->rho, s->sigma_z, s->mu, s->nu_state, s->nu_obs, y_t, n);

    // ─── 4. Normalize second-stage weights → weighted mean ───
    bpf_set_scalar<<<1, 1, 0, st>>>(s->d_scalars + 0, -1e30f);
    bpf_reduce_max<<<g, b, smem, st>>>(s->d_log_w, s->d_scalars + 0, n);
    bpf_exp_sub_dev<<<g, b, 0, st>>>(s->d_w, s->d_log_w, s->d_scalars + 0, n);
    bpf_set_scalar<<<1, 1, 0, st>>>(s->d_scalars + 1, 0.0f);
    bpf_reduce_sum<<<g, b, smem, st>>>(s->d_w, s->d_scalars + 1, n);
    bpf_scale_wh_dev<<<g, b, 0, st>>>(s->d_w, s->d_wh, s->d_h, s->d_scalars + 1, n);
    bpf_set_scalar<<<1, 1, 0, st>>>(s->d_scalars + 2, 0.0f);
    bpf_reduce_sum<<<g, b, smem, st>>>(s->d_wh, s->d_scalars + 2, n);
    bpf_compute_loglik<<<1, 1, 0, st>>>(s->d_scalars, n);

    // ─── 5. Resample for next step (by second-stage weights) ───
    thrust::inclusive_scan(
        thrust::cuda::par.on(st),
        thrust::device_ptr<float>(s->d_w),
        thrust::device_ptr<float>(s->d_w + n),
        thrust::device_ptr<float>(s->d_cdf));
    u = bpf_pcg32_float(&s->host_rng_state) / (float)n;
    bpf_resample<<<g, b, 0, st>>>(s->d_h2, s->d_h, s->d_cdf, u, n);
    tmp = s->d_h; s->d_h = s->d_h2; s->d_h2 = tmp;

    cudaStreamSynchronize(st);
    s->timestep++;

    float scalars[4];
    cudaMemcpy(scalars, s->d_scalars, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    BpfResult r;
    r.h_mean = scalars[2];
    r.log_lik = scalars[3];
    return r;
}

// =============================================================================
// APF Batch RMSE
// =============================================================================

double gpu_apf_run_rmse(
    const double* returns, const double* true_h, int n_ticks,
    int n_particles, float rho, float sigma_z, float mu,
    float nu_state, float nu_obs, int seed
) {
    GpuApfState* state = gpu_apf_create(n_particles, rho, sigma_z, mu,
                                         nu_state, nu_obs, seed);
    int skip = 100;
    double sum_sq = 0.0;
    int count = 0;

    for (int t = 0; t < n_ticks; t++) {
        BpfResult r = gpu_apf_step(state, (float)returns[t]);
        if (t >= skip) {
            double err = (double)r.h_mean - true_h[t];
            sum_sq += err * err;
            count++;
        }
    }

    gpu_apf_destroy(state);
    return sqrt(sum_sq / count);
}

// =============================================================================
// IMM utilities
// =============================================================================

static double log_sum_exp(const double* x, int n) {
    double mx = -1e30;
    for (int i = 0; i < n; i++)
        if (x[i] > mx) mx = x[i];
    double s = 0.0;
    for (int i = 0; i < n; i++)
        s += exp(x[i] - mx);
    return mx + log(s);
}

// =============================================================================
// IMM Create / Destroy
// =============================================================================

GpuImmState* gpu_imm_create(
    const ImmModelParams* models, int n_models,
    int n_particles_per_model,
    const float* transition_matrix,
    int seed
) {
    if (n_models > IMM_MAX_MODELS) {
        fprintf(stderr, "IMM: n_models=%d exceeds max=%d\n", n_models, IMM_MAX_MODELS);
        return NULL;
    }

    GpuImmState* s = (GpuImmState*)calloc(1, sizeof(GpuImmState));
    s->n_models = n_models;
    s->n_particles_per_model = n_particles_per_model;
    s->timestep = 0;

    s->filters = (GpuBpfState**)malloc(n_models * sizeof(GpuBpfState*));
    for (int k = 0; k < n_models; k++) {
        s->filters[k] = gpu_bpf_create(
            n_particles_per_model,
            models[k].rho, models[k].sigma_z, models[k].mu,
            models[k].nu_state, models[k].nu_obs,
            seed + k * 7919);
    }

    s->log_pi = (double*)malloc(n_models * sizeof(double));
    s->log_pi_pred = (double*)malloc(n_models * sizeof(double));
    double log_uniform = -log((double)n_models);
    for (int k = 0; k < n_models; k++)
        s->log_pi[k] = log_uniform;

    s->log_T = (double*)malloc(n_models * n_models * sizeof(double));
    if (transition_matrix) {
        for (int i = 0; i < n_models * n_models; i++)
            s->log_T[i] = log(fmax((double)transition_matrix[i], 1e-30));
    } else {
        double p_stay = 0.95;
        double p_switch = (n_models > 1) ? (1.0 - p_stay) / (n_models - 1) : 0.0;
        for (int i = 0; i < n_models; i++)
            for (int j = 0; j < n_models; j++)
                s->log_T[i * n_models + j] = log(fmax((i == j) ? p_stay : p_switch, 1e-30));
    }

    return s;
}

void gpu_imm_destroy(GpuImmState* s) {
    if (!s) return;
    for (int k = 0; k < s->n_models; k++)
        gpu_bpf_destroy(s->filters[k]);
    free(s->filters);
    free(s->log_pi);
    free(s->log_pi_pred);
    free(s->log_T);
    free(s);
}

// =============================================================================
// IMM Step — all K BPFs launch async, one sync, then mix
// =============================================================================

ImmResult gpu_imm_step(GpuImmState* s, float y_t) {
    int K = s->n_models;

    // ─── 1. Interaction: π_k^- = Σ_j T(j→k) * π_j ───
    for (int k = 0; k < K; k++) {
        double terms[IMM_MAX_MODELS];
        for (int j = 0; j < K; j++)
            terms[j] = s->log_T[j * K + k] + s->log_pi[j];
        s->log_pi_pred[k] = log_sum_exp(terms, K);
    }

    // ─── 2. Launch ALL BPF steps async on their streams ───
    for (int k = 0; k < K; k++)
        gpu_bpf_step_async(s->filters[k], y_t);

    // ─── 3. Single sync — wait for all streams ───
    cudaDeviceSynchronize();

    // ─── 4. Collect results ───
    double log_liks[IMM_MAX_MODELS];
    float h_ests[IMM_MAX_MODELS];
    for (int k = 0; k < K; k++) {
        BpfResult r = gpu_bpf_get_result(s->filters[k]);
        h_ests[k] = r.h_mean;
        log_liks[k] = (double)r.log_lik;
    }

    // ─── 5. Update: log π_k = log π_k^- + log p(y|model_k) - log Z ───
    double log_joint[IMM_MAX_MODELS];
    for (int k = 0; k < K; k++)
        log_joint[k] = s->log_pi_pred[k] + log_liks[k];

    double log_Z = log_sum_exp(log_joint, K);
    for (int k = 0; k < K; k++)
        s->log_pi[k] = log_joint[k] - log_Z;

    // ─── 6. Mixed output ───
    double h_mixed = 0.0;
    int best_k = 0;
    double best_log_pi = -1e30;
    for (int k = 0; k < K; k++) {
        double pi_k = exp(s->log_pi[k]);
        h_mixed += pi_k * (double)h_ests[k];
        if (s->log_pi[k] > best_log_pi) {
            best_log_pi = s->log_pi[k];
            best_k = k;
        }
    }

    s->timestep++;

    ImmResult r;
    r.h_mean = (float)h_mixed;
    r.vol = expf((float)h_mixed * 0.5f);
    r.log_lik = (float)log_Z;
    r.best_model = best_k;
    r.best_prob = (float)exp(best_log_pi);
    return r;
}

// =============================================================================
// IMM Accessors
// =============================================================================

float gpu_imm_get_prob(const GpuImmState* state, int k) {
    if (k < 0 || k >= state->n_models) return 0.0f;
    return (float)exp(state->log_pi[k]);
}

void gpu_imm_get_probs(const GpuImmState* state, float* probs_out) {
    for (int k = 0; k < state->n_models; k++)
        probs_out[k] = (float)exp(state->log_pi[k]);
}

// =============================================================================
// Grid Builder
// =============================================================================

ImmModelParams* gpu_imm_build_grid(
    const float* rhos, int n_rho,
    const float* sigma_zs, int n_sigma,
    const float* mus, int n_mu,
    float nu_state, float nu_obs,
    int* out_n_models
) {
    int total = n_rho * n_sigma * n_mu;
    ImmModelParams* grid = (ImmModelParams*)malloc(total * sizeof(ImmModelParams));
    int idx = 0;
    for (int r = 0; r < n_rho; r++)
        for (int s = 0; s < n_sigma; s++)
            for (int m = 0; m < n_mu; m++) {
                grid[idx].rho = rhos[r];
                grid[idx].sigma_z = sigma_zs[s];
                grid[idx].mu = mus[m];
                grid[idx].nu_state = nu_state;
                grid[idx].nu_obs = nu_obs;
                idx++;
            }
    *out_n_models = total;
    return grid;
}
