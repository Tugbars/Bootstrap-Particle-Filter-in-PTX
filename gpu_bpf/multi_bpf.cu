/**
 * @file multi_bpf.cu
 * @brief K-Instance Bootstrap PF for volatility tracking
 *
 * Pure BPF × K instances. No bands, no Silverman, no ESS gating.
 * Each instance on its own CUDA stream → fully async.
 * One cudaDeviceSynchronize() per tick, then host-side Bayesian mixing.
 */

#include "multi_bpf.cuh"
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

__device__ static float mbpf_sample_t(curandState* s, float nu) {
    if (nu <= 0.0f || nu > 100.0f) return curand_normal(s);
    float z = curand_normal(s);
    float chi2 = 0.0f;
    for (int k = 0; k < (int)nu; k++) {
        float g = curand_normal(s);
        chi2 += g * g;
    }
    return z * rsqrtf(chi2 / nu);
}

__device__ static float mbpf_log_t_pdf(float x, float nu) {
    return lgammaf((nu + 1.0f) / 2.0f) - lgammaf(nu / 2.0f)
         - 0.5f * logf(nu * 3.14159265f)
         - (nu + 1.0f) / 2.0f * logf(1.0f + x * x / nu);
}

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
// BPF Kernels (minimal, no bands/Silverman/ESS)
// =============================================================================

__global__ void mbpf_init_rng(curandState* states, unsigned long long seed, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) curand_init(seed, i, 0, &states[i]);
}

__global__ void mbpf_init_particles(float* h, curandState* states,
                                     float mu, float std_stat, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = mu + std_stat * curand_normal(&states[i]);
}

__global__ void mbpf_propagate_weight(
    float* h, float* log_w, curandState* states,
    float rho, float sigma_z, float mu,
    float nu_state, float nu_obs, float y_t,
    int n, int do_propagate
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (do_propagate) {
        float eps = mbpf_sample_t(&states[i], nu_state);
        h[i] = mu + rho * (h[i] - mu) + sigma_z * eps;
    }

    float h_i = h[i];
    float eta = y_t * __expf(-h_i * 0.5f);
    log_w[i] = (nu_obs > 0.0f)
        ? mbpf_log_t_pdf(eta, nu_obs) - h_i * 0.5f
        : -0.9189385f - 0.5f * eta * eta - h_i * 0.5f;
}

__global__ void mbpf_set_scalar(float* scalar, float val) {
    *scalar = val;
}

__global__ void mbpf_reduce_max(const float* in, float* out, int n) {
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

__global__ void mbpf_reduce_sum(const float* in, float* out, int n) {
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

__global__ void mbpf_exp_sub(float* w, const float* log_w, const float* d_max, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) w[i] = __expf(log_w[i] - *d_max);
}

__global__ void mbpf_normalize_wh(float* w, float* wh, const float* h,
                                   const float* d_sum, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float inv_sum = 1.0f / *d_sum;
        w[i] *= inv_sum;
        wh[i] = w[i] * h[i];
    }
}

__global__ void mbpf_loglik(float* d_scalars, int n) {
    d_scalars[3] = d_scalars[0] + logf(fmaxf(d_scalars[1] / (float)n, 1e-30f));
}

__global__ void mbpf_resample(float* h_out, const float* h_in,
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
// Host PCG32
// =============================================================================

static inline unsigned int mbpf_pcg32(unsigned long long* state) {
    unsigned long long old = *state;
    *state = old * 6364136223846793005ULL + 1442695040888963407ULL;
    unsigned int xor_shifted = (unsigned int)(((old >> 18u) ^ old) >> 27u);
    unsigned int rot = (unsigned int)(old >> 59u);
    return (xor_shifted >> rot) | (xor_shifted << ((-rot) & 31));
}

static inline float mbpf_pcg32_float(unsigned long long* state) {
    return (float)(mbpf_pcg32(state) >> 9) * (1.0f / 8388608.0f);
}

// =============================================================================
// Host: log-sum-exp
// =============================================================================

static double lse(const double* x, int n) {
    double mx = -1e30;
    for (int i = 0; i < n; i++)
        if (x[i] > mx) mx = x[i];
    double s = 0.0;
    for (int i = 0; i < n; i++)
        s += exp(x[i] - mx);
    return mx + log(s);
}

// =============================================================================
// Create / Destroy
// =============================================================================

MbpfState* mbpf_create(
    int K, int n_particles,
    const MbpfParams* params,
    float p_stay, int seed
) {
    if (K > MBPF_MAX_K) {
        fprintf(stderr, "mbpf: K=%d exceeds max=%d\n", K, MBPF_MAX_K);
        return NULL;
    }

    MbpfState* s = (MbpfState*)calloc(1, sizeof(MbpfState));
    s->K = K;
    s->n_particles = n_particles;
    s->block = 256;
    s->grid = (n_particles + s->block - 1) / s->block;
    s->host_rng = (unsigned long long)seed * 67890ULL + 12345ULL;

    // Uniform prior
    double log_uniform = -log((double)K);
    for (int k = 0; k < K; k++)
        s->log_pi[k] = log_uniform;

    // Transition matrix
    float p_switch = (K > 1) ? (1.0f - p_stay) / (K - 1) : 0.0f;
    for (int i = 0; i < K; i++)
        for (int j = 0; j < K; j++)
            s->log_T[i * K + j] = log(fmax((double)((i == j) ? p_stay : p_switch), 1e-30));

    // Create K instances
    for (int k = 0; k < K; k++) {
        MbpfInstance* inst = &s->inst[k];
        inst->params = params[k];
        inst->timestep = 0;

        cudaStreamCreate(&inst->stream);
        cudaMalloc(&inst->d_h,     n_particles * sizeof(float));
        cudaMalloc(&inst->d_h2,    n_particles * sizeof(float));
        cudaMalloc(&inst->d_log_w, n_particles * sizeof(float));
        cudaMalloc(&inst->d_w,     n_particles * sizeof(float));
        cudaMalloc(&inst->d_cdf,   n_particles * sizeof(float));
        cudaMalloc(&inst->d_wh,    n_particles * sizeof(float));
        cudaMalloc(&inst->d_rng,   n_particles * sizeof(curandState));
        cudaMalloc(&inst->d_scalars, 4 * sizeof(float));

        // Init RNG + particles on instance stream
        mbpf_init_rng<<<s->grid, s->block, 0, inst->stream>>>(
            inst->d_rng, (unsigned long long)(seed + k * 7919), n_particles);

        float sigma_z = params[k].sigma_z;
        float rho = params[k].rho;
        float mu = params[k].mu;
        float std_stat = sqrtf((sigma_z * sigma_z) / fmaxf(1.0f - rho * rho, 1e-6f));

        mbpf_init_particles<<<s->grid, s->block, 0, inst->stream>>>(
            inst->d_h, inst->d_rng, mu, std_stat, n_particles);
    }

    cudaDeviceSynchronize();
    return s;
}

void mbpf_destroy(MbpfState* s) {
    if (!s) return;
    for (int k = 0; k < s->K; k++) {
        MbpfInstance* inst = &s->inst[k];
        cudaStreamDestroy(inst->stream);
        cudaFree(inst->d_h);
        cudaFree(inst->d_h2);
        cudaFree(inst->d_log_w);
        cudaFree(inst->d_w);
        cudaFree(inst->d_cdf);
        cudaFree(inst->d_wh);
        cudaFree(inst->d_rng);
        cudaFree(inst->d_scalars);
    }
    free(s);
}

// =============================================================================
// Step: launch all K async, sync once, mix on host
// =============================================================================

// Launch one BPF instance (async on its stream)
static void mbpf_step_instance_async(MbpfState* s, int k, float y_t) {
    MbpfInstance* inst = &s->inst[k];
    MbpfParams* p = &inst->params;
    int n = s->n_particles;
    int g = s->grid;
    int b = s->block;
    cudaStream_t st = inst->stream;
    size_t smem = b * sizeof(float);

    // 1. Propagate + weight
    mbpf_propagate_weight<<<g, b, 0, st>>>(
        inst->d_h, inst->d_log_w, inst->d_rng,
        p->rho, p->sigma_z, p->mu, p->nu_state, p->nu_obs, y_t,
        n, (inst->timestep > 0) ? 1 : 0);

    // 2. max(log_w)
    mbpf_set_scalar<<<1, 1, 0, st>>>(inst->d_scalars + 0, -1e30f);
    mbpf_reduce_max<<<g, b, smem, st>>>(inst->d_log_w, inst->d_scalars + 0, n);

    // 3. w = exp(log_w - max)
    mbpf_exp_sub<<<g, b, 0, st>>>(inst->d_w, inst->d_log_w, inst->d_scalars + 0, n);

    // 4. sum(w)
    mbpf_set_scalar<<<1, 1, 0, st>>>(inst->d_scalars + 1, 0.0f);
    mbpf_reduce_sum<<<g, b, smem, st>>>(inst->d_w, inst->d_scalars + 1, n);

    // 5. normalize + w*h
    mbpf_normalize_wh<<<g, b, 0, st>>>(inst->d_w, inst->d_wh, inst->d_h,
                                        inst->d_scalars + 1, n);

    // 6. h_est = sum(w*h)
    mbpf_set_scalar<<<1, 1, 0, st>>>(inst->d_scalars + 2, 0.0f);
    mbpf_reduce_sum<<<g, b, smem, st>>>(inst->d_wh, inst->d_scalars + 2, n);

    // 7. log_lik
    mbpf_loglik<<<1, 1, 0, st>>>(inst->d_scalars, n);

    // 8. CDF for resampling
    thrust::inclusive_scan(
        thrust::cuda::par.on(st),
        thrust::device_ptr<float>(inst->d_w),
        thrust::device_ptr<float>(inst->d_w + n),
        thrust::device_ptr<float>(inst->d_cdf));

    // 9. Resample
    float u = mbpf_pcg32_float(&s->host_rng) / (float)n;
    mbpf_resample<<<g, b, 0, st>>>(inst->d_h2, inst->d_h, inst->d_cdf, u, n);

    // Swap
    float* tmp = inst->d_h;
    inst->d_h = inst->d_h2;
    inst->d_h2 = tmp;

    inst->timestep++;
}

// Read instance result (call after sync)
static MbpfInstanceResult mbpf_read_instance(MbpfInstance* inst) {
    float scalars[4];
    cudaMemcpy(scalars, inst->d_scalars, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    MbpfInstanceResult r;
    r.h_mean = scalars[2];
    r.log_lik = scalars[3];
    return r;
}

MbpfResult mbpf_step(MbpfState* s, float y_t) {
    int K = s->K;

    // 1. Launch all K async
    for (int k = 0; k < K; k++)
        mbpf_step_instance_async(s, k, y_t);

    // 2. Single sync
    cudaDeviceSynchronize();

    // 3. Collect per-instance results
    double log_liks[MBPF_MAX_K];
    float h_ests[MBPF_MAX_K];
    for (int k = 0; k < K; k++) {
        MbpfInstanceResult r = mbpf_read_instance(&s->inst[k]);
        h_ests[k] = r.h_mean;
        log_liks[k] = (double)r.log_lik;
    }

    // 4. Bayesian model averaging
    //    Predict: log π_k^- = logsumexp_j( log T(j→k) + log π_j )
    double log_pi_pred[MBPF_MAX_K];
    for (int k = 0; k < K; k++) {
        double terms[MBPF_MAX_K];
        for (int j = 0; j < K; j++)
            terms[j] = s->log_T[j * K + k] + s->log_pi[j];
        log_pi_pred[k] = lse(terms, K);
    }

    //    Update: log π_k = log π_k^- + log p_k(y) - log Z
    double log_joint[MBPF_MAX_K];
    for (int k = 0; k < K; k++)
        log_joint[k] = log_pi_pred[k] + log_liks[k];

    double log_Z = lse(log_joint, K);
    for (int k = 0; k < K; k++)
        s->log_pi[k] = log_joint[k] - log_Z;

    // 5. Mixed output
    double h_mixed = 0.0;
    int best_k = 0;
    double best_lp = -1e30;
    for (int k = 0; k < K; k++) {
        double pi_k = exp(s->log_pi[k]);
        h_mixed += pi_k * (double)h_ests[k];
        if (s->log_pi[k] > best_lp) {
            best_lp = s->log_pi[k];
            best_k = k;
        }
    }

    MbpfResult r;
    r.h_mean = (float)h_mixed;
    r.vol = expf((float)h_mixed * 0.5f);
    r.log_lik = (float)log_Z;
    r.best_k = best_k;
    r.best_prob = (float)exp(best_lp);
    return r;
}

// =============================================================================
// Parameter update (from external learner)
// =============================================================================

void mbpf_set_params(MbpfState* s, int k, const MbpfParams* p) {
    if (k < 0 || k >= s->K) return;
    s->inst[k].params = *p;
}

void mbpf_get_probs(const MbpfState* s, float* probs_out) {
    for (int k = 0; k < s->K; k++)
        probs_out[k] = (float)exp(s->log_pi[k]);
}

// =============================================================================
// Batch RMSE
// =============================================================================

double mbpf_run_rmse(
    const double* returns, const double* true_h, int n_ticks,
    int K, int n_particles,
    const MbpfParams* params, float p_stay, int seed
) {
    MbpfState* s = mbpf_create(K, n_particles, params, p_stay, seed);
    int skip = 100;
    double sum_sq = 0.0;
    int count = 0;

    for (int t = 0; t < n_ticks; t++) {
        MbpfResult r = mbpf_step(s, (float)returns[t]);
        if (t >= skip) {
            double err = (double)r.h_mean - true_h[t];
            sum_sq += err * err;
            count++;
        }
    }

    mbpf_destroy(s);
    return (count > 0) ? sqrt(sum_sq / count) : 0.0;
}
