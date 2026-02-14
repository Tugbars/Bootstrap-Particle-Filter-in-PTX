/**
 * @file smc2_rbpf_cuda.cu
 * @brief SMC² with RBPF Inner Filter - Kernel Implementations (4-param version)
 * 
 * Learned parameters: rho, sigma_z, mu_base, sigma_base
 * Fixed parameters:   mu_scale, mu_rate, sigma_scale, sigma_rate (in constant memory)
 */

#include "smc2_rbpf_batch.cuh"
#include "smc2_noise_precision.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <curand.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

/*═══════════════════════════════════════════════════════════════════════════════
 * OCSN CONSTANT MEMORY
 *═══════════════════════════════════════════════════════════════════════════════*/

__device__ __constant__ float d_OCSN_WEIGHTS[OCSN_K] = {
    0.00609f, 0.04775f, 0.13057f, 0.20674f, 0.22715f,
    0.18842f, 0.12047f, 0.05591f, 0.01575f, 0.00115f
};
__device__ __constant__ float d_OCSN_MEANS[OCSN_K] = {
    1.92677f,  1.34744f,  0.73504f,  0.02266f, -0.85173f,
   -1.97278f, -3.46788f, -5.55246f, -8.68384f, -14.65000f
};
__device__ __constant__ float d_OCSN_VARS[OCSN_K] = {
    0.11265f, 0.17788f, 0.26768f, 0.40611f, 0.62699f,
    0.98583f, 1.57469f, 2.54498f, 4.16591f, 7.33342f
};
__device__ __constant__ float d_OCSN_LOG_WEIGHTS[OCSN_K] = {
    -5.1011072f, -3.0417762f, -2.0358458f, -1.5762933f, -1.4821447f,
    -1.6690818f, -2.1163545f, -2.8840120f, -4.1509149f, -6.7679933f
};
__device__ __constant__ float d_OCSN_INV_VARS[OCSN_K] = {
    8.87705282f, 5.62176748f, 3.73580395f, 2.46238704f, 1.59492177f,
    1.01437367f, 0.63504563f, 0.39293040f, 0.24004359f, 0.13636202f
};
__device__ __constant__ float d_OCSN_LOG_VARS[OCSN_K] = {
    -2.18346961f, -1.72664611f, -1.31796304f, -0.90113122f, -0.46682469f,
    -0.01427135f,  0.45405843f,  0.93412279f,  1.42693474f,  1.99244198f
};

/*═══════════════════════════════════════════════════════════════════════════════
 * OCSN Kalman Update (Marginalized)
 *
 * Computes posterior E[h|y] and Var[h|y] by moment-matching over the
 * 10-component mixture. Deterministic — no sampling.
 *═══════════════════════════════════════════════════════════════════════════════*/

__device__
void ocsn_kalman_update(
    float y,
    float mu_pred,
    float var_pred,
    float* mu_post,
    float* var_post,
    float* log_lik
) {
    float safe_var = fmaxf(var_pred, 1e-6f);
    
    float log_alpha_tilde[OCSN_K];
    float log_max = -1e30f;
    
    #pragma unroll
    for (int k = 0; k < OCSN_K; k++) {
        float v_k = d_OCSN_VARS[k];
        float inv_v_k = d_OCSN_INV_VARS[k];
        float log_v_k = d_OCSN_LOG_VARS[k];
        
        float S = safe_var + v_k;
        float inv_S = 1.0f / S;
        float innov = y - mu_pred - d_OCSN_MEANS[k];
        float log_S = log_v_k + log1pf(safe_var * inv_v_k);
        
        float val = d_OCSN_LOG_WEIGHTS[k] - 0.5f * (log_S + innov * innov * inv_S);
        log_alpha_tilde[k] = val;
        log_max = fmaxf(log_max, val);
    }
    
    float sum_exp = 0.0f;
    #pragma unroll
    for (int k = 0; k < OCSN_K; k++) {
        sum_exp += __expf(log_alpha_tilde[k] - log_max);
    }
    float log_norm = log_max + __logf(sum_exp);
    
    float mu_out = 0.0f;
    float E_h_sq = 0.0f;
    
    #pragma unroll
    for (int k = 0; k < OCSN_K; k++) {
        float w = __expf(log_alpha_tilde[k] - log_norm);
        
        float S = safe_var + d_OCSN_VARS[k];
        float inv_S = 1.0f / S;
        float innov = y - mu_pred - d_OCSN_MEANS[k];
        float K = safe_var * inv_S;
        
        float mu_k = mu_pred + K * innov;
        float var_k = (1.0f - K) * safe_var;
        
        mu_out += w * mu_k;
        E_h_sq += w * (var_k + mu_k * mu_k);
    }
    
    *mu_post = mu_out;
    *var_post = fmaxf(E_h_sq - mu_out * mu_out, 1e-6f);
    *log_lik = log_norm;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * CONSTANT MEMORY — Model parameters
 *═══════════════════════════════════════════════════════════════════════════════*/

__constant__ SVPrior       d_prior;
__constant__ SVBounds      d_bounds;
__constant__ SVCurve       d_theta_curve;
__constant__ SVFixedCurves d_fixed_curves;           /* NEW: fixed curve shapes */
__constant__ float         d_proposal_std[N_PARAMS];
__constant__ float         d_proposal_chol[N_PARAMS * N_PARAMS];

/*═══════════════════════════════════════════════════════════════════════════════
 * LOG PRIOR — 4 parameters only
 *═══════════════════════════════════════════════════════════════════════════════*/

__device__ float log_prior_theta(float rho, float sigma_z, float mu_base, float sigma_base) {
    if (rho < d_bounds.rho_min || rho > d_bounds.rho_max) return -INFINITY;
    if (sigma_z < d_bounds.sigma_z_min || sigma_z > d_bounds.sigma_z_max) return -INFINITY;
    if (mu_base < d_bounds.mu_base_min || mu_base > d_bounds.mu_base_max) return -INFINITY;
    if (sigma_base < d_bounds.sigma_base_min || sigma_base > d_bounds.sigma_base_max) return -INFINITY;
    
    float d_rho = (rho - d_prior.rho_mean) / d_prior.rho_std;
    float d_sz  = (sigma_z - d_prior.sigma_z_mean) / d_prior.sigma_z_std;
    float d_mb  = (mu_base - d_prior.mu_base_mean) / d_prior.mu_base_std;
    float d_sb  = (sigma_base - d_prior.sigma_base_mean) / d_prior.sigma_base_std;
    
    return -0.5f * (d_rho*d_rho + d_sz*d_sz + d_mb*d_mb + d_sb*d_sb);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: RNG Initialization
 *═══════════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_init_rng(curandState* states, unsigned long long seed, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) curand_init(seed, idx, 0, &states[idx]);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: Initialize from Prior
 *═══════════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_init_from_prior(
    ThetaParticlesSoA particles,
    int N_theta, int N_inner,
    noise_t* d_z_noise, noise_t* d_u0_noise,
    int noise_capacity
) {
    int theta_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    int global_idx = theta_idx * N_inner + inner_idx;
    
    if (theta_idx >= N_theta) return;
    
    curandState* rng = &particles.rng_states[global_idx];
    
    __shared__ float s_rho, s_sigma_z, s_mu_base, s_sigma_base;
    
    if (inner_idx == 0) {
        int attempts = 0, valid = 0;
        while (!valid && attempts < 1000) {
            s_rho = d_prior.rho_mean + d_prior.rho_std * curand_normal(rng);
            s_sigma_z = d_prior.sigma_z_mean + d_prior.sigma_z_std * curand_normal(rng);
            s_mu_base = d_prior.mu_base_mean + d_prior.mu_base_std * curand_normal(rng);
            s_sigma_base = d_prior.sigma_base_mean + d_prior.sigma_base_std * curand_normal(rng);
            
            valid = (s_rho >= d_bounds.rho_min && s_rho <= d_bounds.rho_max &&
                     s_sigma_z >= d_bounds.sigma_z_min && s_sigma_z <= d_bounds.sigma_z_max &&
                     s_mu_base >= d_bounds.mu_base_min && s_mu_base <= d_bounds.mu_base_max &&
                     s_sigma_base >= d_bounds.sigma_base_min && s_sigma_base <= d_bounds.sigma_base_max);
            attempts++;
        }
        
        particles.rho[theta_idx] = s_rho;
        particles.sigma_z[theta_idx] = s_sigma_z;
        particles.mu_base[theta_idx] = s_mu_base;
        particles.sigma_base[theta_idx] = s_sigma_base;
        
        particles.log_weight[theta_idx] = 0.0f;
        particles.weight[theta_idx] = 1.0f / N_theta;
        particles.log_likelihood[theta_idx] = 0.0f;
        particles.ess_inner[theta_idx] = (float)N_inner;
    }
    __syncthreads();
    
    float rho = s_rho;
    float sigma_z = s_sigma_z;
    
    float one_minus_rho_sq = fmaxf(1.0f - rho * rho, 1e-6f);
    float z_tilde_stat_std = sigma_z / sqrtf(one_minus_rho_sq);
    
    float z_noise_raw = curand_normal(rng);
    int64_t z_noise_idx = (int64_t)theta_idx * N_inner * (noise_capacity + 1) + inner_idx;
    float z_noise_init = noise_store_roundtrip(d_z_noise, z_noise_idx, z_noise_raw);
    
    if (inner_idx == 0) {
        float u0_noise_raw = curand_normal(rng);
        int64_t u0_noise_idx = (int64_t)theta_idx * (noise_capacity + 1);
        noise_store(d_u0_noise, u0_noise_idx, u0_noise_raw);
    }
    
    float z_tilde = z_tilde_stat_std * z_noise_init;
    float z = z_tilde_to_z(z_tilde);
    
    /* Fixed curve shapes from constant memory, learned bases from shared */
    float theta_z = eval_curve(d_theta_curve.base, d_theta_curve.scale, d_theta_curve.rate, z);
    float mu_z = eval_curve(s_mu_base, d_fixed_curves.mu_scale, d_fixed_curves.mu_rate, z);
    float sigma_h = eval_curve(s_sigma_base, d_fixed_curves.sigma_scale, d_fixed_curves.sigma_rate, z);
    
    float phi = 1.0f - theta_z;
    float one_minus_phi_sq = fmaxf(1.0f - phi * phi, 1e-6f);
    float h_stat_var = (sigma_h * sigma_h) / one_minus_phi_sq;
    
    particles.inner_z[global_idx] = z_tilde;
    particles.inner_mu_h[global_idx] = mu_z;
    particles.inner_var_h[global_idx] = h_stat_var;
    particles.inner_log_w[global_idx] = -__logf((float)N_inner);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: RBPF Forward Step
 *═══════════════════════════════════════════════════════════════════════════════*/

template<int N_INNER>
__global__ 
__launch_bounds__(N_INNER)
void kernel_rbpf_step_impl(
    ThetaParticlesSoA particles,
    float y_obs,
    int N_theta,
    noise_t* d_z_noise, noise_t* d_u0_noise,
    int t_current, int noise_capacity
) {
    static_assert(N_INNER <= 1024, "N_INNER must be <= 1024");
    
    int theta_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    int global_idx = theta_idx * N_INNER + inner_idx;
    
    if (theta_idx >= N_theta || inner_idx >= N_INNER) return;
    
    extern __shared__ char shared_raw[];
    float* s_reduction = reinterpret_cast<float*>(shared_raw);
    float* s_z = &s_reduction[32];
    float* s_mu = &s_z[N_INNER];
    float* s_var = &s_mu[N_INNER];
    float* s_cumsum = &s_var[N_INNER];
    int* s_idx = reinterpret_cast<int*>(&s_cumsum[N_INNER]);
    void* s_cub_temp = reinterpret_cast<void*>(&s_idx[N_INNER]);
    
    __shared__ float s_rho, s_sigma_z, s_mu_base, s_sigma_base;
    __shared__ float s_log_max, s_sum_w, s_u0;
    
    if (inner_idx == 0) {
        s_rho = particles.rho[theta_idx];
        s_sigma_z = particles.sigma_z[theta_idx];
        s_mu_base = particles.mu_base[theta_idx];
        s_sigma_base = particles.sigma_base[theta_idx];
    }
    __syncthreads();
    
    curandState local_rng = particles.rng_states[global_idx];
    
    float z_tilde = particles.inner_z[global_idx];
    float mu_h = particles.inner_mu_h[global_idx];
    float var_h = particles.inner_var_h[global_idx];
    float log_w = particles.inner_log_w[global_idx];
    
    int64_t z_noise_base = (int64_t)theta_idx * N_INNER * (noise_capacity + 1);
    int64_t z_noise_idx = z_noise_base + (int64_t)(t_current + 1) * N_INNER + inner_idx;
    int64_t u0_noise_idx = (int64_t)theta_idx * (noise_capacity + 1) + (t_current + 1);
    
    float z_noise_raw = curand_normal(&local_rng);
    float z_noise = noise_store_roundtrip(d_z_noise, z_noise_idx, z_noise_raw);
    
    if (inner_idx == 0) {
        float u0_noise_raw = curand_normal(&local_rng);
        float u0_stored = noise_store_roundtrip(d_u0_noise, u0_noise_idx, u0_noise_raw);
        s_u0 = u0_from_noise(u0_stored);
    }
    __syncthreads();
    
    /* RESAMPLE */
    {
        s_z[inner_idx] = z_tilde;
        s_mu[inner_idx] = mu_h;
        s_var[inner_idx] = var_h;
        __syncthreads();
        
        float log_max = block_reduce_max(log_w, s_reduction);
        if (inner_idx == 0) s_log_max = log_max;
        __syncthreads();
        
        float w_unnorm = __expf(log_w - s_log_max);
        float sum_w = block_reduce_sum(w_unnorm, s_reduction);
        if (inner_idx == 0) s_sum_w = sum_w;
        __syncthreads();
        
        s_cumsum[inner_idx] = w_unnorm / s_sum_w;
        __syncthreads();
        block_inclusive_scan(s_cumsum, N_INNER);
        if (inner_idx == N_INNER - 1) s_cumsum[N_INNER - 1] = 1.0f;
        __syncthreads();
        
        float u = (s_u0 + (float)inner_idx) / (float)N_INNER;
        int lo = 0, hi = N_INNER - 1;
        while (lo < hi) { int mid = (lo+hi)/2; if (s_cumsum[mid] < u) lo = mid+1; else hi = mid; }
        
        z_tilde = s_z[lo];
        mu_h = s_mu[lo];
        var_h = s_var[lo];
        log_w = -__logf((float)N_INNER);
        __syncthreads();
        
        if ((t_current % SORT_EVERY_K) == 0) {
            s_z[inner_idx] = z_tilde;
            s_mu[inner_idx] = mu_h;
            s_var[inner_idx] = var_h;
            __syncthreads();
            cpmmh_sort<N_INNER>(s_z, s_mu, s_var, s_idx, s_cub_temp);
            z_tilde = s_z[inner_idx];
            mu_h = s_mu[inner_idx];
            var_h = s_var[inner_idx];
            __syncthreads();
        }
    }
    
    /* PROPAGATE z̃ */
    float z_tilde_new = s_rho * z_tilde + s_sigma_z * z_noise;
    float z = z_tilde_to_z(z_tilde_new);
    
    /* KALMAN PREDICT — learned bases + fixed shapes from constant memory */
    float theta_z = eval_curve(d_theta_curve.base, d_theta_curve.scale, d_theta_curve.rate, z);
    float mu_z = eval_curve(s_mu_base, d_fixed_curves.mu_scale, d_fixed_curves.mu_rate, z);
    float sigma_h = eval_curve(s_sigma_base, d_fixed_curves.sigma_scale, d_fixed_curves.sigma_rate, z);
    
    float phi = 1.0f - theta_z;
    float mu_pred = phi * mu_h + theta_z * mu_z;
    float var_pred = phi * phi * var_h + sigma_h * sigma_h;
    var_pred = fmaxf(var_pred, 1e-8f);
    
    /* OCSN UPDATE */
    float mu_post, var_post, log_lik;
    ocsn_kalman_update(y_obs, mu_pred, var_pred, &mu_post, &var_post, &log_lik);
    log_w += log_lik;
    
    /* NORMALIZE + ESS */
    float log_max = block_reduce_max(log_w, s_reduction);
    if (inner_idx == 0) s_log_max = log_max;
    __syncthreads();
    
    float w_unnorm = __expf(log_w - s_log_max);
    float sum_w = block_reduce_sum(w_unnorm, s_reduction);
    if (inner_idx == 0) s_sum_w = sum_w;
    __syncthreads();
    
    float w_norm = w_unnorm / fmaxf(s_sum_w, 1e-30f);
    float w_sq = w_norm * w_norm;
    float sum_w_sq = block_reduce_sum(w_sq, s_reduction);
    float ess = 1.0f / fmaxf(sum_w_sq, 1e-30f);
    
    float ll_incr = s_log_max + __logf(fmaxf(s_sum_w, 1e-30f)) - __logf((float)N_INNER);
    
    particles.inner_z[global_idx] = z_tilde_new;
    particles.inner_mu_h[global_idx] = mu_post;
    particles.inner_var_h[global_idx] = var_post;
    particles.inner_log_w[global_idx] = log_w;
    particles.rng_states[global_idx] = local_rng;
    
    if (inner_idx == 0) {
        particles.ess_inner[theta_idx] = ess;
        particles.log_weight[theta_idx] += ll_incr;
        particles.log_likelihood[theta_idx] += ll_incr;
    }
}

/* Wrapper */
__global__ void kernel_rbpf_step(
    ThetaParticlesSoA particles, float y_obs,
    int N_theta, int N_inner,
    noise_t* d_z_noise, noise_t* d_u0_noise,
    int t_current, int noise_capacity
) { /* Dispatch via template from host */ }

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: Reset Outer Weights
 *═══════════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_reset_outer_weights(ThetaParticlesSoA particles, int N_theta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_theta) return;
    particles.log_weight[idx] = 0.0f;
    particles.weight[idx] = 1.0f / (float)N_theta;
    particles.log_likelihood[idx] = 0.0f;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: Compute Particle Moments (4D)
 *═══════════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_compute_particle_moments(
    ThetaParticlesSoA particles,
    float* d_mean,
    float* d_cov,
    int N_theta
) {
    extern __shared__ float s_data[];
    float* s_scratch = s_data;
    int tid = threadIdx.x;
    
    float p[N_PARAMS];
    if (tid < N_theta) {
        p[0] = particles.rho[tid];
        p[1] = particles.sigma_z[tid];
        p[2] = particles.mu_base[tid];
        p[3] = particles.sigma_base[tid];
    } else {
        for (int i = 0; i < N_PARAMS; i++) p[i] = 0.0f;
    }
    
    __shared__ float s_mean[N_PARAMS];
    for (int i = 0; i < N_PARAMS; i++) {
        s_scratch[tid] = (tid < N_theta) ? p[i] : 0.0f;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) s_scratch[tid] += s_scratch[tid + s];
            __syncthreads();
        }
        if (tid == 0) s_mean[i] = s_scratch[0] / (float)N_theta;
        __syncthreads();
    }
    
    if (tid < N_PARAMS) d_mean[tid] = s_mean[tid];
    
    float c[N_PARAMS];
    for (int i = 0; i < N_PARAMS; i++) c[i] = p[i] - s_mean[i];
    
    float inv_N_1 = 1.0f / (float)(N_theta - 1);
    
    for (int i = 0; i < N_PARAMS; i++) {
        for (int j = 0; j <= i; j++) {
            float prod = (tid < N_theta) ? (c[i] * c[j]) : 0.0f;
            s_scratch[tid] = prod;
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) s_scratch[tid] += s_scratch[tid + s];
                __syncthreads();
            }
            if (tid == 0) {
                float cov_ij = s_scratch[0] * inv_N_1;
                d_cov[i * N_PARAMS + j] = cov_ij;
                d_cov[j * N_PARAMS + i] = cov_ij;
            }
            __syncthreads();
        }
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: Compute Outer ESS
 *═══════════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_compute_outer_ess(
    ThetaParticlesSoA particles, float* d_ess_out, int N_theta
) {
    extern __shared__ float s_data[];
    int idx = threadIdx.x;
    
    float log_w = (idx < N_theta) ? particles.log_weight[idx] : -1e30f;
    
    float log_max = block_reduce_max(log_w, s_data);
    __shared__ float s_log_max;
    if (idx == 0) s_log_max = log_max;
    __syncthreads();
    
    float w = (idx < N_theta) ? __expf(log_w - s_log_max) : 0.0f;
    float sum_w = block_reduce_sum(w, s_data);
    __shared__ float s_sum_w;
    if (idx == 0) s_sum_w = sum_w;
    __syncthreads();
    
    if (idx < N_theta) {
        w /= s_sum_w;
        particles.weight[idx] = w;
    }
    
    float w_sq = (idx < N_theta) ? w * w : 0.0f;
    float sum_w_sq = block_reduce_sum(w_sq, s_data);
    if (idx == 0) *d_ess_out = 1.0f / fmaxf(sum_w_sq, 1e-30f);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: Outer Resampling
 *═══════════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_outer_resample(
    ThetaParticlesSoA particles, int* d_ancestors, float* d_uniform, int N_theta
) {
    extern __shared__ float s_cumsum[];
    int idx = threadIdx.x;
    
    if (idx < N_theta) s_cumsum[idx] = particles.weight[idx];
    __syncthreads();
    
    if (idx == 0) {
        for (int i = 1; i < N_theta; i++) s_cumsum[i] += s_cumsum[i-1];
        s_cumsum[N_theta - 1] = 1.0f;
    }
    __syncthreads();
    
    if (idx < N_theta) {
        float u = (*d_uniform + (float)idx) / (float)N_theta;
        int lo = 0, hi = N_theta - 1;
        while (lo < hi) { int mid = (lo+hi)/2; if (s_cumsum[mid] < u) lo = mid+1; else hi = mid; }
        d_ancestors[idx] = lo;
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: Copy θ-Particles After Resampling (4 learned params)
 *═══════════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_copy_theta_particles(
    ThetaParticlesSoA src, ThetaParticlesSoA dst, int* d_ancestors,
    int N_theta, int N_inner, unsigned long long resample_seed
) {
    int theta_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    
    if (theta_idx >= N_theta) return;
    
    int ancestor = d_ancestors[theta_idx];
    
    if (inner_idx == 0) {
        dst.rho[theta_idx] = src.rho[ancestor];
        dst.sigma_z[theta_idx] = src.sigma_z[ancestor];
        dst.mu_base[theta_idx] = src.mu_base[ancestor];
        dst.sigma_base[theta_idx] = src.sigma_base[ancestor];
        
        dst.log_weight[theta_idx] = 0.0f;
        dst.weight[theta_idx] = 1.0f / N_theta;
        dst.log_likelihood[theta_idx] = src.log_likelihood[ancestor];
        dst.ess_inner[theta_idx] = src.ess_inner[ancestor];
    }
    
    if (inner_idx < N_inner) {
        int src_idx = ancestor * N_inner + inner_idx;
        int dst_idx = theta_idx * N_inner + inner_idx;
        
        dst.inner_z[dst_idx] = src.inner_z[src_idx];
        dst.inner_mu_h[dst_idx] = src.inner_mu_h[src_idx];
        dst.inner_var_h[dst_idx] = src.inner_var_h[src_idx];
        dst.inner_log_w[dst_idx] = src.inner_log_w[src_idx];
        
        curand_init(resample_seed, dst_idx, 0, &dst.rng_states[dst_idx]);
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: Copy Noise Arrays (Ping-Pong) — unchanged
 *═══════════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_copy_noise_arrays(
    const noise_t* src_z_noise, noise_t* dst_z_noise,
    const noise_t* src_u0_noise, noise_t* dst_u0_noise,
    const int* d_ancestors,
    int N_theta, int N_inner, int t_current, int noise_capacity
) {
    int theta_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    
    if (theta_idx >= N_theta || inner_idx >= N_inner) return;
    
    int ancestor = d_ancestors[theta_idx];
    
    int64_t dst_z_base = (int64_t)theta_idx * N_inner * (noise_capacity + 1);
    int64_t src_z_base = (int64_t)ancestor * N_inner * (noise_capacity + 1);
    
    for (int t = 0; t <= t_current + 1; t++) {
        int64_t src_idx = src_z_base + t * N_inner + inner_idx;
        int64_t dst_idx = dst_z_base + t * N_inner + inner_idx;
        dst_z_noise[dst_idx] = src_z_noise[src_idx];
    }
    
    if (inner_idx == 0) {
        int64_t dst_u0_base = (int64_t)theta_idx * (noise_capacity + 1);
        int64_t src_u0_base = (int64_t)ancestor * (noise_capacity + 1);
        for (int t = 0; t <= t_current + 1; t++) {
            dst_u0_noise[dst_u0_base + t] = src_u0_noise[src_u0_base + t];
        }
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: Copy Checkpoint Arrays — unchanged
 *═══════════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_copy_checkpoint(
    const float* src_z, const float* src_mu_h, const float* src_var_h,
    const float* src_log_w, const float* src_ll,
    float* dst_z, float* dst_mu_h, float* dst_var_h,
    float* dst_log_w, float* dst_ll,
    const int* d_ancestors, int N_theta, int N_inner
) {
    int theta_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    if (theta_idx >= N_theta || inner_idx >= N_inner) return;
    
    int ancestor = d_ancestors[theta_idx];
    int src_global = ancestor * N_inner + inner_idx;
    int dst_global = theta_idx * N_inner + inner_idx;
    
    dst_z[dst_global] = src_z[src_global];
    dst_mu_h[dst_global] = src_mu_h[src_global];
    dst_var_h[dst_global] = src_var_h[src_global];
    dst_log_w[dst_global] = src_log_w[src_global];
    
    if (inner_idx == 0) dst_ll[theta_idx] = src_ll[ancestor];
}

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: CPMMH Fused Rejuvenation (4D proposal)
 *═══════════════════════════════════════════════════════════════════════════════*/

template<int N_INNER>
__global__
__launch_bounds__(N_INNER)
void kernel_cpmmh_rejuvenate_fused_impl(
    ThetaParticlesSoA particles,
    ThetaParticlesSoA particles_scratch,
    const float* y_history,
    noise_t* d_z_noise_curr, noise_t* d_z_noise_other,
    noise_t* d_u0_noise_curr, noise_t* d_u0_noise_other,
    int t_current, int N_theta, int noise_capacity,
    float cpmmh_rho,
    int* d_accepts, int* d_swap_flags,
    unsigned long long seed, int move_id, int block_id,
    int t_checkpoint,
    const float* d_checkpoint_z, const float* d_checkpoint_mu_h,
    const float* d_checkpoint_var_h, const float* d_checkpoint_log_w,
    const float* d_checkpoint_ll
) {
    static_assert(N_INNER <= 1024, "N_INNER must be <= 1024");
    (void)seed; (void)move_id; (void)block_id;
    
    int theta_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    int global_idx = theta_idx * N_INNER + inner_idx;
    
    if (theta_idx >= N_theta || inner_idx >= N_INNER) return;
    
    extern __shared__ char shared_raw[];
    float* s_reduction = reinterpret_cast<float*>(shared_raw);
    float* s_z = &s_reduction[32];
    float* s_mu = &s_z[N_INNER];
    float* s_var = &s_mu[N_INNER];
    float* s_cdf = &s_var[N_INNER];
    int* s_idx = reinterpret_cast<int*>(&s_cdf[N_INNER]);
    void* s_cub_temp = reinterpret_cast<void*>(&s_idx[N_INNER]);
    
    __shared__ float s_log_max, s_sum_w, s_ess_prop;
    __shared__ float s_rho_curr, s_sigma_z_curr, s_mu_base_curr, s_sigma_base_curr;
    __shared__ float s_rho_prop, s_sigma_z_prop, s_mu_base_prop, s_sigma_base_prop;
    __shared__ float s_ll_curr, s_ll_prop, s_lp_curr, s_lp_prop;
    __shared__ int s_accept, s_valid;
    __shared__ float s_u0_shared;
    
    curandState local_rng = particles.rng_states[global_idx];
    
    /* PROPOSE θ* (thread 0) — 4D random walk */
    if (inner_idx == 0) {
        s_rho_curr = particles.rho[theta_idx];
        s_sigma_z_curr = particles.sigma_z[theta_idx];
        s_mu_base_curr = particles.mu_base[theta_idx];
        s_sigma_base_curr = particles.sigma_base[theta_idx];
        
        s_ll_curr = particles.log_likelihood[theta_idx];
        s_lp_curr = log_prior_theta(s_rho_curr, s_sigma_z_curr, s_mu_base_curr, s_sigma_base_curr);
        
        float z_rnd[N_PARAMS];
        for (int i = 0; i < N_PARAMS; i++) z_rnd[i] = curand_normal(&local_rng);
        
        float pert[N_PARAMS] = {0};
        float mix_u = curand_uniform(&local_rng);
        
        if (mix_u > 0.05f) {
            /* 95%: Adaptive correlated proposal via Cholesky */
            for (int i = 0; i < N_PARAMS; i++) {
                float sum = 0.0f;
                for (int j = 0; j <= i; j++)
                    sum += d_proposal_chol[i * N_PARAMS + j] * z_rnd[j];
                pert[i] = sum;
            }
        } else {
            /* 5%: Fixed independent for ergodicity */
            for (int i = 0; i < N_PARAMS; i++)
                pert[i] = d_proposal_std[i] * z_rnd[i];
        }
        
        s_rho_prop = s_rho_curr + pert[0];
        s_sigma_z_prop = s_sigma_z_curr + pert[1];
        s_mu_base_prop = s_mu_base_curr + pert[2];
        s_sigma_base_prop = s_sigma_base_curr + pert[3];
        
        s_lp_prop = log_prior_theta(s_rho_prop, s_sigma_z_prop, s_mu_base_prop, s_sigma_base_prop);
        s_valid = isfinite(s_lp_prop) ? 1 : 0;
        s_accept = 0;
    }
    __syncthreads();
    
    if (s_valid == 0) {
        if (inner_idx == 0) d_swap_flags[theta_idx] = 0;
        particles.rng_states[global_idx] = local_rng;
        return;
    }
    
    int64_t z_noise_base = (int64_t)theta_idx * N_INNER * (noise_capacity + 1);
    float scale = sqrtf(1.0f - cpmmh_rho * cpmmh_rho);
    
    /* Use proposed learned params, fixed shapes from constant memory */
    float rho = s_rho_prop;
    float sigma_z = s_sigma_z_prop;
    float mu_base = s_mu_base_prop;
    float sigma_base = s_sigma_base_prop;
    
    float z_tilde, mu_h, var_h, log_w;
    float ll_accum = 0.0f;
    int t_start;
    
    if (t_checkpoint >= 0 && d_checkpoint_z != nullptr) {
        z_tilde = d_checkpoint_z[global_idx];
        mu_h = d_checkpoint_mu_h[global_idx];
        var_h = d_checkpoint_var_h[global_idx];
        log_w = d_checkpoint_log_w[global_idx];
        t_start = t_checkpoint + 1;
    } else {
        float one_minus_rho_sq = fmaxf(1.0f - rho * rho, 1e-6f);
        float z_tilde_stat_std = sigma_z / sqrtf(one_minus_rho_sq);
        
        float z_noise_curr_0 = noise_load(d_z_noise_curr, z_noise_base + inner_idx);
        float z_noise_fresh_0 = curand_normal(&local_rng);
        float z_noise_prop_0 = cpmmh_rho * z_noise_curr_0 + scale * z_noise_fresh_0;
        noise_store(d_z_noise_other, z_noise_base + inner_idx, z_noise_prop_0);
        
        z_tilde = z_tilde_stat_std * z_noise_prop_0;
        float z_init = z_tilde_to_z(z_tilde);
        
        float theta_z_init = eval_curve(d_theta_curve.base, d_theta_curve.scale, d_theta_curve.rate, z_init);
        float mu_z_init = eval_curve(mu_base, d_fixed_curves.mu_scale, d_fixed_curves.mu_rate, z_init);
        float sigma_h_init = eval_curve(sigma_base, d_fixed_curves.sigma_scale, d_fixed_curves.sigma_rate, z_init);
        float phi_init = 1.0f - theta_z_init;
        float h_stat_var = (sigma_h_init * sigma_h_init) / fmaxf(1.0f - phi_init * phi_init, 1e-6f);
        
        mu_h = mu_z_init;
        var_h = h_stat_var;
        log_w = -__logf((float)N_INNER);
        t_start = 0;
    }
    
    /* Replay observations */
    for (int t = t_start; t <= t_current; t++) {
        float y_obs = y_history[t];
        
        float log_max = block_reduce_max(log_w, s_reduction);
        if (inner_idx == 0) s_log_max = log_max;
        __syncthreads();
        log_max = s_log_max;
        
        float w_unnorm = __expf(log_w - log_max);
        float sum_w = block_reduce_sum(w_unnorm, s_reduction);
        if (inner_idx == 0) s_sum_w = sum_w;
        __syncthreads();
        sum_w = s_sum_w;
        
        float w_norm = w_unnorm / fmaxf(sum_w, 1e-30f);
        
        s_z[inner_idx] = z_tilde;
        s_mu[inner_idx] = mu_h;
        s_var[inner_idx] = var_h;
        s_cdf[inner_idx] = w_norm;
        __syncthreads();
        
        block_inclusive_scan(s_cdf, N_INNER);
        if (inner_idx == N_INNER - 1) s_cdf[N_INNER - 1] = 1.0f;
        __syncthreads();
        
        int64_t z_idx_t1 = z_noise_base + (int64_t)(t + 1) * N_INNER + inner_idx;
        float z_noise_curr_t1 = noise_load(d_z_noise_curr, z_idx_t1);
        float z_noise_fresh_t1 = curand_normal(&local_rng);
        float z_noise_prop_t1_raw = cpmmh_rho * z_noise_curr_t1 + scale * z_noise_fresh_t1;
        float z_noise_prop_t1 = noise_store_roundtrip(d_z_noise_other, z_idx_t1, z_noise_prop_t1_raw);
        
        if (inner_idx == 0) {
            int64_t u0_idx_t1 = (int64_t)theta_idx * (noise_capacity + 1) + (t + 1);
            float u0_noise_curr = noise_load(d_u0_noise_curr, u0_idx_t1);
            float u0_noise_fresh = curand_normal(&local_rng);
            float u0_noise_prop_raw = cpmmh_rho * u0_noise_curr + scale * u0_noise_fresh;
            float u0_stored = noise_store_roundtrip(d_u0_noise_other, u0_idx_t1, u0_noise_prop_raw);
            s_u0_shared = u0_from_noise(u0_stored);
        }
        __syncthreads();
        
        float u = (s_u0_shared + (float)inner_idx) / (float)N_INNER;
        int lo = 0, hi = N_INNER - 1;
        while (lo < hi) { int mid = (lo+hi)/2; if (s_cdf[mid] < u) lo = mid+1; else hi = mid; }
        
        z_tilde = s_z[lo];
        mu_h = s_mu[lo];
        var_h = s_var[lo];
        log_w = -__logf((float)N_INNER);
        __syncthreads();
        
        if ((t % SORT_EVERY_K) == 0) {
            s_z[inner_idx] = z_tilde;
            s_mu[inner_idx] = mu_h;
            s_var[inner_idx] = var_h;
            __syncthreads();
            cpmmh_sort<N_INNER>(s_z, s_mu, s_var, s_idx, s_cub_temp);
            z_tilde = s_z[inner_idx];
            mu_h = s_mu[inner_idx];
            var_h = s_var[inner_idx];
            __syncthreads();
        }
        
        /* Propagate + Kalman with fixed curve shapes */
        float z_tilde_new = rho * z_tilde + sigma_z * z_noise_prop_t1;
        float z = z_tilde_to_z(z_tilde_new);
        
        float theta_z = eval_curve(d_theta_curve.base, d_theta_curve.scale, d_theta_curve.rate, z);
        float mu_z_val = eval_curve(mu_base, d_fixed_curves.mu_scale, d_fixed_curves.mu_rate, z);
        float sigma_h = eval_curve(sigma_base, d_fixed_curves.sigma_scale, d_fixed_curves.sigma_rate, z);
        float phi = 1.0f - theta_z;
        
        float mu_pred = phi * mu_h + theta_z * mu_z_val;
        float var_pred = phi * phi * var_h + sigma_h * sigma_h;
        var_pred = fmaxf(var_pred, 1e-8f);
        
        float mu_post, var_post, log_lik;
        ocsn_kalman_update(y_obs, mu_pred, var_pred, &mu_post, &var_post, &log_lik);
        log_w += log_lik;
        
        log_max = block_reduce_max(log_w, s_reduction);
        if (inner_idx == 0) s_log_max = log_max;
        __syncthreads();
        log_max = s_log_max;
        
        w_unnorm = __expf(log_w - log_max);
        sum_w = block_reduce_sum(w_unnorm, s_reduction);
        if (inner_idx == 0) s_sum_w = sum_w;
        __syncthreads();
        sum_w = s_sum_w;
        
        float ll_incr = log_max + __logf(fmaxf(sum_w, 1e-30f)) - __logf((float)N_INNER);
        ll_accum += ll_incr;
        
        z_tilde = z_tilde_new;
        mu_h = mu_post;
        var_h = var_post;
    }
    
    /* Final ESS */
    float w_norm = __expf(log_w - s_log_max) / fmaxf(s_sum_w, 1e-30f);
    float w_sq = w_norm * w_norm;
    float sum_w_sq = block_reduce_sum(w_sq, s_reduction);
    float ess = 1.0f / fmaxf(sum_w_sq, 1e-30f);
    
    particles_scratch.inner_z[global_idx] = z_tilde;
    particles_scratch.inner_mu_h[global_idx] = mu_h;
    particles_scratch.inner_var_h[global_idx] = var_h;
    particles_scratch.inner_log_w[global_idx] = log_w;
    
    __shared__ float s_ll_base;
    
    if (inner_idx == 0) {
        float ll_base = (t_checkpoint >= 0 && d_checkpoint_ll != nullptr) 
                        ? d_checkpoint_ll[theta_idx] : 0.0f;
        s_ll_base = ll_base;
        s_ll_prop = ll_base + ll_accum;
        s_ess_prop = ess;
    }
    __syncthreads();
    
    /* MH ACCEPT/REJECT */
    if (inner_idx == 0) {
        float ll_curr_effective = s_ll_curr - s_ll_base;
        float ll_prop_effective = ll_accum;
        
        float log_alpha = (ll_prop_effective + s_lp_prop) - (ll_curr_effective + s_lp_curr);
        
        float u = curand_uniform(&local_rng);
        s_accept = (__logf(u) < log_alpha) ? 1 : 0;
        
        if (s_accept) {
            particles.rho[theta_idx] = s_rho_prop;
            particles.sigma_z[theta_idx] = s_sigma_z_prop;
            particles.mu_base[theta_idx] = s_mu_base_prop;
            particles.sigma_base[theta_idx] = s_sigma_base_prop;
            particles.log_likelihood[theta_idx] = s_ll_prop;
            particles.ess_inner[theta_idx] = s_ess_prop;
            atomicAdd(d_accepts, 1);
        }
        d_swap_flags[theta_idx] = s_accept;
    }
    __syncthreads();
    
    if (s_accept) {
        particles.inner_z[global_idx] = particles_scratch.inner_z[global_idx];
        particles.inner_mu_h[global_idx] = particles_scratch.inner_mu_h[global_idx];
        particles.inner_var_h[global_idx] = particles_scratch.inner_var_h[global_idx];
        particles.inner_log_w[global_idx] = particles_scratch.inner_log_w[global_idx];
    }
    
    particles.rng_states[global_idx] = local_rng;
}

/* Wrapper */
__global__ void kernel_cpmmh_rejuvenate_fused(
    ThetaParticlesSoA particles, ThetaParticlesSoA particles_scratch,
    const float* y_history,
    noise_t* d_z_noise_curr, noise_t* d_z_noise_other,
    noise_t* d_u0_noise_curr, noise_t* d_u0_noise_other,
    int t_current, int N_theta, int N_inner,
    int noise_capacity, float cpmmh_rho,
    int* d_accepts, int* d_swap_flags,
    unsigned long long seed, int move_id, int block_id,
    int t_checkpoint,
    const float* d_checkpoint_z, const float* d_checkpoint_mu_h,
    const float* d_checkpoint_var_h, const float* d_checkpoint_log_w,
    const float* d_checkpoint_ll
) { /* Use template version from host */ }

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: Commit Accepted Noise — unchanged
 *═══════════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_commit_accepted_noise(
    noise_t* d_z_noise_0, noise_t* d_z_noise_1,
    noise_t* d_u0_noise_0, noise_t* d_u0_noise_1,
    const int* d_swap_flags,
    int N_theta, int N_inner,
    int t_current, int noise_capacity, int t_start
) {
    int theta_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    if (theta_idx >= N_theta || inner_idx >= N_inner) return;
    if (d_swap_flags[theta_idx] == 0) return;
    
    int64_t z_base = (int64_t)theta_idx * N_inner * (noise_capacity + 1);
    for (int t = t_start; t <= t_current + 1; t++) {
        int64_t idx = z_base + t * N_inner + inner_idx;
        d_z_noise_0[idx] = d_z_noise_1[idx];
    }
    if (inner_idx == 0) {
        int64_t u0_base = (int64_t)theta_idx * (noise_capacity + 1);
        for (int t = t_start; t <= t_current + 1; t++)
            d_u0_noise_0[u0_base + t] = d_u0_noise_1[u0_base + t];
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: Save Checkpoint — unchanged
 *═══════════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_save_checkpoint(
    const ThetaParticlesSoA particles,
    float* d_checkpoint_z, float* d_checkpoint_mu_h,
    float* d_checkpoint_var_h, float* d_checkpoint_log_w, float* d_checkpoint_ll,
    int N_theta, int N_inner
) {
    int theta_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    int global_idx = theta_idx * N_inner + inner_idx;
    if (theta_idx >= N_theta || inner_idx >= N_inner) return;
    
    d_checkpoint_z[global_idx] = particles.inner_z[global_idx];
    d_checkpoint_mu_h[global_idx] = particles.inner_mu_h[global_idx];
    d_checkpoint_var_h[global_idx] = particles.inner_var_h[global_idx];
    d_checkpoint_log_w[global_idx] = particles.inner_log_w[global_idx];
    if (inner_idx == 0) d_checkpoint_ll[theta_idx] = particles.log_likelihood[theta_idx];
}

/*═══════════════════════════════════════════════════════════════════════════════
 * HOST API
 *═══════════════════════════════════════════════════════════════════════════════*/

static inline uint64_t xorshift64star(uint64_t* state) {
    uint64_t x = *state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static inline float xorshift64star_uniform(uint64_t* state) {
    return (float)((xorshift64star(state) >> 11) + 1) * (1.0f / 9007199254740994.0f);
}

#define ALLOC_THETA_FIELD(field) \
    CUDA_CHECK(cudaMalloc(&state->d_particles.field, N_theta * sizeof(float))); \
    CUDA_CHECK(cudaMalloc(&state->d_particles_temp.field, N_theta * sizeof(float)))

#define ALLOC_INNER_FIELD(field) \
    CUDA_CHECK(cudaMalloc(&state->d_particles.field, N_total * sizeof(float))); \
    CUDA_CHECK(cudaMalloc(&state->d_particles_temp.field, N_total * sizeof(float)))

SMC2StateCUDA* smc2_cuda_alloc(int N_theta, int N_inner) {
    SMC2StateCUDA* state = (SMC2StateCUDA*)calloc(1, sizeof(SMC2StateCUDA));
    if (!state) return NULL;
    
    state->N_theta = N_theta;
    state->N_inner = N_inner;
    state->ess_threshold_outer = 0.5f;
    state->ess_threshold_inner = 0.5f;
    state->K_rejuv = 5;
    
    int N_total = N_theta * N_inner;
    
    /* 4 learned θ-level params */
    ALLOC_THETA_FIELD(rho);
    ALLOC_THETA_FIELD(sigma_z);
    ALLOC_THETA_FIELD(mu_base);
    ALLOC_THETA_FIELD(sigma_base);
    ALLOC_THETA_FIELD(log_weight);
    ALLOC_THETA_FIELD(weight);
    ALLOC_THETA_FIELD(log_likelihood);
    ALLOC_THETA_FIELD(ess_inner);
    
    /* Inner RBPF arrays */
    ALLOC_INNER_FIELD(inner_z);
    ALLOC_INNER_FIELD(inner_mu_h);
    ALLOC_INNER_FIELD(inner_var_h);
    ALLOC_INNER_FIELD(inner_log_w);
    CUDA_CHECK(cudaMalloc(&state->d_particles.rng_states, N_total * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc(&state->d_particles_temp.rng_states, N_total * sizeof(curandState)));
    
    /* RNG init */
    kernel_init_rng<<<(N_total + 255) / 256, 256>>>(state->d_particles.rng_states, 12345ULL, N_total);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    /* Observation history */
    state->y_history_capacity = 8000;
    CUDA_CHECK(cudaMalloc(&state->d_y_history, state->y_history_capacity * sizeof(float)));
    state->y_history_len = 0;
    
    /* Scratch */
    CUDA_CHECK(cudaMalloc(&state->d_ancestors, N_theta * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state->d_uniform, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_ess, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_accepts, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state->d_swap_flags, N_theta * sizeof(int)));
    
    /* CPMMH noise */
    state->noise_capacity = 2048;
    state->cpmmh_rho = 0.99f;
    state->noise_buf = 0;
    state->user_seed = 0;
    state->host_rng_state = 0x853C49E6748FEA9BULL ^ (uint64_t)time(NULL);
    
    int64_t z_noise_size = (int64_t)N_theta * N_inner * (state->noise_capacity + 1);
    int64_t u0_noise_size = (int64_t)N_theta * (state->noise_capacity + 1);
    CUDA_CHECK(cudaMalloc(&state->d_z_noise[0], noise_array_bytes(z_noise_size)));
    CUDA_CHECK(cudaMalloc(&state->d_z_noise[1], noise_array_bytes(z_noise_size)));
    CUDA_CHECK(cudaMalloc(&state->d_u0_noise[0], noise_array_bytes(u0_noise_size)));
    CUDA_CHECK(cudaMalloc(&state->d_u0_noise[1], noise_array_bytes(u0_noise_size)));
    
    /* Prior (4 params) */
    state->prior.rho_mean = 0.95f;          state->prior.rho_std = 0.02f;
    state->prior.sigma_z_mean = 0.1f;       state->prior.sigma_z_std = 0.1f;
    state->prior.mu_base_mean = -1.0f;      state->prior.mu_base_std = 0.5f;
    state->prior.sigma_base_mean = 0.15f;   state->prior.sigma_base_std = 0.05f;
    
    /* Bounds (4 params) */
    state->bounds.rho_min = 0.8f;           state->bounds.rho_max = 0.999f;
    state->bounds.sigma_z_min = 0.01f;      state->bounds.sigma_z_max = 1.0f;
    state->bounds.mu_base_min = -10.0f;     state->bounds.mu_base_max = 5.0f;
    state->bounds.sigma_base_min = 0.01f;   state->bounds.sigma_base_max = 1.0f;
    
    /* Fixed curve shapes (calibrated offline) */
    state->fixed_curves.mu_scale = 0.5f;
    state->fixed_curves.mu_rate = 1.0f;
    state->fixed_curves.sigma_scale = 0.1f;
    state->fixed_curves.sigma_rate = 1.0f;
    
    /* Theta curve (fixed) */
    state->theta_curve.base = 0.02f;
    state->theta_curve.scale = 0.08f;
    state->theta_curve.rate = 1.5f;
    
    /* Proposal std (4 params) */
    state->proposal_std[0] = 0.01f;    /* rho */
    state->proposal_std[1] = 0.02f;    /* sigma_z */
    state->proposal_std[2] = 0.1f;     /* mu_base */
    state->proposal_std[3] = 0.02f;    /* sigma_base */
    
    /* Fixed-lag checkpoint */
    state->fixed_lag_L = 0;
    state->t_checkpoint = -1;
    CUDA_CHECK(cudaMalloc(&state->d_checkpoint_z, N_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_checkpoint_mu_h, N_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_checkpoint_var_h, N_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_checkpoint_log_w, N_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_checkpoint_ll, N_theta * sizeof(float)));
    
    /* Adaptive proposal scratch (4D) */
    CUDA_CHECK(cudaMalloc(&state->d_temp_mean, N_PARAMS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_temp_cov, N_PARAMS * N_PARAMS * sizeof(float)));
    state->use_adaptive_proposals = true;
    
    return state;
}

#undef ALLOC_THETA_FIELD
#undef ALLOC_INNER_FIELD

#define FREE_THETA_FIELD(field) \
    cudaFree(state->d_particles.field); \
    cudaFree(state->d_particles_temp.field)

#define FREE_INNER_FIELD(field) \
    cudaFree(state->d_particles.field); \
    cudaFree(state->d_particles_temp.field)

void smc2_cuda_free(SMC2StateCUDA* state) {
    if (!state) return;
    
    FREE_THETA_FIELD(rho);
    FREE_THETA_FIELD(sigma_z);
    FREE_THETA_FIELD(mu_base);
    FREE_THETA_FIELD(sigma_base);
    FREE_THETA_FIELD(log_weight);
    FREE_THETA_FIELD(weight);
    FREE_THETA_FIELD(log_likelihood);
    FREE_THETA_FIELD(ess_inner);
    FREE_INNER_FIELD(inner_z);
    FREE_INNER_FIELD(inner_mu_h);
    FREE_INNER_FIELD(inner_var_h);
    FREE_INNER_FIELD(inner_log_w);
    cudaFree(state->d_particles.rng_states);
    cudaFree(state->d_particles_temp.rng_states);
    
    cudaFree(state->d_y_history);
    cudaFree(state->d_ancestors);
    cudaFree(state->d_uniform);
    cudaFree(state->d_ess);
    cudaFree(state->d_accepts);
    cudaFree(state->d_swap_flags);
    
    cudaFree(state->d_z_noise[0]); cudaFree(state->d_z_noise[1]);
    cudaFree(state->d_u0_noise[0]); cudaFree(state->d_u0_noise[1]);
    
    cudaFree(state->d_checkpoint_z);
    cudaFree(state->d_checkpoint_mu_h);
    cudaFree(state->d_checkpoint_var_h);
    cudaFree(state->d_checkpoint_log_w);
    cudaFree(state->d_checkpoint_ll);
    
    cudaFree(state->d_temp_mean);
    cudaFree(state->d_temp_cov);
    
    free(state);
}

#undef FREE_THETA_FIELD
#undef FREE_INNER_FIELD

void smc2_cuda_set_seed(SMC2StateCUDA* state, uint64_t seed) {
    state->user_seed = seed;
    if (seed != 0) state->host_rng_state = 0x853C49E6748FEA9BULL ^ seed;
}

void smc2_cuda_set_noise_capacity(SMC2StateCUDA* state, int capacity) {
    if (capacity <= state->noise_capacity) return;
    
    int64_t new_z_size = (int64_t)state->N_theta * state->N_inner * (capacity + 1);
    int64_t old_z_size = (int64_t)state->N_theta * state->N_inner * (state->noise_capacity + 1);
    int64_t new_u0_size = (int64_t)state->N_theta * (capacity + 1);
    int64_t old_u0_size = (int64_t)state->N_theta * (state->noise_capacity + 1);
    
    noise_t *new_z_0, *new_z_1, *new_u0_0, *new_u0_1;
    CUDA_CHECK(cudaMalloc(&new_z_0, noise_array_bytes(new_z_size)));
    CUDA_CHECK(cudaMalloc(&new_z_1, noise_array_bytes(new_z_size)));
    CUDA_CHECK(cudaMalloc(&new_u0_0, noise_array_bytes(new_u0_size)));
    CUDA_CHECK(cudaMalloc(&new_u0_1, noise_array_bytes(new_u0_size)));
    
    if (state->d_z_noise[0] && old_z_size > 0)
        CUDA_CHECK(cudaMemcpy(new_z_0, state->d_z_noise[0], noise_array_bytes(old_z_size), cudaMemcpyDeviceToDevice));
    if (state->d_u0_noise[0] && old_u0_size > 0)
        CUDA_CHECK(cudaMemcpy(new_u0_0, state->d_u0_noise[0], noise_array_bytes(old_u0_size), cudaMemcpyDeviceToDevice));
    
    cudaFree(state->d_z_noise[0]); cudaFree(state->d_z_noise[1]);
    cudaFree(state->d_u0_noise[0]); cudaFree(state->d_u0_noise[1]);
    
    state->d_z_noise[0] = new_z_0; state->d_z_noise[1] = new_z_1;
    state->d_u0_noise[0] = new_u0_0; state->d_u0_noise[1] = new_u0_1;
    state->noise_buf = 0;
    state->noise_capacity = capacity;
}

void smc2_cuda_set_fixed_lag(SMC2StateCUDA* state, int L) {
    state->fixed_lag_L = L;
    state->t_checkpoint = -1;
}

void smc2_cuda_set_proposal_std(SMC2StateCUDA* state, const float* std) {
    if (std) {
        memcpy(state->proposal_std, std, N_PARAMS * sizeof(float));
    } else {
        state->proposal_std[0] = 0.01f;
        state->proposal_std[1] = 0.02f;
        state->proposal_std[2] = 0.1f;
        state->proposal_std[3] = 0.02f;
    }
    CUDA_CHECK(cudaMemcpyToSymbol(d_proposal_std, state->proposal_std, N_PARAMS * sizeof(float)));
}

void smc2_cuda_set_cpmmh_rho(SMC2StateCUDA* state, float rho) {
    state->cpmmh_rho = rho;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * Adaptive Proposal Covariance (4×4)
 *═══════════════════════════════════════════════════════════════════════════════*/

#define ADAPTIVE_SCALE_FACTOR (2.38f * 2.38f / (float)N_PARAMS)

void smc2_update_adaptive_covariance(SMC2StateCUDA* state) {
    if (!state->use_adaptive_proposals) return;
    
    float h_cov[N_PARAMS * N_PARAMS];
    float h_chol[N_PARAMS * N_PARAMS] = {0};
    
    int block_size = 1;
    while (block_size < state->N_theta && block_size < 1024) block_size *= 2;
    
    kernel_compute_particle_moments<<<1, block_size, block_size * sizeof(float)>>>(
        state->d_particles, state->d_temp_mean, state->d_temp_cov, state->N_theta
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_cov, state->d_temp_cov, N_PARAMS * N_PARAMS * sizeof(float), cudaMemcpyDeviceToHost));
    
    /* Regularize */
    for (int i = 0; i < N_PARAMS; i++) h_cov[i * N_PARAMS + i] += 1e-6f;
    
    /* Cholesky */
    bool chol_success = true;
    for (int i = 0; i < N_PARAMS; i++) {
        for (int j = 0; j <= i; j++) {
            float sum = 0.0f;
            for (int k = 0; k < j; k++) sum += h_chol[i * N_PARAMS + k] * h_chol[j * N_PARAMS + k];
            if (i == j) {
                float val = h_cov[i * N_PARAMS + i] - sum;
                if (val <= 0.0f) { chol_success = false; val = 1e-8f; }
                h_chol[i * N_PARAMS + j] = sqrtf(val);
            } else {
                float diag = h_chol[j * N_PARAMS + j];
                h_chol[i * N_PARAMS + j] = (diag > 1e-10f) ? (h_cov[i * N_PARAMS + j] - sum) / diag : 0.0f;
            }
        }
    }
    
    if (!chol_success) {
        memset(h_chol, 0, N_PARAMS * N_PARAMS * sizeof(float));
        for (int i = 0; i < N_PARAMS; i++)
            h_chol[i * N_PARAMS + i] = sqrtf(fmaxf(h_cov[i * N_PARAMS + i], 1e-8f));
    }
    
    float scale = sqrtf(ADAPTIVE_SCALE_FACTOR);
    for (int i = 0; i < N_PARAMS * N_PARAMS; i++) h_chol[i] *= scale;
    
    CUDA_CHECK(cudaMemcpyToSymbol(d_proposal_chol, h_chol, N_PARAMS * N_PARAMS * sizeof(float)));
}

void smc2_cuda_init_from_prior(SMC2StateCUDA* state) {
    CUDA_CHECK(cudaMemcpyToSymbol(d_prior, &state->prior, sizeof(SVPrior)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_bounds, &state->bounds, sizeof(SVBounds)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_theta_curve, &state->theta_curve, sizeof(SVCurve)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_fixed_curves, &state->fixed_curves, sizeof(SVFixedCurves)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_proposal_std, state->proposal_std, N_PARAMS * sizeof(float)));
    
    /* Initial Cholesky = diagonal from proposal_std */
    float h_chol[N_PARAMS * N_PARAMS] = {0};
    float scale = sqrtf(ADAPTIVE_SCALE_FACTOR);
    for (int i = 0; i < N_PARAMS; i++)
        h_chol[i * N_PARAMS + i] = state->proposal_std[i] * scale;
    CUDA_CHECK(cudaMemcpyToSymbol(d_proposal_chol, h_chol, N_PARAMS * N_PARAMS * sizeof(float)));
    
    int N_total = state->N_theta * state->N_inner;
    unsigned long long rng_seed = (state->user_seed != 0) ? state->user_seed : 12345ULL;
    kernel_init_rng<<<(N_total + 255) / 256, 256>>>(state->d_particles.rng_states, rng_seed, N_total);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    kernel_init_from_prior<<<state->N_theta, state->N_inner>>>(
        state->d_particles, state->N_theta, state->N_inner,
        state->d_z_noise[state->noise_buf], state->d_u0_noise[state->noise_buf],
        state->noise_capacity
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    state->n_resamples = 0;
    state->n_rejuv_accepts = 0;
    state->n_rejuv_total = 0;
    state->y_history_len = 0;
    state->t_current = -1;
}

float smc2_cuda_update(SMC2StateCUDA* state, float y_obs) {
    /* Store observation */
    if (state->y_history_len >= state->y_history_capacity) {
        int new_cap = state->y_history_capacity * 2;
        float* new_hist;
        CUDA_CHECK(cudaMalloc(&new_hist, new_cap * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(new_hist, state->d_y_history,
                              state->y_history_len * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaFree(state->d_y_history);
        state->d_y_history = new_hist;
        state->y_history_capacity = new_cap;
    }
    CUDA_CHECK(cudaMemcpy(&state->d_y_history[state->y_history_len], &y_obs,
                          sizeof(float), cudaMemcpyHostToDevice));
    state->y_history_len++;
    state->t_current++;
    
    if (state->t_current >= state->noise_capacity)
        smc2_cuda_set_noise_capacity(state, state->noise_capacity * 2);
    
    #define DISPATCH_RBPF_STEP(N) \
        kernel_rbpf_step_impl<N><<<state->N_theta, N, rbpf_shared_mem_size<N>()>>>( \
            state->d_particles, y_obs, state->N_theta, \
            state->d_z_noise[state->noise_buf], state->d_u0_noise[state->noise_buf], \
            state->t_current, state->noise_capacity)
    
    switch (state->N_inner) {
        case 64:  DISPATCH_RBPF_STEP(64);  break;
        case 128: DISPATCH_RBPF_STEP(128); break;
        case 256: DISPATCH_RBPF_STEP(256); break;
        case 512: DISPATCH_RBPF_STEP(512); break;
        default: fprintf(stderr, "Unsupported N_inner=%d\n", state->N_inner); exit(EXIT_FAILURE);
    }
    #undef DISPATCH_RBPF_STEP
    CUDA_CHECK(cudaDeviceSynchronize());
    
    kernel_compute_outer_ess<<<1, state->N_theta, 32 * sizeof(float)>>>(
        state->d_particles, state->d_ess, state->N_theta);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float h_ess;
    CUDA_CHECK(cudaMemcpy(&h_ess, state->d_ess, sizeof(float), cudaMemcpyDeviceToHost));
    
    if (h_ess < state->ess_threshold_outer * state->N_theta) {
        state->n_resamples++;
        
        float h_uniform = xorshift64star_uniform(&state->host_rng_state);
        CUDA_CHECK(cudaMemcpy(state->d_uniform, &h_uniform, sizeof(float), cudaMemcpyHostToDevice));
        
        kernel_outer_resample<<<1, state->N_theta, state->N_theta * sizeof(float)>>>(
            state->d_particles, state->d_ancestors, state->d_uniform, state->N_theta);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        unsigned long long resample_seed = time(NULL) * 1000ULL + state->n_resamples * 12345ULL;
        kernel_copy_theta_particles<<<state->N_theta, state->N_inner>>>(
            state->d_particles, state->d_particles_temp, state->d_ancestors,
            state->N_theta, state->N_inner, resample_seed);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        int other_buf = 1 - state->noise_buf;
        kernel_copy_noise_arrays<<<state->N_theta, state->N_inner>>>(
            state->d_z_noise[state->noise_buf], state->d_z_noise[other_buf],
            state->d_u0_noise[state->noise_buf], state->d_u0_noise[other_buf],
            state->d_ancestors, state->N_theta, state->N_inner,
            state->t_current, state->noise_capacity);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        state->noise_buf = other_buf;
        
        ThetaParticlesSoA tmp = state->d_particles;
        state->d_particles = state->d_particles_temp;
        state->d_particles_temp = tmp;
        
        /* Copy checkpoint if fixed-lag */
        if (state->fixed_lag_L > 0 && state->t_checkpoint >= 0) {
            kernel_copy_checkpoint<<<state->N_theta, state->N_inner>>>(
                state->d_checkpoint_z, state->d_checkpoint_mu_h,
                state->d_checkpoint_var_h, state->d_checkpoint_log_w, state->d_checkpoint_ll,
                state->d_particles_temp.inner_z, state->d_particles_temp.inner_mu_h,
                state->d_particles_temp.inner_var_h, state->d_particles_temp.inner_log_w,
                state->d_particles_temp.log_likelihood,
                state->d_ancestors, state->N_theta, state->N_inner);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            int N_total = state->N_theta * state->N_inner;
            CUDA_CHECK(cudaMemcpy(state->d_checkpoint_z, state->d_particles_temp.inner_z, N_total * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(state->d_checkpoint_mu_h, state->d_particles_temp.inner_mu_h, N_total * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(state->d_checkpoint_var_h, state->d_particles_temp.inner_var_h, N_total * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(state->d_checkpoint_log_w, state->d_particles_temp.inner_log_w, N_total * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(state->d_checkpoint_ll, state->d_particles_temp.log_likelihood, state->N_theta * sizeof(float), cudaMemcpyDeviceToDevice));
        }
        
        /* Determine fixed-lag checkpoint */
        int t_checkpoint_use = -1;
        const float *cp_z = nullptr, *cp_mu = nullptr, *cp_var = nullptr, *cp_logw = nullptr, *cp_ll = nullptr;
        
        if (state->fixed_lag_L > 0 && state->t_checkpoint >= 0) {
            int steps = state->t_current - state->t_checkpoint;
            if (steps > 0 && steps <= 2 * state->fixed_lag_L) {
                t_checkpoint_use = state->t_checkpoint;
                cp_z = state->d_checkpoint_z; cp_mu = state->d_checkpoint_mu_h;
                cp_var = state->d_checkpoint_var_h; cp_logw = state->d_checkpoint_log_w;
                cp_ll = state->d_checkpoint_ll;
            }
        }
        
        smc2_update_adaptive_covariance(state);
        
        #define DISPATCH_CPMMH(N) \
            kernel_cpmmh_rejuvenate_fused_impl<N><<<state->N_theta, N, cpmmh_shared_mem_size<N>()>>>( \
                state->d_particles, state->d_particles_temp, state->d_y_history, \
                curr_noise, other_noise, curr_u0, other_u0, \
                state->t_current, state->N_theta, state->noise_capacity, state->cpmmh_rho, \
                state->d_accepts, state->d_swap_flags, \
                state->user_seed, state->n_rejuv_total / state->N_theta, k % 3, \
                t_checkpoint_use, cp_z, cp_mu, cp_var, cp_logw, cp_ll)
        
        for (int k = 0; k < state->K_rejuv; k++) {
            int h_accepts = 0;
            CUDA_CHECK(cudaMemcpy(state->d_accepts, &h_accepts, sizeof(int), cudaMemcpyHostToDevice));
            
            noise_t* curr_noise = state->d_z_noise[state->noise_buf];
            noise_t* other_noise = state->d_z_noise[1 - state->noise_buf];
            noise_t* curr_u0 = state->d_u0_noise[state->noise_buf];
            noise_t* other_u0 = state->d_u0_noise[1 - state->noise_buf];
            
            switch (state->N_inner) {
                case 64:  DISPATCH_CPMMH(64);  break;
                case 128: DISPATCH_CPMMH(128); break;
                case 256: DISPATCH_CPMMH(256); break;
                case 512: DISPATCH_CPMMH(512); break;
                default: fprintf(stderr, "Unsupported N_inner=%d\n", state->N_inner); exit(EXIT_FAILURE);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            
            int t_start_commit = (t_checkpoint_use >= 0) ? (t_checkpoint_use + 1) : 0;
            kernel_commit_accepted_noise<<<state->N_theta, state->N_inner>>>(
                curr_noise, other_noise, curr_u0, other_u0,
                state->d_swap_flags, state->N_theta, state->N_inner,
                state->t_current, state->noise_capacity, t_start_commit);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            CUDA_CHECK(cudaMemcpy(&h_accepts, state->d_accepts, sizeof(int), cudaMemcpyDeviceToHost));
            state->n_rejuv_accepts += h_accepts;
            state->n_rejuv_total += state->N_theta;
        }
        #undef DISPATCH_CPMMH
        
        kernel_reset_outer_weights<<<(state->N_theta + 255) / 256, 256>>>(
            state->d_particles, state->N_theta);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        kernel_compute_outer_ess<<<1, state->N_theta, 32 * sizeof(float)>>>(
            state->d_particles, state->d_ess, state->N_theta);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_ess, state->d_ess, sizeof(float), cudaMemcpyDeviceToHost));
    }
    
    /* Save checkpoint */
    if (state->fixed_lag_L > 0) {
        int target = (state->t_current / state->fixed_lag_L) * state->fixed_lag_L;
        if (target > state->t_checkpoint && state->t_current > 0) {
            kernel_save_checkpoint<<<state->N_theta, state->N_inner>>>(
                state->d_particles,
                state->d_checkpoint_z, state->d_checkpoint_mu_h,
                state->d_checkpoint_var_h, state->d_checkpoint_log_w, state->d_checkpoint_ll,
                state->N_theta, state->N_inner);
            CUDA_CHECK(cudaDeviceSynchronize());
            state->t_checkpoint = target;
        }
    }
    
    return h_ess;
}

void smc2_cuda_get_theta_mean(SMC2StateCUDA* state, float* theta_mean) {
    float* h_weight = (float*)malloc(state->N_theta * sizeof(float));
    float* h_params[N_PARAMS];
    for (int i = 0; i < N_PARAMS; i++) h_params[i] = (float*)malloc(state->N_theta * sizeof(float));
    
    CUDA_CHECK(cudaMemcpy(h_weight, state->d_particles.weight, state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_params[0], state->d_particles.rho, state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_params[1], state->d_particles.sigma_z, state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_params[2], state->d_particles.mu_base, state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_params[3], state->d_particles.sigma_base, state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < N_PARAMS; i++) theta_mean[i] = 0.0f;
    for (int j = 0; j < state->N_theta; j++) {
        float w = h_weight[j];
        for (int i = 0; i < N_PARAMS; i++) theta_mean[i] += w * h_params[i][j];
    }
    
    free(h_weight);
    for (int i = 0; i < N_PARAMS; i++) free(h_params[i]);
}

void smc2_cuda_get_theta_std(SMC2StateCUDA* state, float* theta_std) {
    float theta_mean[N_PARAMS];
    smc2_cuda_get_theta_mean(state, theta_mean);
    
    float* h_weight = (float*)malloc(state->N_theta * sizeof(float));
    float* h_params[N_PARAMS];
    for (int i = 0; i < N_PARAMS; i++) h_params[i] = (float*)malloc(state->N_theta * sizeof(float));
    
    CUDA_CHECK(cudaMemcpy(h_weight, state->d_particles.weight, state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_params[0], state->d_particles.rho, state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_params[1], state->d_particles.sigma_z, state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_params[2], state->d_particles.mu_base, state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_params[3], state->d_particles.sigma_base, state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < N_PARAMS; i++) theta_std[i] = 0.0f;
    for (int j = 0; j < state->N_theta; j++) {
        float w = h_weight[j];
        for (int i = 0; i < N_PARAMS; i++) {
            float d = h_params[i][j] - theta_mean[i];
            theta_std[i] += w * d * d;
        }
    }
    for (int i = 0; i < N_PARAMS; i++) theta_std[i] = sqrtf(theta_std[i]);
    
    free(h_weight);
    for (int i = 0; i < N_PARAMS; i++) free(h_params[i]);
}

float smc2_cuda_get_outer_ess(SMC2StateCUDA* state) {
    float h_ess;
    CUDA_CHECK(cudaMemcpy(&h_ess, state->d_ess, sizeof(float), cudaMemcpyDeviceToHost));
    return h_ess;
}
