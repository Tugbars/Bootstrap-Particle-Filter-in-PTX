/**
 * @file smc2_engine.cu
 * @brief SMC² with RBPF Inner Filter — Complete Implementation
 *
 * Internal organization (sections flow in dependency order):
 *
 *   §1   OCSN Model Constants + Kalman Update
 *   §2   Constant Memory Declarations
 *   §3   Device Helpers (log_prior, noise indexing)
 *   §4   Kernels — Initialization
 *   §5   Kernels — RBPF Forward Step
 *   §6   Kernels — Outer Particle Management
 *   §7   Kernels — CPMMH Rejuvenation
 *   §8   Kernels — Checkpoint
 *   §9   Host Internal — Template Dispatch (static)
 *   §10  Host Internal — Utilities (RNG, adaptive covariance)
 *   §11  Host — Memory Management (alloc, free, resize)
 *   §12  Host — Configuration
 *   §13  Host — Resample + Rejuvenate (THE shared path)
 *   §14  Host — Update Loop (init, update, update_batch)
 *   §15  Host — Queries & Diagnostics
 *
 * Key structural improvements over the original:
 *   - Resample+rejuvenate extracted into smc2_resample_rejuvenate() (§13)
 *     — eliminates 150-line duplication between update() and update_batch()
 *   - Kernel launches wrapped in static dispatch functions (§9)
 *     — eliminates DISPATCH_* macros scattered across multiple call sites
 *   - Model defaults factored to named functions in the header (§3 of .cuh)
 *     — alloc() does allocation, configuration is explicit
 *   - Diagnostics exposed via SMC2Diagnostics struct
 *     — param_tracker no longer reaches into SMC² internals
 */

#include "smc2_engine.cuh"
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
 * §1: OCSN MODEL — Constants + Kalman Update
 *
 * Omori et al. (2007) centered parameterization:
 *   y_t = log(r_t^2), observation equation uses these mixture approx values.
 *   If your data uses y_t = log(r_t^2) - 1.2704 (Kim et al.), swap these
 *   constants with the Kim parameterization.
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

/**
 * @brief OCSN Kalman Update — marginalized moment-matching
 *
 * Computes posterior E[h|y] and Var[h|y] by moment-matching over the
 * 10-component mixture. Deterministic — no sampling.
 */
__device__
void ocsn_kalman_update(
    float y, float mu_pred, float var_pred,
    float* mu_post, float* var_post, float* log_lik
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
    for (int k = 0; k < OCSN_K; k++)
        sum_exp += __expf(log_alpha_tilde[k] - log_max);
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
 * §2: CONSTANT MEMORY DECLARATIONS
 *═══════════════════════════════════════════════════════════════════════════════*/

__constant__ SVPrior       d_prior;
__constant__ SVBounds      d_bounds;
__constant__ SVCurve       d_theta_curve;
__constant__ float         d_proposal_std[N_PARAMS];
__constant__ float         d_proposal_chol[N_PARAMS * N_PARAMS];
__constant__ uint8_t       d_fixed_mask[N_PARAMS];
__constant__ float         d_fixed_values[N_PARAMS];


/*═══════════════════════════════════════════════════════════════════════════════
 * §3: DEVICE HELPERS
 *═══════════════════════════════════════════════════════════════════════════════*/

/* ── Log prior (8D) ── */

__device__ float log_prior_theta(
    float rho, float sigma_total, float r_split,
    float mu_base, float mu_scale, float mu_rate,
    float sigma_scale, float sigma_rate
) {
    if (rho < d_bounds.rho_min || rho > d_bounds.rho_max) return -INFINITY;
    if (sigma_total < d_bounds.sigma_total_min || sigma_total > d_bounds.sigma_total_max) return -INFINITY;
    if (r_split < d_bounds.r_split_min || r_split > d_bounds.r_split_max) return -INFINITY;
    if (mu_base < d_bounds.mu_base_min || mu_base > d_bounds.mu_base_max) return -INFINITY;
    if (mu_scale < d_bounds.mu_scale_min || mu_scale > d_bounds.mu_scale_max) return -INFINITY;
    if (mu_rate < d_bounds.mu_rate_min || mu_rate > d_bounds.mu_rate_max) return -INFINITY;
    if (sigma_scale < d_bounds.sigma_scale_min || sigma_scale > d_bounds.sigma_scale_max) return -INFINITY;
    if (sigma_rate < d_bounds.sigma_rate_min || sigma_rate > d_bounds.sigma_rate_max) return -INFINITY;

    float d_rho = d_fixed_mask[0] ? 0.0f : (rho - d_prior.rho_mean) / d_prior.rho_std;
    float d_st  = d_fixed_mask[1] ? 0.0f : (sigma_total - d_prior.sigma_total_mean) / d_prior.sigma_total_std;
    float d_rs  = d_fixed_mask[2] ? 0.0f : (r_split - d_prior.r_split_mean) / d_prior.r_split_std;
    float d_mb  = d_fixed_mask[3] ? 0.0f : (mu_base - d_prior.mu_base_mean) / d_prior.mu_base_std;
    float d_ms  = d_fixed_mask[4] ? 0.0f : (mu_scale - d_prior.mu_scale_mean) / d_prior.mu_scale_std;
    float d_mr  = d_fixed_mask[5] ? 0.0f : (mu_rate - d_prior.mu_rate_mean) / d_prior.mu_rate_std;
    float d_ss  = d_fixed_mask[6] ? 0.0f : (sigma_scale - d_prior.sigma_scale_mean) / d_prior.sigma_scale_std;
    float d_sr  = d_fixed_mask[7] ? 0.0f : (sigma_rate - d_prior.sigma_rate_mean) / d_prior.sigma_rate_std;

    return -0.5f * (d_rho*d_rho + d_st*d_st + d_rs*d_rs + d_mb*d_mb
                  + d_ms*d_ms + d_mr*d_mr + d_ss*d_ss + d_sr*d_sr);
}

/* ── Noise indexing (circular buffer) ── */

__device__ __forceinline__
int64_t z_noise_slot(int theta_idx, int t, int inner_idx, int N_inner, int cap) {
    int t_slot = t % cap;
    return (int64_t)theta_idx * N_inner * cap + (int64_t)t_slot * N_inner + inner_idx;
}

__device__ __forceinline__
int64_t u0_noise_slot(int theta_idx, int t, int cap) {
    int t_slot = t % cap;
    return (int64_t)theta_idx * cap + t_slot;
}


/*═══════════════════════════════════════════════════════════════════════════════
 * §4: KERNELS — Initialization
 *═══════════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_init_rng(curandState* states, unsigned long long seed, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) curand_init(seed, idx, 0, &states[idx]);
}

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

    __shared__ float s_rho, s_sigma_total, s_r_split;
    __shared__ float s_mu_base, s_mu_scale, s_mu_rate;
    __shared__ float s_sigma_scale, s_sigma_rate;
    __shared__ float s_sigma_z, s_sigma_base;

    if (inner_idx == 0) {
        int attempts = 0, valid = 0;
        while (!valid && attempts < 1000) {
            s_rho = d_prior.rho_mean + d_prior.rho_std * curand_normal(rng);
            s_sigma_total = d_prior.sigma_total_mean + d_prior.sigma_total_std * curand_normal(rng);
            s_r_split = d_prior.r_split_mean + d_prior.r_split_std * curand_normal(rng);
            s_mu_base = d_prior.mu_base_mean + d_prior.mu_base_std * curand_normal(rng);
            s_mu_scale = d_prior.mu_scale_mean + d_prior.mu_scale_std * curand_normal(rng);
            s_mu_rate = d_prior.mu_rate_mean + d_prior.mu_rate_std * curand_normal(rng);
            s_sigma_scale = d_prior.sigma_scale_mean + d_prior.sigma_scale_std * curand_normal(rng);
            s_sigma_rate = d_prior.sigma_rate_mean + d_prior.sigma_rate_std * curand_normal(rng);

            valid = (s_rho >= d_bounds.rho_min && s_rho <= d_bounds.rho_max &&
                     s_sigma_total >= d_bounds.sigma_total_min && s_sigma_total <= d_bounds.sigma_total_max &&
                     s_r_split >= d_bounds.r_split_min && s_r_split <= d_bounds.r_split_max &&
                     s_mu_base >= d_bounds.mu_base_min && s_mu_base <= d_bounds.mu_base_max &&
                     s_mu_scale >= d_bounds.mu_scale_min && s_mu_scale <= d_bounds.mu_scale_max &&
                     s_mu_rate >= d_bounds.mu_rate_min && s_mu_rate <= d_bounds.mu_rate_max &&
                     s_sigma_scale >= d_bounds.sigma_scale_min && s_sigma_scale <= d_bounds.sigma_scale_max &&
                     s_sigma_rate >= d_bounds.sigma_rate_min && s_sigma_rate <= d_bounds.sigma_rate_max);
            attempts++;
        }

        /* Override fixed params */
        if (d_fixed_mask[0]) s_rho         = d_fixed_values[0];
        if (d_fixed_mask[1]) s_sigma_total = d_fixed_values[1];
        if (d_fixed_mask[2]) s_r_split     = d_fixed_values[2];
        if (d_fixed_mask[3]) s_mu_base     = d_fixed_values[3];
        if (d_fixed_mask[4]) s_mu_scale    = d_fixed_values[4];
        if (d_fixed_mask[5]) s_mu_rate     = d_fixed_values[5];
        if (d_fixed_mask[6]) s_sigma_scale = d_fixed_values[6];
        if (d_fixed_mask[7]) s_sigma_rate  = d_fixed_values[7];

        /* Derive physical params */
        s_sigma_z = s_r_split * s_sigma_total;
        s_sigma_base = sqrtf(fmaxf(1.0f - s_r_split * s_r_split, 1e-6f)) * s_sigma_total;

        particles.rho[theta_idx] = s_rho;
        particles.sigma_total[theta_idx] = s_sigma_total;
        particles.r_split[theta_idx] = s_r_split;
        particles.mu_base[theta_idx] = s_mu_base;
        particles.mu_scale[theta_idx] = s_mu_scale;
        particles.mu_rate[theta_idx] = s_mu_rate;
        particles.sigma_scale[theta_idx] = s_sigma_scale;
        particles.sigma_rate[theta_idx] = s_sigma_rate;

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
    int64_t z_noise_idx = z_noise_slot(theta_idx, 0, inner_idx, N_inner, noise_capacity);
    float z_noise_init = noise_store_roundtrip(d_z_noise, z_noise_idx, z_noise_raw);

    if (inner_idx == 0) {
        float u0_noise_raw = curand_normal(rng);
        int64_t u0_idx = u0_noise_slot(theta_idx, 0, noise_capacity);
        noise_store(d_u0_noise, u0_idx, u0_noise_raw);
    }

    float z_tilde = z_tilde_stat_std * z_noise_init;
    float z = z_tilde_to_z(z_tilde);

    float theta_z = eval_curve(d_theta_curve.base, d_theta_curve.scale, d_theta_curve.rate, z);
    float mu_z = eval_curve(s_mu_base, s_mu_scale, s_mu_rate, z);
    float sigma_h = eval_curve(s_sigma_base, s_sigma_scale, s_sigma_rate, z);

    float phi = 1.0f - theta_z;
    float one_minus_phi_sq = fmaxf(1.0f - phi * phi, 1e-6f);
    float h_stat_var = (sigma_h * sigma_h) / one_minus_phi_sq;

    particles.inner_z[global_idx] = z_tilde;
    particles.inner_mu_h[global_idx] = mu_z;
    particles.inner_var_h[global_idx] = h_stat_var;
    particles.inner_log_w[global_idx] = -__logf((float)N_inner);
}


/*═══════════════════════════════════════════════════════════════════════════════
 * §5: KERNELS — RBPF Forward Step
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
    __shared__ float s_mu_scale, s_mu_rate, s_sigma_scale, s_sigma_rate;
    __shared__ float s_log_max, s_sum_w, s_u0;

    if (inner_idx == 0) {
        s_rho = particles.rho[theta_idx];
        float sigma_total = particles.sigma_total[theta_idx];
        float r = particles.r_split[theta_idx];
        s_mu_base = particles.mu_base[theta_idx];
        s_mu_scale = particles.mu_scale[theta_idx];
        s_mu_rate = particles.mu_rate[theta_idx];
        s_sigma_scale = particles.sigma_scale[theta_idx];
        s_sigma_rate = particles.sigma_rate[theta_idx];
        s_sigma_z = r * sigma_total;
        s_sigma_base = sqrtf(fmaxf(1.0f - r * r, 1e-6f)) * sigma_total;
    }
    __syncthreads();

    curandState local_rng = particles.rng_states[global_idx];

    float z_tilde = particles.inner_z[global_idx];
    float mu_h = particles.inner_mu_h[global_idx];
    float var_h = particles.inner_var_h[global_idx];
    float log_w = particles.inner_log_w[global_idx];

    int64_t z_noise_idx = z_noise_slot(theta_idx, t_current + 1, inner_idx, N_INNER, noise_capacity);
    int64_t u0_noise_idx = u0_noise_slot(theta_idx, t_current + 1, noise_capacity);

    float z_noise_raw = curand_normal(&local_rng);
    float z_noise = noise_store_roundtrip(d_z_noise, z_noise_idx, z_noise_raw);

    if (inner_idx == 0) {
        float u0_noise_raw = curand_normal(&local_rng);
        float u0_stored = noise_store_roundtrip(d_u0_noise, u0_noise_idx, u0_noise_raw);
        s_u0 = u0_from_noise(u0_stored);
    }
    __syncthreads();

    /* ── RESAMPLE ── */
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

    /* ── PROPAGATE z̃ ── */
    float z_tilde_new = s_rho * z_tilde + s_sigma_z * z_noise;
    float z = z_tilde_to_z(z_tilde_new);

    /* ── KALMAN PREDICT ── */
    float theta_z = eval_curve(d_theta_curve.base, d_theta_curve.scale, d_theta_curve.rate, z);
    float mu_z = eval_curve(s_mu_base, s_mu_scale, s_mu_rate, z);
    float sigma_h = eval_curve(s_sigma_base, s_sigma_scale, s_sigma_rate, z);

    float phi = 1.0f - theta_z;
    float mu_pred = phi * mu_h + theta_z * mu_z;
    float var_pred = phi * phi * var_h + sigma_h * sigma_h;
    var_pred = fmaxf(var_pred, 1e-8f);

    /* ── OCSN UPDATE ── */
    float mu_post, var_post, log_lik;
    ocsn_kalman_update(y_obs, mu_pred, var_pred, &mu_post, &var_post, &log_lik);
    log_w += log_lik;

    /* ── NORMALIZE + ESS ── */
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


/*═══════════════════════════════════════════════════════════════════════════════
 * §6: KERNELS — Outer Particle Management
 *═══════════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_reset_outer_weights(ThetaParticlesSoA particles, int N_theta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_theta) return;
    particles.log_weight[idx] = 0.0f;
    particles.weight[idx] = 1.0f / (float)N_theta;
    particles.log_likelihood[idx] = 0.0f;
}

__global__ void kernel_compute_particle_moments(
    ThetaParticlesSoA particles,
    float* d_mean, float* d_cov,
    int N_theta
) {
    extern __shared__ float s_data[];
    float* s_scratch = s_data;
    int tid = threadIdx.x;

    const float* param_ptrs[N_PARAMS];
    param_ptrs[0] = particles.rho;          param_ptrs[1] = particles.sigma_total;
    param_ptrs[2] = particles.r_split;      param_ptrs[3] = particles.mu_base;
    param_ptrs[4] = particles.mu_scale;     param_ptrs[5] = particles.mu_rate;
    param_ptrs[6] = particles.sigma_scale;  param_ptrs[7] = particles.sigma_rate;

    /* Phase 1: Means via thread-strided reduction */
    __shared__ float s_mean[N_PARAMS];
    for (int pi = 0; pi < N_PARAMS; pi++) {
        float local_sum = 0.0f;
        for (int i = tid; i < N_theta; i += blockDim.x)
            local_sum += param_ptrs[pi][i];
        float total = block_reduce_sum(local_sum, s_scratch);
        if (tid == 0) s_mean[pi] = total / (float)N_theta;
        __syncthreads();
    }
    if (tid < N_PARAMS) d_mean[tid] = s_mean[tid];

    /* Phase 2: Covariance via thread-strided reduction */
    float inv_N_1 = 1.0f / (float)(N_theta - 1);
    for (int i = 0; i < N_PARAMS; i++) {
        for (int j = 0; j <= i; j++) {
            float local_sum = 0.0f;
            for (int k = tid; k < N_theta; k += blockDim.x) {
                float ci = param_ptrs[i][k] - s_mean[i];
                float cj = param_ptrs[j][k] - s_mean[j];
                local_sum += ci * cj;
            }
            float total = block_reduce_sum(local_sum, s_scratch);
            if (tid == 0) {
                float cov_ij = total * inv_N_1;
                d_cov[i * N_PARAMS + j] = cov_ij;
                d_cov[j * N_PARAMS + i] = cov_ij;
            }
            __syncthreads();
        }
    }
}

__global__ void kernel_compute_outer_ess(
    ThetaParticlesSoA particles, float* d_ess_out, int N_theta
) {
    extern __shared__ float s_data[];
    int tid = threadIdx.x;

    float local_max = -1e30f;
    for (int i = tid; i < N_theta; i += blockDim.x)
        local_max = fmaxf(local_max, particles.log_weight[i]);
    float log_max = block_reduce_max(local_max, s_data);
    __shared__ float s_log_max;
    if (tid == 0) s_log_max = log_max;
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < N_theta; i += blockDim.x)
        local_sum += __expf(particles.log_weight[i] - s_log_max);
    float sum_w = block_reduce_sum(local_sum, s_data);
    __shared__ float s_sum_w;
    if (tid == 0) s_sum_w = sum_w;
    __syncthreads();

    float local_sq = 0.0f;
    for (int i = tid; i < N_theta; i += blockDim.x) {
        float w = __expf(particles.log_weight[i] - s_log_max) / s_sum_w;
        particles.weight[i] = w;
        local_sq += w * w;
    }
    float sum_w_sq = block_reduce_sum(local_sq, s_data);
    if (tid == 0) *d_ess_out = 1.0f / fmaxf(sum_w_sq, 1e-30f);
}

__global__ void kernel_outer_resample(
    ThetaParticlesSoA particles, int* d_ancestors, float* d_uniform, int N_theta
) {
    extern __shared__ float s_cumsum[];
    int tid = threadIdx.x;

    for (int i = tid; i < N_theta; i += blockDim.x)
        s_cumsum[i] = particles.weight[i];
    __syncthreads();

    if (tid == 0) {
        for (int i = 1; i < N_theta; i++) s_cumsum[i] += s_cumsum[i-1];
        s_cumsum[N_theta - 1] = 1.0f;
    }
    __syncthreads();

    for (int idx = tid; idx < N_theta; idx += blockDim.x) {
        float u = (*d_uniform + (float)idx) / (float)N_theta;
        int lo = 0, hi = N_theta - 1;
        while (lo < hi) { int mid = (lo+hi)/2; if (s_cumsum[mid] < u) lo = mid+1; else hi = mid; }
        d_ancestors[idx] = lo;
    }
}

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
        dst.sigma_total[theta_idx] = src.sigma_total[ancestor];
        dst.r_split[theta_idx] = src.r_split[ancestor];
        dst.mu_base[theta_idx] = src.mu_base[ancestor];
        dst.mu_scale[theta_idx] = src.mu_scale[ancestor];
        dst.mu_rate[theta_idx] = src.mu_rate[ancestor];
        dst.sigma_scale[theta_idx] = src.sigma_scale[ancestor];
        dst.sigma_rate[theta_idx] = src.sigma_rate[ancestor];

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

        curand_init(resample_seed, (unsigned long long)dst_idx, 0, &dst.rng_states[dst_idx]);
    }
}

__global__ void kernel_copy_noise_arrays(
    const noise_t* src_z_noise, noise_t* dst_z_noise,
    const noise_t* src_u0_noise, noise_t* dst_u0_noise,
    const int* d_ancestors,
    int N_theta, int N_inner, int t_current, int noise_capacity, int t_start
) {
    int theta_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    if (theta_idx >= N_theta || inner_idx >= N_inner) return;

    int ancestor = d_ancestors[theta_idx];

    for (int t = t_start; t <= t_current + 1; t++) {
        int64_t src_idx = z_noise_slot(ancestor, t, inner_idx, N_inner, noise_capacity);
        int64_t dst_idx = z_noise_slot(theta_idx, t, inner_idx, N_inner, noise_capacity);
        dst_z_noise[dst_idx] = src_z_noise[src_idx];
    }

    if (inner_idx == 0) {
        for (int t = t_start; t <= t_current + 1; t++) {
            int64_t src_idx = u0_noise_slot(ancestor, t, noise_capacity);
            int64_t dst_idx = u0_noise_slot(theta_idx, t, noise_capacity);
            dst_u0_noise[dst_idx] = src_u0_noise[src_idx];
        }
    }
}


/*═══════════════════════════════════════════════════════════════════════════════
 * §7: KERNELS — CPMMH Rejuvenation
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
    int t_checkpoint,
    const float* d_checkpoint_z, const float* d_checkpoint_mu_h,
    const float* d_checkpoint_var_h, const float* d_checkpoint_log_w,
    const float* d_checkpoint_ll
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
    float* s_cdf = &s_var[N_INNER];
    int* s_idx = reinterpret_cast<int*>(&s_cdf[N_INNER]);
    void* s_cub_temp = reinterpret_cast<void*>(&s_idx[N_INNER]);

    __shared__ float s_log_max, s_sum_w, s_ess_prop;
    __shared__ float s_rho_curr, s_sigma_total_curr, s_r_split_curr;
    __shared__ float s_mu_base_curr, s_mu_scale_curr, s_mu_rate_curr;
    __shared__ float s_sigma_scale_curr, s_sigma_rate_curr;
    __shared__ float s_rho_prop, s_sigma_total_prop, s_r_split_prop;
    __shared__ float s_mu_base_prop, s_mu_scale_prop, s_mu_rate_prop;
    __shared__ float s_sigma_scale_prop, s_sigma_rate_prop;
    __shared__ float s_sigma_z_prop, s_sigma_base_prop;
    __shared__ float s_ll_curr, s_ll_prop, s_lp_curr, s_lp_prop;
    __shared__ int s_accept, s_valid;
    __shared__ float s_u0_shared;

    curandState local_rng = particles.rng_states[global_idx];

    /* ── PROPOSE θ* (thread 0) — 8D random walk ── */
    if (inner_idx == 0) {
        s_rho_curr = particles.rho[theta_idx];
        s_sigma_total_curr = particles.sigma_total[theta_idx];
        s_r_split_curr = particles.r_split[theta_idx];
        s_mu_base_curr = particles.mu_base[theta_idx];
        s_mu_scale_curr = particles.mu_scale[theta_idx];
        s_mu_rate_curr = particles.mu_rate[theta_idx];
        s_sigma_scale_curr = particles.sigma_scale[theta_idx];
        s_sigma_rate_curr = particles.sigma_rate[theta_idx];

        s_ll_curr = particles.log_likelihood[theta_idx];
        s_lp_curr = log_prior_theta(s_rho_curr, s_sigma_total_curr, s_r_split_curr,
                                     s_mu_base_curr, s_mu_scale_curr, s_mu_rate_curr,
                                     s_sigma_scale_curr, s_sigma_rate_curr);

        float z_rnd[N_PARAMS];
        for (int i = 0; i < N_PARAMS; i++) z_rnd[i] = curand_normal(&local_rng);

        float pert[N_PARAMS] = {0};
        float mix_u = curand_uniform(&local_rng);

        if (mix_u > 0.05f) {
            for (int i = 0; i < N_PARAMS; i++) {
                float sum = 0.0f;
                for (int j = 0; j <= i; j++)
                    sum += d_proposal_chol[i * N_PARAMS + j] * z_rnd[j];
                pert[i] = sum;
            }
        } else {
            for (int i = 0; i < N_PARAMS; i++)
                pert[i] = d_proposal_std[i] * z_rnd[i];
        }

        for (int i = 0; i < N_PARAMS; i++)
            if (d_fixed_mask[i]) pert[i] = 0.0f;

        s_rho_prop          = s_rho_curr          + pert[0];
        s_sigma_total_prop  = s_sigma_total_curr  + pert[1];
        s_r_split_prop      = s_r_split_curr      + pert[2];
        s_mu_base_prop      = s_mu_base_curr      + pert[3];
        s_mu_scale_prop     = s_mu_scale_curr     + pert[4];
        s_mu_rate_prop      = s_mu_rate_curr      + pert[5];
        s_sigma_scale_prop  = s_sigma_scale_curr  + pert[6];
        s_sigma_rate_prop   = s_sigma_rate_curr   + pert[7];

        s_sigma_z_prop = s_r_split_prop * s_sigma_total_prop;
        s_sigma_base_prop = sqrtf(fmaxf(1.0f - s_r_split_prop * s_r_split_prop, 1e-6f)) * s_sigma_total_prop;

        s_lp_prop = log_prior_theta(s_rho_prop, s_sigma_total_prop, s_r_split_prop,
                                     s_mu_base_prop, s_mu_scale_prop, s_mu_rate_prop,
                                     s_sigma_scale_prop, s_sigma_rate_prop);
        s_valid = isfinite(s_lp_prop) ? 1 : 0;
        s_accept = 0;
    }
    __syncthreads();

    if (s_valid == 0) {
        if (inner_idx == 0) d_swap_flags[theta_idx] = 0;
        particles.rng_states[global_idx] = local_rng;
        return;
    }

    float scale = sqrtf(1.0f - cpmmh_rho * cpmmh_rho);

    float rho = s_rho_prop;
    float sigma_z = s_sigma_z_prop;
    float mu_base = s_mu_base_prop;
    float sigma_base = s_sigma_base_prop;
    float mu_scale = s_mu_scale_prop;
    float mu_rate = s_mu_rate_prop;
    float sigma_scale = s_sigma_scale_prop;
    float sigma_rate = s_sigma_rate_prop;

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

        float z_noise_curr_0 = noise_load(d_z_noise_curr, z_noise_slot(theta_idx, 0, inner_idx, N_INNER, noise_capacity));
        float z_noise_fresh_0 = curand_normal(&local_rng);
        float z_noise_prop_0 = cpmmh_rho * z_noise_curr_0 + scale * z_noise_fresh_0;
        noise_store(d_z_noise_other, z_noise_slot(theta_idx, 0, inner_idx, N_INNER, noise_capacity), z_noise_prop_0);

        z_tilde = z_tilde_stat_std * z_noise_prop_0;
        float z_init = z_tilde_to_z(z_tilde);

        float theta_z_init = eval_curve(d_theta_curve.base, d_theta_curve.scale, d_theta_curve.rate, z_init);
        float mu_z_init = eval_curve(mu_base, mu_scale, mu_rate, z_init);
        float sigma_h_init = eval_curve(sigma_base, sigma_scale, sigma_rate, z_init);
        float phi_init = 1.0f - theta_z_init;
        float h_stat_var = (sigma_h_init * sigma_h_init) / fmaxf(1.0f - phi_init * phi_init, 1e-6f);

        mu_h = mu_z_init;
        var_h = h_stat_var;
        log_w = -__logf((float)N_INNER);
        t_start = 0;
    }

    /* ── Replay observations ── */
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

        int64_t z_idx_t1 = z_noise_slot(theta_idx, t + 1, inner_idx, N_INNER, noise_capacity);
        float z_noise_curr_t1 = noise_load(d_z_noise_curr, z_idx_t1);
        float z_noise_fresh_t1 = curand_normal(&local_rng);
        float z_noise_prop_t1_raw = cpmmh_rho * z_noise_curr_t1 + scale * z_noise_fresh_t1;
        float z_noise_prop_t1 = noise_store_roundtrip(d_z_noise_other, z_idx_t1, z_noise_prop_t1_raw);

        if (inner_idx == 0) {
            int64_t u0_idx_t1 = u0_noise_slot(theta_idx, t + 1, noise_capacity);
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

        float z_tilde_new = rho * z_tilde + sigma_z * z_noise_prop_t1;
        float z = z_tilde_to_z(z_tilde_new);

        float theta_z = eval_curve(d_theta_curve.base, d_theta_curve.scale, d_theta_curve.rate, z);
        float mu_z_val = eval_curve(mu_base, mu_scale, mu_rate, z);
        float sigma_h = eval_curve(sigma_base, sigma_scale, sigma_rate, z);
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

    __syncthreads();

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

    /* ── MH ACCEPT/REJECT ── */
    if (inner_idx == 0) {
        float ll_curr_effective = s_ll_curr - s_ll_base;
        float ll_prop_effective = ll_accum;

        float log_alpha = (ll_prop_effective + s_lp_prop) - (ll_curr_effective + s_lp_curr);

        float u = curand_uniform(&local_rng);
        s_accept = (__logf(u) < log_alpha) ? 1 : 0;

        if (s_accept) {
            particles.rho[theta_idx] = s_rho_prop;
            particles.sigma_total[theta_idx] = s_sigma_total_prop;
            particles.r_split[theta_idx] = s_r_split_prop;
            particles.mu_base[theta_idx] = s_mu_base_prop;
            particles.mu_scale[theta_idx] = s_mu_scale_prop;
            particles.mu_rate[theta_idx] = s_mu_rate_prop;
            particles.sigma_scale[theta_idx] = s_sigma_scale_prop;
            particles.sigma_rate[theta_idx] = s_sigma_rate_prop;
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

    for (int t = t_start; t <= t_current + 1; t++) {
        int64_t idx = z_noise_slot(theta_idx, t, inner_idx, N_inner, noise_capacity);
        d_z_noise_0[idx] = d_z_noise_1[idx];
    }
    if (inner_idx == 0) {
        for (int t = t_start; t <= t_current + 1; t++) {
            int64_t idx = u0_noise_slot(theta_idx, t, noise_capacity);
            d_u0_noise_0[idx] = d_u0_noise_1[idx];
        }
    }
}


/*═══════════════════════════════════════════════════════════════════════════════
 * §8: KERNELS — Checkpoint
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
 * §9: HOST INTERNAL — Template Dispatch
 *
 * Each kernel template is dispatched exactly once. No more DISPATCH_* macros
 * scattered across update(), update_batch(), and future callers.
 *═══════════════════════════════════════════════════════════════════════════════*/

static void smc2_launch_rbpf_step(SMC2StateCUDA* s, float y_obs, cudaStream_t stream = 0) {
    noise_t* z = s->d_z_noise[s->noise_buf];
    noise_t* u = s->d_u0_noise[s->noise_buf];
    int Nt = s->N_theta, tc = s->t_current, cap = s->noise_capacity;

    #define CASE(N) case N: kernel_rbpf_step_impl<N> \
        <<<Nt, N, rbpf_shared_mem_size<N>(), stream>>> \
        (s->d_particles, y_obs, Nt, z, u, tc, cap); break

    switch (s->N_inner) { CASE(64); CASE(128); CASE(256); CASE(512);
        default: fprintf(stderr, "Unsupported N_inner=%d\n", s->N_inner); exit(1); }
    #undef CASE
}

static void smc2_launch_cpmmh(
    SMC2StateCUDA* s, int t_cp,
    const float* cp_z, const float* cp_mu, const float* cp_var,
    const float* cp_logw, const float* cp_ll
) {
    noise_t* curr_z  = s->d_z_noise[s->noise_buf];
    noise_t* other_z = s->d_z_noise[1 - s->noise_buf];
    noise_t* curr_u  = s->d_u0_noise[s->noise_buf];
    noise_t* other_u = s->d_u0_noise[1 - s->noise_buf];

    #define CASE(N) case N: kernel_cpmmh_rejuvenate_fused_impl<N> \
        <<<s->N_theta, N, cpmmh_shared_mem_size<N>()>>>( \
            s->d_particles, s->d_particles_temp, s->d_y_history, \
            curr_z, other_z, curr_u, other_u, \
            s->t_current, s->N_theta, s->noise_capacity, s->cpmmh_rho, \
            s->d_accepts, s->d_swap_flags, \
            t_cp, cp_z, cp_mu, cp_var, cp_logw, cp_ll); break

    switch (s->N_inner) { CASE(64); CASE(128); CASE(256); CASE(512);
        default: fprintf(stderr, "Unsupported N_inner=%d\n", s->N_inner); exit(1); }
    #undef CASE
}

static void smc2_launch_commit_noise(SMC2StateCUDA* s, int t_start) {
    noise_t* curr_z  = s->d_z_noise[s->noise_buf];
    noise_t* other_z = s->d_z_noise[1 - s->noise_buf];
    noise_t* curr_u  = s->d_u0_noise[s->noise_buf];
    noise_t* other_u = s->d_u0_noise[1 - s->noise_buf];

    kernel_commit_accepted_noise<<<s->N_theta, s->N_inner>>>(
        curr_z, other_z, curr_u, other_u,
        s->d_swap_flags, s->N_theta, s->N_inner,
        s->t_current, s->noise_capacity, t_start);
}

/** ESS block size: next power-of-2, clamped to 1024 */
static int smc2_ess_block_size(int N_theta) {
    int b = 1;
    while (b < N_theta) b *= 2;
    return (b > 1024) ? 1024 : b;
}


/*═══════════════════════════════════════════════════════════════════════════════
 * §10: HOST INTERNAL — Utilities
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

#define ADAPTIVE_SCALE_FACTOR (2.38f * 2.38f / (float)N_PARAMS)

void smc2_update_adaptive_covariance(SMC2StateCUDA* state) {
    if (!state->use_adaptive_proposals) return;

    float h_cov[N_PARAMS * N_PARAMS];
    float h_chol[N_PARAMS * N_PARAMS] = {0};

    int block_size = smc2_ess_block_size(state->N_theta);

    kernel_compute_particle_moments<<<1, block_size, block_size * sizeof(float)>>>(
        state->d_particles, state->d_temp_mean, state->d_temp_cov, state->N_theta
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_cov, state->d_temp_cov, N_PARAMS * N_PARAMS * sizeof(float), cudaMemcpyDeviceToHost));

    /* Regularize + Cholesky */
    for (int i = 0; i < N_PARAMS; i++) h_cov[i * N_PARAMS + i] += 1e-6f;

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

/** Ensure y_history can hold `needed` observations */
static void smc2_ensure_y_capacity(SMC2StateCUDA* state, int needed) {
    if (needed <= state->y_history_capacity) return;
    int new_cap = state->y_history_capacity;
    while (new_cap < needed) new_cap *= 2;
    float* new_hist;
    CUDA_CHECK(cudaMalloc(&new_hist, new_cap * sizeof(float)));
    if (state->y_history_len > 0) {
        CUDA_CHECK(cudaMemcpy(new_hist, state->d_y_history,
                              state->y_history_len * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    cudaFree(state->d_y_history);
    state->d_y_history = new_hist;
    state->y_history_capacity = new_cap;
}

/** Upload constant memory for model specification */
static void smc2_upload_constants(SMC2StateCUDA* state) {
    CUDA_CHECK(cudaMemcpyToSymbol(d_prior, &state->prior, sizeof(SVPrior)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_bounds, &state->bounds, sizeof(SVBounds)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_theta_curve, &state->theta_curve, sizeof(SVCurve)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_proposal_std, state->proposal_std, N_PARAMS * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_fixed_mask, state->fixed_mask, N_PARAMS * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_fixed_values, state->fixed_values, N_PARAMS * sizeof(float)));
}

/** Upload initial Cholesky from proposal_std (diagonal) */
static void smc2_upload_initial_cholesky(SMC2StateCUDA* state) {
    float h_chol[N_PARAMS * N_PARAMS] = {0};
    float scale = sqrtf(ADAPTIVE_SCALE_FACTOR);
    for (int i = 0; i < N_PARAMS; i++)
        h_chol[i * N_PARAMS + i] = state->proposal_std[i] * scale;
    CUDA_CHECK(cudaMemcpyToSymbol(d_proposal_chol, h_chol, N_PARAMS * N_PARAMS * sizeof(float)));
}

/** Save checkpoint at the current timestep */
static void smc2_save_checkpoint_if_due(SMC2StateCUDA* state, cudaStream_t stream = 0) {
    if (state->fixed_lag_L <= 0) return;
    int target = (state->t_current / state->fixed_lag_L) * state->fixed_lag_L;
    if (target <= state->t_checkpoint || state->t_current <= 0) return;

    kernel_save_checkpoint<<<state->N_theta, state->N_inner, 0, stream>>>(
        state->d_particles,
        state->d_checkpoint_z, state->d_checkpoint_mu_h,
        state->d_checkpoint_var_h, state->d_checkpoint_log_w, state->d_checkpoint_ll,
        state->N_theta, state->N_inner);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    state->t_checkpoint = target;
}


/*═══════════════════════════════════════════════════════════════════════════════
 * §11: HOST — Memory Management
 *═══════════════════════════════════════════════════════════════════════════════*/

#define ALLOC_THETA_FIELD(field) \
    CUDA_CHECK(cudaMalloc(&state->d_particles.field, N_theta * sizeof(float))); \
    CUDA_CHECK(cudaMalloc(&state->d_particles_temp.field, N_theta * sizeof(float)))

#define ALLOC_INNER_FIELD(field) \
    CUDA_CHECK(cudaMalloc(&state->d_particles.field, N_total * sizeof(float))); \
    CUDA_CHECK(cudaMalloc(&state->d_particles_temp.field, N_total * sizeof(float)))

SMC2StateCUDA* smc2_cuda_alloc(int N_theta, int N_inner) {
    if (N_theta > MAX_N_THETA) {
        fprintf(stderr, "ERROR: N_theta=%d exceeds %d.\n", N_theta, MAX_N_THETA);
        exit(EXIT_FAILURE);
    }
    if (N_theta < 2) {
        fprintf(stderr, "ERROR: N_theta=%d too small, need >= 2.\n", N_theta);
        exit(EXIT_FAILURE);
    }
    if (N_inner != 64 && N_inner != 128 && N_inner != 256 && N_inner != 512) {
        fprintf(stderr, "ERROR: N_inner=%d not supported. Use 64, 128, 256, or 512.\n", N_inner);
        exit(EXIT_FAILURE);
    }

    SMC2StateCUDA* state = (SMC2StateCUDA*)calloc(1, sizeof(SMC2StateCUDA));
    if (!state) return NULL;

    state->N_theta = N_theta;
    state->N_inner = N_inner;
    int N_total = N_theta * N_inner;

    /* ── Particle SoA (both buffers) ── */
    ALLOC_THETA_FIELD(rho);          ALLOC_THETA_FIELD(sigma_total);
    ALLOC_THETA_FIELD(r_split);      ALLOC_THETA_FIELD(mu_base);
    ALLOC_THETA_FIELD(mu_scale);     ALLOC_THETA_FIELD(mu_rate);
    ALLOC_THETA_FIELD(sigma_scale);  ALLOC_THETA_FIELD(sigma_rate);
    ALLOC_THETA_FIELD(log_weight);   ALLOC_THETA_FIELD(weight);
    ALLOC_THETA_FIELD(log_likelihood); ALLOC_THETA_FIELD(ess_inner);

    ALLOC_INNER_FIELD(inner_z);      ALLOC_INNER_FIELD(inner_mu_h);
    ALLOC_INNER_FIELD(inner_var_h);  ALLOC_INNER_FIELD(inner_log_w);
    CUDA_CHECK(cudaMalloc(&state->d_particles.rng_states, N_total * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc(&state->d_particles_temp.rng_states, N_total * sizeof(curandState)));

    /* ── RNG seed ── */
    kernel_init_rng<<<(N_total + 255) / 256, 256>>>(state->d_particles.rng_states, 12345ULL, N_total);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* ── Observation history ── */
    state->y_history_capacity = 8000;
    CUDA_CHECK(cudaMalloc(&state->d_y_history, state->y_history_capacity * sizeof(float)));

    /* ── Scratch ── */
    CUDA_CHECK(cudaMalloc(&state->d_ancestors, N_theta * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state->d_uniform, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_ess, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_accepts, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state->d_swap_flags, N_theta * sizeof(int)));

    /* ── CPMMH noise buffers ── */
    {
        int64_t per_slot_bytes = (int64_t)N_theta * N_inner * NOISE_SIZEOF;
        int default_cap = (int)(512LL * 1024 * 1024 / (per_slot_bytes > 0 ? per_slot_bytes : 1));
        if (default_cap < 64) default_cap = 64;
        if (default_cap > 2048) default_cap = 2048;
        state->noise_capacity = default_cap;
    }
    int64_t z_noise_size = (int64_t)N_theta * N_inner * state->noise_capacity;
    int64_t u0_noise_size = (int64_t)N_theta * state->noise_capacity;
    CUDA_CHECK(cudaMalloc(&state->d_z_noise[0], noise_array_bytes(z_noise_size)));
    CUDA_CHECK(cudaMalloc(&state->d_z_noise[1], noise_array_bytes(z_noise_size)));
    CUDA_CHECK(cudaMalloc(&state->d_u0_noise[0], noise_array_bytes(u0_noise_size)));
    CUDA_CHECK(cudaMalloc(&state->d_u0_noise[1], noise_array_bytes(u0_noise_size)));

    /* ── Checkpoint buffers + scratch ── */
    CUDA_CHECK(cudaMalloc(&state->d_checkpoint_z, N_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_checkpoint_mu_h, N_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_checkpoint_var_h, N_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_checkpoint_log_w, N_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_checkpoint_ll, N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_checkpoint_scratch_z, N_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_checkpoint_scratch_mu_h, N_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_checkpoint_scratch_var_h, N_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_checkpoint_scratch_log_w, N_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_checkpoint_scratch_ll, N_theta * sizeof(float)));

    /* ── Adaptive proposal scratch ── */
    CUDA_CHECK(cudaMalloc(&state->d_temp_mean, N_PARAMS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_temp_cov, N_PARAMS * N_PARAMS * sizeof(float)));

    /* ── Batch update resources ── */
    CUDA_CHECK(cudaStreamCreate(&state->compute_stream));
    CUDA_CHECK(cudaMallocHost(&state->h_ess_pinned, sizeof(float)));

    /* ── Defaults (structural — model config is separate) ── */
    state->ess_threshold_outer = 0.5f;
    state->ess_threshold_inner = 0.5f;
    state->K_rejuv = 8;
    state->cpmmh_rho = 0.99f;
    state->noise_buf = 0;
    state->fixed_lag_L = 50;
    state->t_checkpoint = -1;
    state->ess_check_interval = 10;
    state->use_adaptive_proposals = true;
    state->host_rng_state = 0x853C49E6748FEA9BULL ^ (uint64_t)time(NULL);

    /* ── Model defaults (named functions, overridable) ── */
    state->prior = sv_default_prior();
    state->bounds = sv_default_bounds();
    state->theta_curve = sv_default_theta_curve();
    sv_default_proposal_std(state->proposal_std);
    memset(state->fixed_mask, 0, sizeof(state->fixed_mask));
    memset(state->fixed_values, 0, sizeof(state->fixed_values));

    return state;
}

#undef ALLOC_THETA_FIELD
#undef ALLOC_INNER_FIELD

#define FREE_FIELD(field) do { cudaFree(state->d_particles.field); cudaFree(state->d_particles_temp.field); } while(0)

void smc2_cuda_free(SMC2StateCUDA* state) {
    if (!state) return;

    FREE_FIELD(rho);          FREE_FIELD(sigma_total);
    FREE_FIELD(r_split);      FREE_FIELD(mu_base);
    FREE_FIELD(mu_scale);     FREE_FIELD(mu_rate);
    FREE_FIELD(sigma_scale);  FREE_FIELD(sigma_rate);
    FREE_FIELD(log_weight);   FREE_FIELD(weight);
    FREE_FIELD(log_likelihood); FREE_FIELD(ess_inner);
    FREE_FIELD(inner_z);      FREE_FIELD(inner_mu_h);
    FREE_FIELD(inner_var_h);  FREE_FIELD(inner_log_w);
    cudaFree(state->d_particles.rng_states);
    cudaFree(state->d_particles_temp.rng_states);

    cudaFree(state->d_y_history);
    cudaFree(state->d_ancestors); cudaFree(state->d_uniform);
    cudaFree(state->d_ess);       cudaFree(state->d_accepts);
    cudaFree(state->d_swap_flags);

    cudaFree(state->d_z_noise[0]); cudaFree(state->d_z_noise[1]);
    cudaFree(state->d_u0_noise[0]); cudaFree(state->d_u0_noise[1]);

    cudaFree(state->d_checkpoint_z);     cudaFree(state->d_checkpoint_mu_h);
    cudaFree(state->d_checkpoint_var_h); cudaFree(state->d_checkpoint_log_w);
    cudaFree(state->d_checkpoint_ll);
    cudaFree(state->d_checkpoint_scratch_z);     cudaFree(state->d_checkpoint_scratch_mu_h);
    cudaFree(state->d_checkpoint_scratch_var_h); cudaFree(state->d_checkpoint_scratch_log_w);
    cudaFree(state->d_checkpoint_scratch_ll);

    cudaFree(state->d_temp_mean); cudaFree(state->d_temp_cov);

    cudaStreamDestroy(state->compute_stream);
    cudaFreeHost(state->h_ess_pinned);

    free(state);
}

#undef FREE_FIELD

void smc2_cuda_set_noise_capacity(SMC2StateCUDA* state, int capacity) {
    if (state->fixed_lag_L > 0) {
        int min_cap = state->fixed_lag_L + 256;
        if (capacity > min_cap) capacity = min_cap;
    }
    if (capacity <= state->noise_capacity) return;
    if (capacity > MAX_NOISE_CAPACITY) {
        fprintf(stderr, "WARNING: noise_capacity=%d clamped to %d.\n", capacity, MAX_NOISE_CAPACITY);
        capacity = MAX_NOISE_CAPACITY;
        if (capacity <= state->noise_capacity) return;
    }

    int64_t new_z_size = (int64_t)state->N_theta * state->N_inner * capacity;
    int64_t old_z_size = (int64_t)state->N_theta * state->N_inner * state->noise_capacity;
    int64_t new_u0_size = (int64_t)state->N_theta * capacity;
    int64_t old_u0_size = (int64_t)state->N_theta * state->noise_capacity;

    int64_t total_bytes = 4 * noise_array_bytes(new_z_size) + 4 * noise_array_bytes(new_u0_size);
    if (total_bytes > (int64_t)8 * 1024 * 1024 * 1024LL) {
        static bool warned = false;
        if (!warned) {
            fprintf(stderr, "WARNING: Noise buffer %.1f GB too large. Use set_fixed_lag(L>0).\n",
                    (double)total_bytes / (1024.0 * 1024.0 * 1024.0));
            warned = true;
        }
        return;
    }

    noise_t *new_z_0, *new_z_1, *new_u0_0, *new_u0_1;
    CUDA_CHECK(cudaMalloc(&new_z_0, noise_array_bytes(new_z_size)));
    CUDA_CHECK(cudaMalloc(&new_z_1, noise_array_bytes(new_z_size)));
    CUDA_CHECK(cudaMalloc(&new_u0_0, noise_array_bytes(new_u0_size)));
    CUDA_CHECK(cudaMalloc(&new_u0_1, noise_array_bytes(new_u0_size)));

    if (state->d_z_noise[0] && old_z_size > 0) {
        int64_t copy_z = (old_z_size < new_z_size) ? old_z_size : new_z_size;
        CUDA_CHECK(cudaMemcpy(new_z_0, state->d_z_noise[0], noise_array_bytes(copy_z), cudaMemcpyDeviceToDevice));
    }
    if (state->d_u0_noise[0] && old_u0_size > 0) {
        int64_t copy_u0 = (old_u0_size < new_u0_size) ? old_u0_size : new_u0_size;
        CUDA_CHECK(cudaMemcpy(new_u0_0, state->d_u0_noise[0], noise_array_bytes(copy_u0), cudaMemcpyDeviceToDevice));
    }

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

    if (L > 0) {
        int target_cap = L + 256;
        if (state->noise_capacity != target_cap) {
            int64_t new_z_size = (int64_t)state->N_theta * state->N_inner * target_cap;
            int64_t new_u0_size = (int64_t)state->N_theta * target_cap;
            int64_t total_bytes = 4 * noise_array_bytes(new_z_size) + 4 * noise_array_bytes(new_u0_size);
            if (total_bytes > (int64_t)8 * 1024 * 1024 * 1024LL) {
                fprintf(stderr, "WARNING: Fixed-lag noise %.1f GB too large.\n",
                        (double)total_bytes / (1024.0 * 1024.0 * 1024.0));
                return;
            }

            noise_t *new_z_0, *new_z_1, *new_u0_0, *new_u0_1;
            CUDA_CHECK(cudaMalloc(&new_z_0, noise_array_bytes(new_z_size)));
            CUDA_CHECK(cudaMalloc(&new_z_1, noise_array_bytes(new_z_size)));
            CUDA_CHECK(cudaMalloc(&new_u0_0, noise_array_bytes(new_u0_size)));
            CUDA_CHECK(cudaMalloc(&new_u0_1, noise_array_bytes(new_u0_size)));

            cudaFree(state->d_z_noise[0]); cudaFree(state->d_z_noise[1]);
            cudaFree(state->d_u0_noise[0]); cudaFree(state->d_u0_noise[1]);

            state->d_z_noise[0] = new_z_0; state->d_z_noise[1] = new_z_1;
            state->d_u0_noise[0] = new_u0_0; state->d_u0_noise[1] = new_u0_1;
            state->noise_buf = 0;
            state->noise_capacity = target_cap;
        }
    }
}


/*═══════════════════════════════════════════════════════════════════════════════
 * §12: HOST — Configuration
 *═══════════════════════════════════════════════════════════════════════════════*/

void smc2_cuda_set_seed(SMC2StateCUDA* state, uint64_t seed) {
    state->user_seed = seed;
    if (seed != 0) state->host_rng_state = 0x853C49E6748FEA9BULL ^ seed;
}

void smc2_cuda_set_proposal_std(SMC2StateCUDA* state, const float* std) {
    if (std) {
        memcpy(state->proposal_std, std, N_PARAMS * sizeof(float));
    } else {
        sv_default_proposal_std(state->proposal_std);
    }
    CUDA_CHECK(cudaMemcpyToSymbol(d_proposal_std, state->proposal_std, N_PARAMS * sizeof(float)));
}

void smc2_cuda_set_cpmmh_rho(SMC2StateCUDA* state, float rho) {
    state->cpmmh_rho = rho;
}

void smc2_cuda_set_fixed_params(SMC2StateCUDA* state,
                                 const uint8_t* mask, const float* values) {
    memcpy(state->fixed_mask, mask, N_PARAMS * sizeof(uint8_t));
    memcpy(state->fixed_values, values, N_PARAMS * sizeof(float));
    CUDA_CHECK(cudaMemcpyToSymbol(d_fixed_mask, state->fixed_mask, N_PARAMS * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_fixed_values, state->fixed_values, N_PARAMS * sizeof(float)));
}


/*═══════════════════════════════════════════════════════════════════════════════
 * §13: HOST — Resample + Rejuvenate (THE shared path)
 *
 * This is the single place where the outer resample → copy → swap →
 * checkpoint reindex → CPMMH rejuvenation sequence executes.
 * Both update() and update_batch() call this when ESS drops.
 *
 * Eliminates the ~150-line duplication from the original code.
 *═══════════════════════════════════════════════════════════════════════════════*/

static void smc2_resample_rejuvenate(SMC2StateCUDA* state) {
    state->n_resamples++;

    /* ── 1. Systematic resample ── */
    float h_uniform = xorshift64star_uniform(&state->host_rng_state);
    CUDA_CHECK(cudaMemcpy(state->d_uniform, &h_uniform, sizeof(float), cudaMemcpyHostToDevice));

    int resample_block = (state->N_theta < 1024) ? state->N_theta : 1024;
    kernel_outer_resample<<<1, resample_block, state->N_theta * sizeof(float)>>>(
        state->d_particles, state->d_ancestors, state->d_uniform, state->N_theta);

    /* ── 2. Copy particles by ancestors ── */
    unsigned long long resample_seed = time(NULL) * 1000ULL + state->n_resamples * 12345ULL;
    kernel_copy_theta_particles<<<state->N_theta, state->N_inner>>>(
        state->d_particles, state->d_particles_temp, state->d_ancestors,
        state->N_theta, state->N_inner, resample_seed);

    /* ── 3. Copy noise window by ancestors ── */
    int other_buf = 1 - state->noise_buf;
    int t_noise_start = (state->fixed_lag_L > 0 && state->t_checkpoint >= 0)
                        ? state->t_checkpoint : 0;
    kernel_copy_noise_arrays<<<state->N_theta, state->N_inner>>>(
        state->d_z_noise[state->noise_buf], state->d_z_noise[other_buf],
        state->d_u0_noise[state->noise_buf], state->d_u0_noise[other_buf],
        state->d_ancestors, state->N_theta, state->N_inner,
        state->t_current, state->noise_capacity, t_noise_start);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* ── 4. Swap buffers (noise + particle ping-pong) ── */
    state->noise_buf = other_buf;
    { ThetaParticlesSoA tmp = state->d_particles;
      state->d_particles = state->d_particles_temp;
      state->d_particles_temp = tmp; }

    /* ── 5. Reindex checkpoint via dedicated scratch ── */
    if (state->fixed_lag_L > 0 && state->t_checkpoint >= 0) {
        kernel_copy_checkpoint<<<state->N_theta, state->N_inner>>>(
            state->d_checkpoint_z, state->d_checkpoint_mu_h,
            state->d_checkpoint_var_h, state->d_checkpoint_log_w, state->d_checkpoint_ll,
            state->d_checkpoint_scratch_z, state->d_checkpoint_scratch_mu_h,
            state->d_checkpoint_scratch_var_h, state->d_checkpoint_scratch_log_w,
            state->d_checkpoint_scratch_ll,
            state->d_ancestors, state->N_theta, state->N_inner);
        CUDA_CHECK(cudaDeviceSynchronize());

        /* Pointer swap instead of 5x D2D memcpy */
        float* tmp_f;
        #define SWAP_CP(a, b) do { tmp_f = (a); (a) = (b); (b) = tmp_f; } while(0)
        SWAP_CP(state->d_checkpoint_z,     state->d_checkpoint_scratch_z);
        SWAP_CP(state->d_checkpoint_mu_h,  state->d_checkpoint_scratch_mu_h);
        SWAP_CP(state->d_checkpoint_var_h, state->d_checkpoint_scratch_var_h);
        SWAP_CP(state->d_checkpoint_log_w, state->d_checkpoint_scratch_log_w);
        SWAP_CP(state->d_checkpoint_ll,    state->d_checkpoint_scratch_ll);
        #undef SWAP_CP
    }

    /* ── 6. Resolve checkpoint for CPMMH replay ── */
    int t_checkpoint_use = -1;
    const float *cp_z = nullptr, *cp_mu = nullptr, *cp_var = nullptr,
                *cp_logw = nullptr, *cp_ll = nullptr;
    if (state->fixed_lag_L > 0 && state->t_checkpoint >= 0) {
        int steps = state->t_current - state->t_checkpoint;
        if (steps > 0 && steps <= 2 * state->fixed_lag_L) {
            t_checkpoint_use = state->t_checkpoint;
            cp_z = state->d_checkpoint_z;       cp_mu = state->d_checkpoint_mu_h;
            cp_var = state->d_checkpoint_var_h; cp_logw = state->d_checkpoint_log_w;
            cp_ll = state->d_checkpoint_ll;
        }
    }

    /* ── 7. Adaptive proposal + CPMMH moves ── */
    smc2_update_adaptive_covariance(state);

    int t_start_commit = (t_checkpoint_use >= 0) ? (t_checkpoint_use + 1) : 0;

    {
        int zero = 0;
        CUDA_CHECK(cudaMemcpy(state->d_accepts, &zero, sizeof(int), cudaMemcpyHostToDevice));
    }

    for (int k = 0; k < state->K_rejuv; k++) {
        smc2_launch_cpmmh(state, t_checkpoint_use, cp_z, cp_mu, cp_var, cp_logw, cp_ll);
        smc2_launch_commit_noise(state, t_start_commit);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Accumulate accept stats */
    {
        int h_accepts = 0;
        CUDA_CHECK(cudaMemcpy(&h_accepts, state->d_accepts, sizeof(int), cudaMemcpyDeviceToHost));
        state->n_rejuv_accepts += h_accepts;
        state->n_rejuv_total += state->K_rejuv * state->N_theta;
    }

    /* ── 8. Reset weights ── */
    kernel_reset_outer_weights<<<(state->N_theta + 255) / 256, 256>>>(
        state->d_particles, state->N_theta);
}


/*═══════════════════════════════════════════════════════════════════════════════
 * §14: HOST — Update Loop
 *═══════════════════════════════════════════════════════════════════════════════*/

void smc2_cuda_init_from_prior(SMC2StateCUDA* state) {
    smc2_upload_constants(state);
    smc2_upload_initial_cholesky(state);

    int N_total = state->N_theta * state->N_inner;
    unsigned long long rng_seed = (state->user_seed != 0) ? state->user_seed : 12345ULL;
    kernel_init_rng<<<(N_total + 255) / 256, 256>>>(state->d_particles.rng_states, rng_seed, N_total);
    CUDA_CHECK(cudaDeviceSynchronize());

    kernel_init_from_prior<<<state->N_theta, state->N_inner>>>(
        state->d_particles, state->N_theta, state->N_inner,
        state->d_z_noise[state->noise_buf], state->d_u0_noise[state->noise_buf],
        state->noise_capacity);
    CUDA_CHECK(cudaDeviceSynchronize());

    state->n_resamples = 0;
    state->n_rejuv_accepts = 0;
    state->n_rejuv_total = 0;
    state->y_history_len = 0;
    state->t_current = -1;
}

void smc2_cuda_init_warm(SMC2StateCUDA* state,
                          const float* warm_mean, const float* warm_cov) {
    /* Update prior from warm values */
    float* prior_means[N_PARAMS] = {
        &state->prior.rho_mean, &state->prior.sigma_total_mean,
        &state->prior.r_split_mean, &state->prior.mu_base_mean,
        &state->prior.mu_scale_mean, &state->prior.mu_rate_mean,
        &state->prior.sigma_scale_mean, &state->prior.sigma_rate_mean
    };
    float* prior_stds[N_PARAMS] = {
        &state->prior.rho_std, &state->prior.sigma_total_std,
        &state->prior.r_split_std, &state->prior.mu_base_std,
        &state->prior.mu_scale_std, &state->prior.mu_rate_std,
        &state->prior.sigma_scale_std, &state->prior.sigma_rate_std
    };
    for (int i = 0; i < N_PARAMS; i++) {
        *prior_means[i] = warm_mean[i];
        *prior_stds[i] = sqrtf(fmaxf(warm_cov[i * N_PARAMS + i], 1e-8f));
    }

    /* Build Cholesky from warm_cov * adaptive_scale */
    float adaptive_scale = 2.38f * 2.38f / (float)N_PARAMS;
    float h_cov_scaled[N_PARAMS * N_PARAMS];
    for (int i = 0; i < N_PARAMS * N_PARAMS; i++) h_cov_scaled[i] = warm_cov[i] * adaptive_scale;
    for (int i = 0; i < N_PARAMS; i++) h_cov_scaled[i * N_PARAMS + i] += 1e-8f;

    float h_chol[N_PARAMS * N_PARAMS] = {0};
    int chol_success = 1;
    for (int i = 0; i < N_PARAMS; i++) {
        for (int j = 0; j <= i; j++) {
            float sum = 0.0f;
            for (int k = 0; k < j; k++) sum += h_chol[i * N_PARAMS + k] * h_chol[j * N_PARAMS + k];
            if (i == j) {
                float val = h_cov_scaled[i * N_PARAMS + i] - sum;
                if (val <= 0.0f) { chol_success = 0; break; }
                h_chol[i * N_PARAMS + j] = sqrtf(val);
            } else {
                float diag = h_chol[j * N_PARAMS + j];
                h_chol[i * N_PARAMS + j] = (diag > 1e-10f)
                    ? (h_cov_scaled[i * N_PARAMS + j] - sum) / diag : 0.0f;
            }
        }
        if (!chol_success) break;
    }
    if (!chol_success) {
        memset(h_chol, 0, sizeof(h_chol));
        float scale = sqrtf(adaptive_scale);
        for (int i = 0; i < N_PARAMS; i++)
            h_chol[i * N_PARAMS + i] = sqrtf(fmaxf(warm_cov[i * N_PARAMS + i], 1e-8f)) * scale;
    }

    for (int i = 0; i < N_PARAMS; i++)
        state->proposal_std[i] = sqrtf(fmaxf(warm_cov[i * N_PARAMS + i], 1e-8f));

    smc2_upload_constants(state);
    CUDA_CHECK(cudaMemcpyToSymbol(d_proposal_chol, h_chol, N_PARAMS * N_PARAMS * sizeof(float)));

    int N_total = state->N_theta * state->N_inner;
    unsigned long long rng_seed = (state->user_seed != 0) ? state->user_seed : 12345ULL;
    kernel_init_rng<<<(N_total + 255) / 256, 256>>>(state->d_particles.rng_states, rng_seed, N_total);
    CUDA_CHECK(cudaDeviceSynchronize());

    kernel_init_from_prior<<<state->N_theta, state->N_inner>>>(
        state->d_particles, state->N_theta, state->N_inner,
        state->d_z_noise[state->noise_buf], state->d_u0_noise[state->noise_buf],
        state->noise_capacity);
    CUDA_CHECK(cudaDeviceSynchronize());

    state->n_resamples = 0;
    state->n_rejuv_accepts = 0;
    state->n_rejuv_total = 0;
    state->y_history_len = 0;
    state->t_current = -1;
}

float smc2_cuda_update(SMC2StateCUDA* state, float y_obs) {
    /* Store observation */
    smc2_ensure_y_capacity(state, state->y_history_len + 1);
    CUDA_CHECK(cudaMemcpy(&state->d_y_history[state->y_history_len], &y_obs,
                          sizeof(float), cudaMemcpyHostToDevice));
    state->y_history_len++;
    state->t_current++;

    /* Noise capacity growth for L=0 mode */
    if (state->fixed_lag_L == 0 && state->t_current >= state->noise_capacity) {
        static bool growth_failed = false;
        if (!growth_failed) {
            int old_cap = state->noise_capacity;
            smc2_cuda_set_noise_capacity(state, state->noise_capacity * 2);
            if (state->noise_capacity == old_cap) {
                growth_failed = true;
                fprintf(stderr, "NOTE: L=0 CPMMH using wrapped noise from t=%d.\n", state->t_current);
            }
        }
    }

    /* RBPF forward step */
    smc2_launch_rbpf_step(state, y_obs);

    /* Compute ESS */
    int ess_block = smc2_ess_block_size(state->N_theta);
    kernel_compute_outer_ess<<<1, ess_block, 32 * sizeof(float)>>>(
        state->d_particles, state->d_ess, state->N_theta);
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_ess;
    CUDA_CHECK(cudaMemcpy(&h_ess, state->d_ess, sizeof(float), cudaMemcpyDeviceToHost));

    /* Resample + rejuvenate if needed */
    if (h_ess < state->ess_threshold_outer * state->N_theta) {
        smc2_resample_rejuvenate(state);

        /* Recompute ESS after rejuvenation */
        kernel_compute_outer_ess<<<1, ess_block, 32 * sizeof(float)>>>(
            state->d_particles, state->d_ess, state->N_theta);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_ess, state->d_ess, sizeof(float), cudaMemcpyDeviceToHost));
    }

    /* Save checkpoint */
    smc2_save_checkpoint_if_due(state);

    return h_ess;
}

float smc2_cuda_update_batch(SMC2StateCUDA* state, const float* y_batch, int n_obs) {
    cudaStream_t stream = state->compute_stream;
    int interval = state->ess_check_interval;
    int ess_block = smc2_ess_block_size(state->N_theta);

    /* Ensure capacity + bulk upload */
    smc2_ensure_y_capacity(state, state->y_history_len + n_obs);
    CUDA_CHECK(cudaMemcpy(&state->d_y_history[state->y_history_len],
                          y_batch, n_obs * sizeof(float), cudaMemcpyHostToDevice));

    float h_ess = (float)state->N_theta;

    for (int i = 0; i < n_obs; i++) {
        float y_obs = y_batch[i];
        state->y_history_len++;
        state->t_current++;

        /* Noise capacity growth for L=0 */
        if (state->fixed_lag_L == 0 && state->t_current >= state->noise_capacity) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            int old_cap = state->noise_capacity;
            smc2_cuda_set_noise_capacity(state, state->noise_capacity * 2);
            if (state->noise_capacity == old_cap) break;
        }

        /* RBPF step + ESS on stream */
        smc2_launch_rbpf_step(state, y_obs, stream);
        kernel_compute_outer_ess<<<1, ess_block, 32 * sizeof(float), stream>>>(
            state->d_particles, state->d_ess, state->N_theta);

        /* Save checkpoint if due */
        smc2_save_checkpoint_if_due(state, stream);

        /* Deferred ESS check */
        bool check_now = ((i + 1) % interval == 0) || (i == n_obs - 1);
        if (!check_now) continue;

        CUDA_CHECK(cudaMemcpyAsync(state->h_ess_pinned, state->d_ess,
                                    sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        h_ess = *state->h_ess_pinned;

        if (h_ess >= state->ess_threshold_outer * state->N_theta) continue;

        /* ── RESAMPLE PATH (via shared function) ── */
        smc2_resample_rejuvenate(state);

        kernel_compute_outer_ess<<<1, ess_block, 32 * sizeof(float)>>>(
            state->d_particles, state->d_ess, state->N_theta);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_ess, state->d_ess, sizeof(float), cudaMemcpyDeviceToHost));
    }

    return h_ess;
}


/*═══════════════════════════════════════════════════════════════════════════════
 * §15: HOST — Queries & Diagnostics
 *═══════════════════════════════════════════════════════════════════════════════*/

static void smc2_cuda_get_theta_moments_internal(
    SMC2StateCUDA* state, float* theta_mean, float* theta_std
) {
    float* h_weight = (float*)malloc(state->N_theta * sizeof(float));
    float* h_params[N_PARAMS];
    for (int i = 0; i < N_PARAMS; i++) h_params[i] = (float*)malloc(state->N_theta * sizeof(float));

    CUDA_CHECK(cudaMemcpy(h_weight, state->d_particles.weight, state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));

    float* param_ptrs[N_PARAMS] = {
        state->d_particles.rho, state->d_particles.sigma_total,
        state->d_particles.r_split, state->d_particles.mu_base,
        state->d_particles.mu_scale, state->d_particles.mu_rate,
        state->d_particles.sigma_scale, state->d_particles.sigma_rate
    };
    for (int i = 0; i < N_PARAMS; i++)
        CUDA_CHECK(cudaMemcpy(h_params[i], param_ptrs[i], state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N_PARAMS; i++) theta_mean[i] = 0.0f;
    for (int j = 0; j < state->N_theta; j++) {
        float w = h_weight[j];
        for (int i = 0; i < N_PARAMS; i++) theta_mean[i] += w * h_params[i][j];
    }

    if (theta_std) {
        for (int i = 0; i < N_PARAMS; i++) theta_std[i] = 0.0f;
        for (int j = 0; j < state->N_theta; j++) {
            float w = h_weight[j];
            for (int i = 0; i < N_PARAMS; i++) {
                float d = h_params[i][j] - theta_mean[i];
                theta_std[i] += w * d * d;
            }
        }
        for (int i = 0; i < N_PARAMS; i++) theta_std[i] = sqrtf(theta_std[i]);
    }

    free(h_weight);
    for (int i = 0; i < N_PARAMS; i++) free(h_params[i]);
}

void smc2_cuda_get_theta_mean(SMC2StateCUDA* state, float* theta_mean) {
    smc2_cuda_get_theta_moments_internal(state, theta_mean, NULL);
}

void smc2_cuda_get_theta_std(SMC2StateCUDA* state, float* theta_std) {
    float theta_mean[N_PARAMS];
    smc2_cuda_get_theta_moments_internal(state, theta_mean, theta_std);
}

void smc2_cuda_get_theta_cov(SMC2StateCUDA* state, float* theta_mean, float* theta_cov) {
    float* h_weight = (float*)malloc(state->N_theta * sizeof(float));
    float* h_params[N_PARAMS];
    for (int i = 0; i < N_PARAMS; i++) h_params[i] = (float*)malloc(state->N_theta * sizeof(float));

    CUDA_CHECK(cudaMemcpy(h_weight, state->d_particles.weight, state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));

    float* param_ptrs[N_PARAMS] = {
        state->d_particles.rho, state->d_particles.sigma_total,
        state->d_particles.r_split, state->d_particles.mu_base,
        state->d_particles.mu_scale, state->d_particles.mu_rate,
        state->d_particles.sigma_scale, state->d_particles.sigma_rate
    };
    for (int i = 0; i < N_PARAMS; i++)
        CUDA_CHECK(cudaMemcpy(h_params[i], param_ptrs[i], state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N_PARAMS; i++) theta_mean[i] = 0.0f;
    for (int j = 0; j < state->N_theta; j++) {
        float w = h_weight[j];
        for (int i = 0; i < N_PARAMS; i++) theta_mean[i] += w * h_params[i][j];
    }

    for (int i = 0; i < N_PARAMS * N_PARAMS; i++) theta_cov[i] = 0.0f;
    for (int k = 0; k < state->N_theta; k++) {
        float w = h_weight[k];
        for (int i = 0; i < N_PARAMS; i++) {
            float di = h_params[i][k] - theta_mean[i];
            for (int j = i; j < N_PARAMS; j++) {
                float dj = h_params[j][k] - theta_mean[j];
                theta_cov[i * N_PARAMS + j] += w * di * dj;
            }
        }
    }
    for (int i = 0; i < N_PARAMS; i++)
        for (int j = 0; j < i; j++)
            theta_cov[i * N_PARAMS + j] = theta_cov[j * N_PARAMS + i];

    free(h_weight);
    for (int i = 0; i < N_PARAMS; i++) free(h_params[i]);
}

float smc2_cuda_get_z_mean(SMC2StateCUDA* state) {
    int N_total = state->N_theta * state->N_inner;
    float* h_weight = (float*)malloc(state->N_theta * sizeof(float));
    float* h_z = (float*)malloc(N_total * sizeof(float));

    CUDA_CHECK(cudaMemcpy(h_weight, state->d_particles.weight,
                          state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_z, state->d_particles.inner_z,
                          N_total * sizeof(float), cudaMemcpyDeviceToHost));

    float z_global = 0.0f;
    for (int j = 0; j < state->N_theta; j++) {
        float z_sum = 0.0f;
        int base = j * state->N_inner;
        for (int i = 0; i < state->N_inner; i++) {
            float z = 1.5f * (1.0f + tanhf(h_z[base + i]));
            z_sum += z;
        }
        z_global += h_weight[j] * (z_sum / state->N_inner);
    }

    free(h_weight);
    free(h_z);
    return z_global;
}

void smc2_cuda_get_z_range(SMC2StateCUDA* state,
                            float* z_mean_out, float* z_min_out, float* z_max_out) {
    int N_total = state->N_theta * state->N_inner;
    float* h_weight = (float*)malloc(state->N_theta * sizeof(float));
    float* h_z = (float*)malloc(N_total * sizeof(float));

    CUDA_CHECK(cudaMemcpy(h_weight, state->d_particles.weight,
                          state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_z, state->d_particles.inner_z,
                          N_total * sizeof(float), cudaMemcpyDeviceToHost));

    float z_global = 0.0f, z_min = 1e6f, z_max = -1e6f;

    for (int j = 0; j < state->N_theta; j++) {
        float z_sum = 0.0f;
        int base = j * state->N_inner;
        for (int i = 0; i < state->N_inner; i++) {
            float z = 1.5f * (1.0f + tanhf(h_z[base + i]));
            z_sum += z;
            if (z < z_min) z_min = z;
            if (z > z_max) z_max = z;
        }
        z_global += h_weight[j] * (z_sum / state->N_inner);
    }

    free(h_weight); free(h_z);
    if (z_mean_out) *z_mean_out = z_global;
    if (z_min_out)  *z_min_out  = z_min;
    if (z_max_out)  *z_max_out  = z_max;
}

void smc2_cuda_get_z_range_robust(SMC2StateCUDA* state,
                                   float* z_mean_out, float* z_min_out, float* z_max_out) {
    int N_total = state->N_theta * state->N_inner;
    float* h_weight = (float*)malloc(state->N_theta * sizeof(float));
    float* h_z = (float*)malloc(N_total * sizeof(float));

    CUDA_CHECK(cudaMemcpy(h_weight, state->d_particles.weight,
                          state->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_z, state->d_particles.inner_z,
                          N_total * sizeof(float), cudaMemcpyDeviceToHost));

    float z_global = 0.0f, z_min = 1e6f, z_max = -1e6f;

    for (int j = 0; j < state->N_theta; j++) {
        float z_sum = 0.0f;
        int base = j * state->N_inner;
        for (int i = 0; i < state->N_inner; i++) {
            float z = 1.5f * (1.0f + tanhf(h_z[base + i]));
            z_sum += z;
        }
        float z_mean_j = z_sum / state->N_inner;
        if (z_mean_j < z_min) z_min = z_mean_j;
        if (z_mean_j > z_max) z_max = z_mean_j;
        z_global += h_weight[j] * z_mean_j;
    }

    free(h_weight); free(h_z);
    if (z_mean_out) *z_mean_out = z_global;
    if (z_min_out)  *z_min_out  = z_min;
    if (z_max_out)  *z_max_out  = z_max;
}

float smc2_cuda_get_outer_ess(SMC2StateCUDA* state) {
    float h_ess;
    CUDA_CHECK(cudaMemcpy(&h_ess, state->d_ess, sizeof(float), cudaMemcpyDeviceToHost));
    return h_ess;
}

void smc2_cuda_get_diagnostics(SMC2StateCUDA* state, SMC2Diagnostics* diag) {
    diag->n_resamples = state->n_resamples;
    diag->n_rejuv_accepts = state->n_rejuv_accepts;
    diag->n_rejuv_total = state->n_rejuv_total;
    diag->outer_ess = smc2_cuda_get_outer_ess(state);
    diag->accept_rate = (state->n_rejuv_total > 0)
        ? (float)state->n_rejuv_accepts / state->n_rejuv_total : 0.0f;
}
