/**
 * @file cpmmh_gpu_learn_v3.cu
 * @brief GPU CPMMH with parameter learning (v3.1 - CPU Parity)
 *
 * v3.1 changes (matching CPU test_cpmmh.c):
 *   [E] Sorting every timestep (SORT_EVERY_K=1) - matches CPU
 *   [F] Proposal stds = 10% of param range - typical MCMC tuning
 *   [G] Raw-space proposals with prior rejection - matches CPU
 *   [H] CPU-matching config: 4 chains, 2000 iters, burnin=500
 *
 * Key CPMMH algorithm (matching CPU):
 *   1. noise_prop = rho * noise_curr + sqrt(1-rho²) * fresh
 *   2. Run PF with proposed params + noise_prop → ll_prop
 *   3. MH test: log_alpha = (ll_prop + lp_prop) - (ll_curr + lp_curr)
 *   4. If accept: swap noise_curr <-> noise_prop
 *   5. If reject: keep noise_curr
 *
 * v3 fixes:
 *   [A] PF init from stationary distribution (matching CPU)
 *   [A'] Separate noise slot for init (T+1 slots)
 *   [B] Periodic cudaDeviceSynchronize for debugging
 *   [C] Guard final_iter > 1 for acceptance rate
 *
 * Previous fixes from v2:
 *   [1] N == BLOCK_SIZE enforced
 *   [2] Double-buffer resampling (no race)
 *   [3] Early-out for invalid proposals
 *   [5] u0 clamped both ends
 *   [8] cudaGetLastError checks
 *   [10] z_ceil consistent
 */

#define NOMINMAX

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <utility>
#include <random>
#include <cassert>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>

// ═══════════════════════════════════════════════════════════════════════════
// CONFIG
// ═══════════════════════════════════════════════════════════════════════════

#define OCSN_K 10
#define BLOCK_SIZE 256

// Set to 1 to match CPU test (only learn 8 params, fix theta curve)
// Set to 0 to learn all 11 params
#ifndef FIX_THETA_CURVE
#define FIX_THETA_CURVE 1
#endif

// Set to 1 to fix sigma_z to ground truth (often poorly identified)
#ifndef FIX_SIGMA_Z
#define FIX_SIGMA_Z 1
#endif

#if FIX_THETA_CURVE && FIX_SIGMA_Z
#define N_PARAMS 7   // 11 - 3 (theta) - 1 (sigma_z) = 7
#elif FIX_THETA_CURVE
#define N_PARAMS 8   // 11 - 3 (theta) = 8
#elif FIX_SIGMA_Z
#define N_PARAMS 10  // 11 - 1 (sigma_z) = 10
#else
#define N_PARAMS 11
#endif

#ifndef RHAT_CHECK_INTERVAL
#define RHAT_CHECK_INTERVAL 100
#endif

#ifndef RHAT_THRESHOLD
#define RHAT_THRESHOLD 1.10f
#endif

#ifndef MIN_ITERS_BEFORE_RHAT
#define MIN_ITERS_BEFORE_RHAT 500  // Match CPU burnin
#endif

#ifndef BURNIN_FRACTION
#define BURNIN_FRACTION 0.25f  // Match CPU: 500/2000 = 0.25
#endif

// v3 FIX [B]: Sync interval for debugging (0 = disabled)
#ifndef DEBUG_SYNC_INTERVAL
#define DEBUG_SYNC_INTERVAL 0  // Set to e.g. 50 to enable periodic sync
#endif

// RNG batch size - generate this many iterations of random numbers at once
#ifndef RNG_BATCH_SIZE
#define RNG_BATCH_SIZE 8
#endif

// v3 FIX [A]: Initial particle jitter scales
#ifndef Z_INIT_JITTER_STD
#define Z_INIT_JITTER_STD 0.1f  // Applied as: z_init + jitter_std * noise
#endif

#ifndef H_INIT_JITTER_STD
#define H_INIT_JITTER_STD 0.15f
#endif

// ═══════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════

__constant__ float c_OCSN_HALF_INV_VARS[OCSN_K];
__constant__ float c_OCSN_MEANS[OCSN_K];
__constant__ float c_OCSN_LOG_WEIGHT_NORM[OCSN_K];

static const float h_OCSN_HALF_INV_VARS[OCSN_K] = {
    0.086275f, 0.191305f, 0.312970f, 0.419970f, 0.493580f,
    0.555860f, 0.612515f, 0.667170f, 0.720815f, 0.781135f
};
static const float h_OCSN_MEANS[OCSN_K] = {
    -10.12999f, -3.97281f, -0.57354f, 1.22474f, 2.58590f,
    3.72372f, 4.73732f, 5.69446f, 6.63386f, 8.06767f
};
static const float h_OCSN_LOG_WEIGHT_NORM[OCSN_K] = {
    -6.90226f, -4.44211f, -3.18920f, -2.58131f, -2.40738f,
    -2.53515f, -2.93189f, -3.66033f, -4.88802f, -7.46701f
};

// ═══════════════════════════════════════════════════════════════════════════
// PARAMETER STRUCTURE
// ═══════════════════════════════════════════════════════════════════════════

struct ChainParams {
    float* rho;
    float* sigma_z;
    float* mu_base;
    float* mu_scale;
    float* mu_rate;
    float* sigma_base;
    float* sigma_scale;
    float* sigma_rate;
    float* theta_base;
    float* theta_scale;
    float* theta_rate;
};

struct PriorConfig {
    float rho_min, rho_max;
    float sigma_z_min, sigma_z_max;
    float mu_base_min, mu_base_max;
    float mu_scale_min, mu_scale_max;
    float mu_rate_min, mu_rate_max;
    float sigma_base_min, sigma_base_max;
    float sigma_scale_min, sigma_scale_max;
    float sigma_rate_min, sigma_rate_max;
    float theta_base_min, theta_base_max;
    float theta_scale_min, theta_scale_max;
    float theta_rate_min, theta_rate_max;
    
    float rho_prop_std;
    float sigma_z_prop_std;
    float mu_base_prop_std;
    float mu_scale_prop_std;
    float mu_rate_prop_std;
    float sigma_base_prop_std;
    float sigma_scale_prop_std;
    float sigma_rate_prop_std;
    float theta_base_prop_std;
    float theta_scale_prop_std;
    float theta_rate_prop_std;
};

__constant__ PriorConfig c_prior;

// ═══════════════════════════════════════════════════════════════════════════
// ERROR MACROS
// ═══════════════════════════════════════════════════════════════════════════

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::exit(1); \
    } \
} while (0)

#define CURAND_CHECK(call) do { \
    curandStatus_t err = (call); \
    if (err != CURAND_STATUS_SUCCESS) { \
        std::fprintf(stderr, "cuRAND error at %s:%d\n", __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while (0)

#define KERNEL_CHECK() do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "Kernel launch error at %s:%d: %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::exit(1); \
    } \
} while (0)

// ═══════════════════════════════════════════════════════════════════════════
// CUB TYPES
// ═══════════════════════════════════════════════════════════════════════════

// CUB types for particle filter operations
using BlockScanT = cub::BlockScan<float, BLOCK_SIZE>;

// Sorting frequency - CPU sorts EVERY timestep for CPMMH coupling
#ifndef SORT_EVERY_K
#define SORT_EVERY_K 7  // Match CPU: sort every step
#endif

// ═══════════════════════════════════════════════════════════════════════════
// DEVICE HELPERS
// ═══════════════════════════════════════════════════════════════════════════

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, o));
    return v;
}

__device__ float block_reduce_sum(float v, volatile float* smem, int tid) {
    v = warp_reduce_sum(v);
    if ((tid & 31) == 0) smem[tid >> 5] = v;
    __syncthreads();
    if (tid < 32) {
        v = (tid < (BLOCK_SIZE >> 5)) ? smem[tid] : 0.0f;
        v = warp_reduce_sum(v);
    }
    return v;
}

__device__ float block_reduce_max(float v, volatile float* smem, int tid) {
    v = warp_reduce_max(v);
    if ((tid & 31) == 0) smem[tid >> 5] = v;
    __syncthreads();
    if (tid < 32) {
        v = (tid < (BLOCK_SIZE >> 5)) ? smem[tid] : -1e30f;
        v = warp_reduce_max(v);
    }
    return v;
}

__device__ float ocsn_loglik(float y, float h) {
    float ymh = y - h;
    float maxv = -1e30f;
    float tmp[OCSN_K];

    #pragma unroll
    for (int k = 0; k < OCSN_K; k++) {
        float d = ymh - c_OCSN_MEANS[k];
        tmp[k] = c_OCSN_LOG_WEIGHT_NORM[k] - d * d * c_OCSN_HALF_INV_VARS[k];
        maxv = fmaxf(maxv, tmp[k]);
    }

    float s = 0.0f;
    #pragma unroll
    for (int k = 0; k < OCSN_K; k++) {
        float d = tmp[k] - maxv;
        s += (d > -87.0f) ? __expf(d) : 0.0f;
    }
    return maxv + __logf(s);
}

__device__ __forceinline__ float clamp_open01(float u) {
    return fmaxf(1e-7f, fminf(1.0f - 1e-7f, u));
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL: COMPUTE CORRELATED NOISE (prop = rho * curr + scale * fresh)
// 
// This must be called BEFORE the PF kernel.
// On accept, we swap curr <-> prop pointers (like CPU code).
// ═══════════════════════════════════════════════════════════════════════════

__global__ void compute_correlated_noise(
    const float2* __restrict__ curr_zh,
    const float* __restrict__ curr_u,           // Current resampling uniforms
    const float2* __restrict__ fresh_zh,
    const float* __restrict__ fresh_u,          // Fresh uniforms for resampling
    float2* __restrict__ prop_zh,
    float* __restrict__ prop_u,                 // Proposal resampling uniforms
    float rho, float scale,
    int64_t T, int N, int n_chains)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t TN = (int64_t)T * N;
    int64_t total = (int64_t)n_chains * TN;
    if (idx >= total) return;
    
    float2 c = curr_zh[idx];
    float2 f = fresh_zh[idx];
    float2 p;
    p.x = rho * c.x + scale * f.x;
    p.y = rho * c.y + scale * f.y;
    prop_zh[idx] = p;
}

// Separate kernel for resampling uniforms (different size)
__global__ void compute_correlated_uniform(
    const float* __restrict__ curr_u,
    const float* __restrict__ fresh_u,
    float* __restrict__ prop_u,
    float rho, float scale,
    int T, int n_chains)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_chains * T;
    if (idx >= total) return;
    
    float u = rho * curr_u[idx] + scale * fresh_u[idx];
    // Wrap to [0, 1] like CPU code
    u = u - floorf(u);
    prop_u[idx] = u;
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL: PROPOSE PARAMETERS (raw space, matching CPU)
// ═══════════════════════════════════════════════════════════════════════════

__global__ void propose_params_kernel(
    const float* __restrict__ curr_rho,
    const float* __restrict__ curr_sigma_z,
    const float* __restrict__ curr_mu_base,
    const float* __restrict__ curr_mu_scale,
    const float* __restrict__ curr_mu_rate,
    const float* __restrict__ curr_sigma_base,
    const float* __restrict__ curr_sigma_scale,
    const float* __restrict__ curr_sigma_rate,
    const float* __restrict__ curr_theta_base,
    const float* __restrict__ curr_theta_scale,
    const float* __restrict__ curr_theta_rate,
    float* __restrict__ prop_rho,
    float* __restrict__ prop_sigma_z,
    float* __restrict__ prop_mu_base,
    float* __restrict__ prop_mu_scale,
    float* __restrict__ prop_mu_rate,
    float* __restrict__ prop_sigma_base,
    float* __restrict__ prop_sigma_scale,
    float* __restrict__ prop_sigma_rate,
    float* __restrict__ prop_theta_base,
    float* __restrict__ prop_theta_scale,
    float* __restrict__ prop_theta_rate,
    const float* __restrict__ prop_noise,
    int* __restrict__ prop_valid,
    int n_chains)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_chains) return;
    
    const float* noise = &prop_noise[c * N_PARAMS];
    int valid = 1;
    
    // Raw space proposal: theta_prop = theta_curr + std * noise
    // Check bounds and reject if outside (matching CPU prior check)
    #define PROPOSE_PARAM(name, idx) \
        prop_##name[c] = curr_##name[c] + c_prior.name##_prop_std * noise[idx]; \
        if (prop_##name[c] < c_prior.name##_min || prop_##name[c] > c_prior.name##_max) valid = 0;
    
    PROPOSE_PARAM(rho, 0)
    
#if FIX_SIGMA_Z
    // Copy sigma_z unchanged (it's fixed)
    prop_sigma_z[c] = curr_sigma_z[c];
    
    PROPOSE_PARAM(mu_base, 1)
    PROPOSE_PARAM(mu_scale, 2)
    PROPOSE_PARAM(mu_rate, 3)
    PROPOSE_PARAM(sigma_base, 4)
    PROPOSE_PARAM(sigma_scale, 5)
    PROPOSE_PARAM(sigma_rate, 6)
#else
    PROPOSE_PARAM(sigma_z, 1)
    PROPOSE_PARAM(mu_base, 2)
    PROPOSE_PARAM(mu_scale, 3)
    PROPOSE_PARAM(mu_rate, 4)
    PROPOSE_PARAM(sigma_base, 5)
    PROPOSE_PARAM(sigma_scale, 6)
    PROPOSE_PARAM(sigma_rate, 7)
#endif
    
#if !FIX_THETA_CURVE
  #if FIX_SIGMA_Z
    PROPOSE_PARAM(theta_base, 7)
    PROPOSE_PARAM(theta_scale, 8)
    PROPOSE_PARAM(theta_rate, 9)
  #else
    PROPOSE_PARAM(theta_base, 8)
    PROPOSE_PARAM(theta_scale, 9)
    PROPOSE_PARAM(theta_rate, 10)
  #endif
#else
    // Copy theta params unchanged (they're fixed)
    prop_theta_base[c] = curr_theta_base[c];
    prop_theta_scale[c] = curr_theta_scale[c];
    prop_theta_rate[c] = curr_theta_rate[c];
#endif
    
    #undef PROPOSE_PARAM
    
    prop_valid[c] = valid;
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL: PARTICLE FILTER WITH PER-CHAIN PARAMETERS (v3)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * v3 changes:
 *   [A] PF init jitter at t=0 using correlated noise
 *       - Preserves CPMMH coupling
 *       - Improves likelihood estimate quality
 */
__global__ void pf_kernel_with_params(
    const float* __restrict__ obs,
    const float2* __restrict__ curr_zh,
    const float2* __restrict__ fresh_zh,
    const float* __restrict__ noise_u,
    float* __restrict__ ll_out,
    const float* __restrict__ p_rho,
    const float* __restrict__ p_sigma_z,
    const float* __restrict__ p_mu_base,
    const float* __restrict__ p_mu_scale,
    const float* __restrict__ p_mu_rate,
    const float* __restrict__ p_sigma_base,
    const float* __restrict__ p_sigma_scale,
    const float* __restrict__ p_sigma_rate,
    const float* __restrict__ p_theta_base,
    const float* __restrict__ p_theta_scale,
    const float* __restrict__ p_theta_rate,
    const int* __restrict__ prop_valid,
    float corr_rho, float corr_scale,
    int T, int N,
    float z_floor, float z_ceil,
    float z_init, float h_init,
    float z_init_jitter_std,  // v3 FIX [A]
    float h_init_jitter_std,  // v3 FIX [A]
    float logN)
{
    const int chain = blockIdx.x;
    const int tid = threadIdx.x;

    // Early-out for invalid proposals
    __shared__ int s_valid;
    if (tid == 0) {
        s_valid = prop_valid[chain];
    }
    __syncthreads();
    
    if (s_valid == 0) {
        if (tid == 0) {
            ll_out[chain] = -INFINITY;
        }
        return;
    }

    // Load chain-specific parameters
    __shared__ float s_rho, s_sigma_z;
    __shared__ float s_mu_base, s_mu_scale, s_mu_rate;
    __shared__ float s_sigma_base, s_sigma_scale, s_sigma_rate;
    __shared__ float s_theta_base, s_theta_scale, s_theta_rate;
    
    if (tid == 0) {
        s_rho = p_rho[chain];
        s_sigma_z = p_sigma_z[chain];
        s_mu_base = p_mu_base[chain];
        s_mu_scale = p_mu_scale[chain];
        s_mu_rate = p_mu_rate[chain];
        s_sigma_base = p_sigma_base[chain];
        s_sigma_scale = p_sigma_scale[chain];
        s_sigma_rate = p_sigma_rate[chain];
        s_theta_base = p_theta_base[chain];
        s_theta_scale = p_theta_scale[chain];
        s_theta_rate = p_theta_rate[chain];
    }
    __syncthreads();

    // CUB temp storage for scan
    __shared__ typename BlockScanT::TempStorage cub_scan_tmp;

    // State arrays
    __shared__ float s_z[BLOCK_SIZE];
    __shared__ float s_h[BLOCK_SIZE];
    __shared__ float s_z_tmp[BLOCK_SIZE];
    __shared__ float s_h_tmp[BLOCK_SIZE];
    __shared__ float s_cdf[BLOCK_SIZE];
    __shared__ float s_red[32];
    __shared__ int s_anc[BLOCK_SIZE];
    __shared__ float s_bcast;

    const int64_t TN = (int64_t)(T + 1) * (int64_t)N;  // Noise has T+1 slots
    const int64_t base = (int64_t)chain * TN;

    // ═══════════════════════════════════════════════════════════════════════
    // t=0: Initialize from STATIONARY DISTRIBUTION (matching CPU)
    //
    // CPU code:
    //   z = z_floor + z_stat_std * noise  (where z_stat_std = sigma_z / sqrt(1-rho^2))
    //   h = mu_z + h_stat_std * noise     (where h_stat_std = sigma_h / sqrt(1-phi^2))
    //
    // Uses t=0 noise slot for initialization.
    // ═══════════════════════════════════════════════════════════════════════
    
    // Compute stationary std for z
    float one_minus_rho_sq = 1.0f - s_rho * s_rho;
    if (one_minus_rho_sq < 1e-6f) one_minus_rho_sq = 1e-6f;
    float z_stat_std = s_sigma_z / sqrtf(one_minus_rho_sq);
    
    if (tid < N) {
        const int64_t idx0 = base + tid;  // t=0 noise slot
        const float2 c0 = curr_zh[idx0];
        const float2 f0 = fresh_zh[idx0];
        
        // Correlated noise for initialization
        const float nz0 = corr_rho * c0.x + corr_scale * f0.x;
        const float nh0 = corr_rho * c0.y + corr_scale * f0.y;
        
        // Sample z from stationary: z = z_floor + z_stat_std * noise
        float z = z_floor + z_stat_std * nz0;
        z = fmaxf(z_floor, fminf(z_ceil, z));
        s_z[tid] = z;
        
        // Compute z-dependent curves for h initialization
        float exp_mu = 1.0f - __expf(-s_mu_rate * z);
        float exp_sig = 1.0f - __expf(-s_sigma_rate * z);
        float exp_theta = 1.0f - __expf(-s_theta_rate * z);
        
        float mu_z = s_mu_base + s_mu_scale * exp_mu;
        float sigma_z_h = s_sigma_base + s_sigma_scale * exp_sig;
        float theta_z = s_theta_base + s_theta_scale * exp_theta;
        
        // Sample h from stationary given z: h = mu_z + h_stat_std * noise
        float phi = 1.0f - theta_z;
        float one_minus_phi_sq = 1.0f - phi * phi;
        if (one_minus_phi_sq < 1e-6f) one_minus_phi_sq = 1e-6f;
        float h_stat_std = sigma_z_h / sqrtf(one_minus_phi_sq);
        
        s_h[tid] = mu_z + h_stat_std * nh0;
    } else {
        s_z[tid] = z_floor;
        s_h[tid] = s_mu_base;
    }
    __syncthreads();

    float ll = 0.0f;

    for (int t = 0; t < T; t++) {
        float logw = -1e30f;

        if (tid < N) {
            // v3.1 FIX: Use t+1 noise slot for propagation (t=0 is init jitter only)
            const int64_t idx = base + (int64_t)(t + 1) * (int64_t)N + tid;
            const float2 c = curr_zh[idx];
            const float2 f = fresh_zh[idx];

            // Correlated noise
            const float nz = corr_rho * c.x + corr_scale * f.x;
            const float nh = corr_rho * c.y + corr_scale * f.y;

            // Propagate z
            float z_mean = s_rho * (s_z[tid] - z_floor) + z_floor;
            float z = fmaxf(z_floor, fminf(z_ceil, z_mean + s_sigma_z * nz));

            // Compute z-dependent curves
            float exp_mu = 1.0f - __expf(-s_mu_rate * z);
            float exp_sig = 1.0f - __expf(-s_sigma_rate * z);
            float exp_theta = 1.0f - __expf(-s_theta_rate * z);
            
            float mu_z = s_mu_base + s_mu_scale * exp_mu;
            float sigma_z_h = s_sigma_base + s_sigma_scale * exp_sig;
            float theta_z = s_theta_base + s_theta_scale * exp_theta;

            // Propagate h
            float h = (1.0f - theta_z) * s_h[tid] + theta_z * mu_z + sigma_z_h * nh;

            s_z[tid] = z;
            s_h[tid] = h;

            logw = ocsn_loglik(obs[t], h);
        }

        // Normalize weights
        float maxw = block_reduce_max(logw, s_red, tid);
        if (tid == 0) s_bcast = maxw;
        __syncthreads();
        maxw = s_bcast;

        float w = (tid < N) ? __expf(logw - maxw) : 0.0f;
        float sumw = block_reduce_sum(w, s_red, tid);

        if (tid == 0) {
            s_bcast = sumw;
            ll += logf(sumw) + maxw - logN;
        }
        __syncthreads();
        sumw = s_bcast;

        float wn = (tid < N) ? (w / sumw) : 0.0f;

        // Systematic resampling
        float cdf;
        BlockScanT(cub_scan_tmp).InclusiveSum(wn, cdf);
        __syncthreads();
        s_cdf[tid] = cdf;
        __syncthreads();

        float u0_raw = noise_u[(int64_t)chain * T + t];
        float u0 = fmaxf(1e-7f, fminf(u0_raw, 1.0f - 1e-7f));

        if (tid < N) {
            float u = (u0 + tid) / (float)N;
            int lo = 0, hi = N - 1;
            while (lo < hi) {
                int m = (lo + hi) >> 1;
                if (s_cdf[m] < u) lo = m + 1;
                else hi = m;
            }
            s_anc[tid] = lo;
        }
        __syncthreads();

        // Double-buffer resampling
        s_z_tmp[tid] = s_z[tid];
        s_h_tmp[tid] = s_h[tid];
        __syncthreads();

        if (tid < N) {
            s_z[tid] = s_z_tmp[s_anc[tid]];
            s_h[tid] = s_h_tmp[s_anc[tid]];
        }
        __syncthreads();

        // Bucket sort for CPMMH coupling (O(N) vs O(N log N) radix sort)
        // Preserves coupling quality: similar h values stay together
        if ((t % SORT_EVERY_K) == 0) {
            constexpr int BINS = 64;
            
            __shared__ int s_bin_count[BINS];
            __shared__ int s_bin_offset[BINS];
            
            // Fixed bounds covering typical h range (based on model parameters)
            const float H_MIN = -20.0f;
            const float H_MAX = 20.0f;
            const float inv_range = BINS / (H_MAX - H_MIN);
            
            // Clear counts
            if (tid < BINS) s_bin_count[tid] = 0;
            __syncthreads();
            
            // Compute bin and count
            int my_bin = 0;
            if (tid < N) {
                my_bin = min(BINS - 1, max(0, (int)((s_h[tid] - H_MIN) * inv_range)));
                atomicAdd(&s_bin_count[my_bin], 1);
            }
            __syncthreads();
            
            // Prefix sum (single thread, BINS=64 is small)
            if (tid == 0) {
                int sum = 0;
                for (int b = 0; b < BINS; b++) {
                    s_bin_offset[b] = sum;
                    sum += s_bin_count[b];
                }
            }
            __syncthreads();
            
            // Reset counts for slot claiming
            if (tid < BINS) s_bin_count[tid] = 0;
            __syncthreads();
            
            // Claim output slot within bin
            int out_idx = tid;
            if (tid < N) {
                int slot = atomicAdd(&s_bin_count[my_bin], 1);
                out_idx = s_bin_offset[my_bin] + slot;
            }
            
            // Scatter to temp buffers
            float my_z = s_z[tid];
            float my_h = s_h[tid];
            __syncthreads();
            
            s_z[out_idx] = my_z;
            s_h[out_idx] = my_h;
            __syncthreads();
        }
    }

    if (tid == 0) ll_out[chain] = ll;
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL: MH ACCEPT/REJECT
// ═══════════════════════════════════════════════════════════════════════════

__global__ void mh_accept_reject_kernel(
    const float* __restrict__ ll_prop,
    float* __restrict__ ll_curr,
    const float* __restrict__ u_mh,
    const int* __restrict__ prop_valid,
    float* __restrict__ curr_rho,
    float* __restrict__ curr_sigma_z,
    float* __restrict__ curr_mu_base,
    float* __restrict__ curr_mu_scale,
    float* __restrict__ curr_mu_rate,
    float* __restrict__ curr_sigma_base,
    float* __restrict__ curr_sigma_scale,
    float* __restrict__ curr_sigma_rate,
    float* __restrict__ curr_theta_base,
    float* __restrict__ curr_theta_scale,
    float* __restrict__ curr_theta_rate,
    const float* __restrict__ prop_rho,
    const float* __restrict__ prop_sigma_z,
    const float* __restrict__ prop_mu_base,
    const float* __restrict__ prop_mu_scale,
    const float* __restrict__ prop_mu_rate,
    const float* __restrict__ prop_sigma_base,
    const float* __restrict__ prop_sigma_scale,
    const float* __restrict__ prop_sigma_rate,
    const float* __restrict__ prop_theta_base,
    const float* __restrict__ prop_theta_scale,
    const float* __restrict__ prop_theta_rate,
    int* __restrict__ accept_count,
    int* __restrict__ accept_flag,
    float* __restrict__ param_trace,
    int iter,
    int n_chains,
    int n_iter)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_chains) return;

    int acc = 0;

    if (iter == 0) {
        ll_curr[c] = ll_prop[c];
        accept_count[c] = 1;
        acc = 1;
    } else if (prop_valid[c]) {
        float ll_p = ll_prop[c];
        float ll_c = ll_curr[c];
        
        if (isfinite(ll_p)) {
            float log_alpha = ll_p - ll_c;
            float u = clamp_open01(u_mh[c]);
            acc = (logf(u) < log_alpha) ? 1 : 0;
            if (acc) {
                ll_curr[c] = ll_p;
                accept_count[c] += 1;
            }
        }
    }

    accept_flag[c] = acc;

    if (acc) {
        curr_rho[c] = prop_rho[c];
        curr_sigma_z[c] = prop_sigma_z[c];
        curr_mu_base[c] = prop_mu_base[c];
        curr_mu_scale[c] = prop_mu_scale[c];
        curr_mu_rate[c] = prop_mu_rate[c];
        curr_sigma_base[c] = prop_sigma_base[c];
        curr_sigma_scale[c] = prop_sigma_scale[c];
        curr_sigma_rate[c] = prop_sigma_rate[c];
        curr_theta_base[c] = prop_theta_base[c];
        curr_theta_scale[c] = prop_theta_scale[c];
        curr_theta_rate[c] = prop_theta_rate[c];
    }

    // Store parameter trace - ITERATION-MAJOR for efficient partial copy
    // Layout: param_trace[iter * n_chains * N_PARAMS + c * N_PARAMS + p]
    if (param_trace) {
        int64_t row = (int64_t)iter * n_chains * N_PARAMS + c * N_PARAMS;
        param_trace[row + 0] = curr_rho[c];
#if FIX_SIGMA_Z
        param_trace[row + 1] = curr_mu_base[c];
        param_trace[row + 2] = curr_mu_scale[c];
        param_trace[row + 3] = curr_mu_rate[c];
        param_trace[row + 4] = curr_sigma_base[c];
        param_trace[row + 5] = curr_sigma_scale[c];
        param_trace[row + 6] = curr_sigma_rate[c];
  #if !FIX_THETA_CURVE
        param_trace[row + 7] = curr_theta_base[c];
        param_trace[row + 8] = curr_theta_scale[c];
        param_trace[row + 9] = curr_theta_rate[c];
  #endif
#else
        param_trace[row + 1] = curr_sigma_z[c];
        param_trace[row + 2] = curr_mu_base[c];
        param_trace[row + 3] = curr_mu_scale[c];
        param_trace[row + 4] = curr_mu_rate[c];
        param_trace[row + 5] = curr_sigma_base[c];
        param_trace[row + 6] = curr_sigma_scale[c];
        param_trace[row + 7] = curr_sigma_rate[c];
  #if !FIX_THETA_CURVE
        param_trace[row + 8] = curr_theta_base[c];
        param_trace[row + 9] = curr_theta_scale[c];
        param_trace[row + 10] = curr_theta_rate[c];
  #endif
#endif
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL: UPDATE NOISE IF ACCEPTED
// ═══════════════════════════════════════════════════════════════════════════

__global__ void update_curr_noise_if_accept_f2(
    float2* __restrict__ curr_zh,
    const float2* __restrict__ fresh_zh,
    const int* __restrict__ accept_flag,
    float rho, float scale,
    int T, int N,
    int n_chains)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t TN = (int64_t)T * N;
    int64_t total = (int64_t)n_chains * TN;
    if (idx >= total) return;

    int chain = (int)(idx / TN);
    if (accept_flag[chain] == 0) return;

    float2 c = curr_zh[idx];
    float2 f = fresh_zh[idx];
    c.x = rho * c.x + scale * f.x;
    c.y = rho * c.y + scale * f.y;
    curr_zh[idx] = c;
}

// ═══════════════════════════════════════════════════════════════════════════
// R-HAT COMPUTATION
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Compute R-hat for a single parameter
 * 
 * Layout: param_trace[iter * n_chains * N_PARAMS + c * N_PARAMS + p]
 * 
 * @param param_trace Full parameter trace buffer
 * @param n_chains    Number of chains
 * @param n_samples   Number of iterations so far
 * @param param_idx   Which parameter (0-10)
 */
float compute_rhat_single(const float* param_trace, int n_chains, int n_samples, int param_idx) {
    if (n_chains < 2 || n_samples < 4) return 999.0f;
    
    int start = (int)(n_samples * BURNIN_FRACTION);
    int n = n_samples - start;
    if (n < 4) return 999.0f;
    
    std::vector<double> chain_mean(n_chains, 0.0);
    std::vector<double> chain_var(n_chains, 0.0);
    
    // Access: param_trace[iter * n_chains * N_PARAMS + c * N_PARAMS + param_idx]
    for (int c = 0; c < n_chains; c++) {
        double sum = 0.0;
        for (int i = start; i < n_samples; i++) {
            int64_t idx = (int64_t)i * n_chains * N_PARAMS + c * N_PARAMS + param_idx;
            sum += param_trace[idx];
        }
        chain_mean[c] = sum / n;
        
        double var_sum = 0.0;
        for (int i = start; i < n_samples; i++) {
            int64_t idx = (int64_t)i * n_chains * N_PARAMS + c * N_PARAMS + param_idx;
            double diff = param_trace[idx] - chain_mean[c];
            var_sum += diff * diff;
        }
        chain_var[c] = var_sum / (n - 1);
    }
    
    double grand_mean = 0.0;
    for (int c = 0; c < n_chains; c++) grand_mean += chain_mean[c];
    grand_mean /= n_chains;
    
    double B = 0.0;
    for (int c = 0; c < n_chains; c++) {
        double diff = chain_mean[c] - grand_mean;
        B += diff * diff;
    }
    B = B * n / (n_chains - 1);
    
    double W = 0.0;
    for (int c = 0; c < n_chains; c++) W += chain_var[c];
    W /= n_chains;
    
    if (W < 1e-10) return 999.0f;
    
    double var_plus = ((n - 1.0) / n) * W + (1.0 / n) * B;
    return (float)std::sqrt(var_plus / W);
}

float compute_max_rhat(const float* param_trace, int n_chains, int n_samples, float* per_param_rhat = nullptr) {
    float max_rhat = 0.0f;
    
    for (int p = 0; p < N_PARAMS; p++) {
        float rhat = compute_rhat_single(param_trace, n_chains, n_samples, p);
        if (per_param_rhat) per_param_rhat[p] = rhat;
        max_rhat = std::max(max_rhat, rhat);
    }
    
    return max_rhat;
}

// ═══════════════════════════════════════════════════════════════════════════
// DATA GENERATION
// ═══════════════════════════════════════════════════════════════════════════

struct GroundTruth {
    float rho, sigma_z, z_floor, z_ceil;
    float mu_base, mu_scale, mu_rate;
    float sigma_base, sigma_scale, sigma_rate;
    float theta_base, theta_scale, theta_rate;
};

static GroundTruth default_ground_truth() {
    return {
        0.985f, 0.06f, 0.0f, 3.0f,
        -4.2f, 2.8f, 0.35f,
        0.07f, 0.35f, 0.25f,
        0.005f, 0.12f, 0.30f
    };
}

static const float h_OCSN_WEIGHTS[OCSN_K] = {
    0.00609f, 0.04775f, 0.13057f, 0.20674f, 0.22715f,
    0.18842f, 0.12047f, 0.05591f, 0.01575f, 0.00115f
};
static const float h_OCSN_VARS[OCSN_K] = {
    5.79596f, 2.61369f, 1.59761f, 1.19057f, 1.01301f,
    0.89951f, 0.81633f, 0.74944f, 0.69367f, 0.64009f
};

static float sample_ocsn(float h, std::mt19937& rng) {
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    
    float u = uniform(rng);
    float cumsum = 0.0f;
    int k = 0;
    for (k = 0; k < OCSN_K - 1; k++) {
        cumsum += h_OCSN_WEIGHTS[k];
        if (u < cumsum) break;
    }
    
    return h + h_OCSN_MEANS[k] + std::sqrt(h_OCSN_VARS[k]) * normal(rng);
}

static void generate_synthetic_data(int T, uint32_t seed, const GroundTruth& gt,
    std::vector<float>& y_out, std::vector<float>& z_out, std::vector<float>& h_out)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    
    y_out.resize(T);
    z_out.resize(T);
    h_out.resize(T);
    
    float z_var = gt.sigma_z * gt.sigma_z / (1.0f - gt.rho * gt.rho);
    float z = gt.z_floor + std::sqrt(z_var) * normal(rng);
    z = std::max(gt.z_floor, std::min(gt.z_ceil, z));
    
    float theta_z = gt.theta_base + gt.theta_scale * (1.0f - std::exp(-gt.theta_rate * z));
    float mu_z = gt.mu_base + gt.mu_scale * (1.0f - std::exp(-gt.mu_rate * z));
    float sigma_z_h = gt.sigma_base + gt.sigma_scale * (1.0f - std::exp(-gt.sigma_rate * z));
    float phi = 1.0f - theta_z;
    float h_var = sigma_z_h * sigma_z_h / (1.0f - phi * phi + 1e-6f);
    float h = mu_z + std::sqrt(h_var) * normal(rng);
    
    for (int t = 0; t < T; t++) {
        z_out[t] = z;
        h_out[t] = h;
        y_out[t] = sample_ocsn(h, rng);
        
        float z_new = gt.rho * z + gt.sigma_z * normal(rng);
        z = std::max(gt.z_floor, std::min(gt.z_ceil, z_new));
        
        theta_z = gt.theta_base + gt.theta_scale * (1.0f - std::exp(-gt.theta_rate * z));
        mu_z = gt.mu_base + gt.mu_scale * (1.0f - std::exp(-gt.mu_rate * z));
        sigma_z_h = gt.sigma_base + gt.sigma_scale * (1.0f - std::exp(-gt.sigma_rate * z));
        
        h = (1.0f - theta_z) * h + theta_z * mu_z + sigma_z_h * normal(rng);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════

static void init_constants() {
    CUDA_CHECK(cudaMemcpyToSymbol(c_OCSN_HALF_INV_VARS, h_OCSN_HALF_INV_VARS, sizeof(h_OCSN_HALF_INV_VARS)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_OCSN_MEANS, h_OCSN_MEANS, sizeof(h_OCSN_MEANS)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_OCSN_LOG_WEIGHT_NORM, h_OCSN_LOG_WEIGHT_NORM, sizeof(h_OCSN_LOG_WEIGHT_NORM)));
}

int main(int argc, char** argv) {
    // Match CPU test configuration for parity testing
    int n_iter = 2000;     // CPU uses 2000 iterations
    int T = 2000;
    int N = BLOCK_SIZE;    // 256 (CPU uses 300, closest power of 2)
    int n_chains = 84;      // CPU uses 4 chains
    float cpmmh_rho = 0.99f;
    uint32_t data_seed = 42;

    if (argc > 1) n_iter = std::atoi(argv[1]);
    if (argc > 2) T = std::atoi(argv[2]);
    if (argc > 3) n_chains = std::atoi(argv[3]);

    if (N != BLOCK_SIZE) {
        std::fprintf(stderr, "ERROR: N (%d) must equal BLOCK_SIZE (%d)\n", N, BLOCK_SIZE);
        return 1;
    }

    float cpmmh_scale = std::sqrt(1.0f - cpmmh_rho * cpmmh_rho);
    float logN = std::log((float)N);

    GroundTruth gt = default_ground_truth();
    
    // v3 FIX [A]: Jitter scales for PF initialization
    float z_init_jitter_std = Z_INIT_JITTER_STD;
    float h_init_jitter_std = H_INIT_JITTER_STD;
    
    // Proposal stds: ~10% of parameter range for ~25-30% acceptance
    // This is a standard heuristic - CPU likely uses similar or adaptive
    PriorConfig prior_cfg = {
        // Bounds
        0.90f, 0.999f,    // rho: width = 0.099
        0.02f, 0.15f,     // sigma_z: width = 0.13
        -6.0f, -2.0f,     // mu_base: width = 4
        1.5f, 4.5f,       // mu_scale: width = 3
        0.15f, 0.6f,      // mu_rate: width = 0.45
        0.03f, 0.15f,     // sigma_base: width = 0.12
        0.15f, 0.6f,      // sigma_scale: width = 0.45
        0.1f, 0.5f,       // sigma_rate: width = 0.4
        0.001f, 0.02f,    // theta_base: width = 0.019
        0.05f, 0.25f,     // theta_scale: width = 0.2
        0.15f, 0.5f,      // theta_rate: width = 0.35
        // Proposal stds (~10% of range, tuned for 25-30% acceptance)
        0.01f,            // rho (0.099 * 0.1)
        0.013f,           // sigma_z (0.13 * 0.1)
        0.4f,             // mu_base (4 * 0.1)
        0.3f,             // mu_scale (3 * 0.1)
        0.045f,           // mu_rate (0.45 * 0.1)
        0.012f,           // sigma_base (0.12 * 0.1)
        0.045f,           // sigma_scale (0.45 * 0.1)
        0.04f,            // sigma_rate (0.4 * 0.1)
        0.002f,           // theta_base (0.019 * 0.1)
        0.02f,            // theta_scale (0.2 * 0.1)
        0.035f            // theta_rate (0.35 * 0.1)
    };
    CUDA_CHECK(cudaMemcpyToSymbol(c_prior, &prior_cfg, sizeof(PriorConfig)));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    std::printf("\n");
    std::printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    std::printf("║   GPU CPMMH v3.1 - Parameter Learning (CPU Parity)                   ║\n");
    std::printf("╚═══════════════════════════════════════════════════════════════════════╝\n\n");
    std::printf("  GPU            : %s (%d SMs)\n", prop.name, prop.multiProcessorCount);
    std::printf("  max_iter       : %d\n", n_iter);
    std::printf("  T              : %d\n", T);
    std::printf("  N              : %d\n", N);
    std::printf("  chains         : %d (matching CPU test)\n", n_chains);
    std::printf("  CPMMH rho      : %.4f\n", cpmmh_rho);
    std::printf("  N_PARAMS       : %d\n", N_PARAMS);
    std::printf("  RHAT_THRESHOLD : %.2f\n", RHAT_THRESHOLD);
    std::printf("  SORT_EVERY_K   : %d (1 = every step, like CPU)\n", SORT_EVERY_K);
    std::printf("  RNG_BATCH_SIZE : %d\n\n", RNG_BATCH_SIZE);

    std::printf("  v3.1 fixes (CPU parity):\n");
    std::printf("    [E] Sorting: every %d step(s) (matching CPU)\n", SORT_EVERY_K);
    std::printf("    [F] Proposal stds: 10%% of param range\n");
    std::printf("    [G] Raw-space proposals with prior rejection\n");
    std::printf("    [H] Burnin fraction: %.0f%%\n\n", BURNIN_FRACTION * 100);
    
    std::printf("  v3 fixes:\n");
    std::printf("    [A] PF init from stationary distribution\n");
    std::printf("    [A'] Separate noise slots: T+1=%d\n", T + 1);
    std::printf("    [B] Debug sync interval: %d (0=disabled)\n", DEBUG_SYNC_INTERVAL);
    std::printf("    [C] Division-by-zero guard: enabled\n");
#if FIX_SIGMA_Z
    std::printf("    [D] sigma_z: FIXED to ground truth\n");
#else
    std::printf("    [D] sigma_z: LEARNED\n");
#endif
#if FIX_THETA_CURVE
    std::printf("    [E] Theta curve: FIXED to ground truth\n");
#else
    std::printf("    [E] Theta curve: LEARNED\n");
#endif
    std::printf("    N_PARAMS = %d\n\n", N_PARAMS);

    std::printf("  Ground truth:\n");
    std::printf("    rho=%.3f, sigma_z=%.3f\n", gt.rho, gt.sigma_z);
    std::printf("    mu: base=%.2f, scale=%.2f, rate=%.2f\n", gt.mu_base, gt.mu_scale, gt.mu_rate);
    std::printf("    sigma: base=%.2f, scale=%.2f, rate=%.2f\n", gt.sigma_base, gt.sigma_scale, gt.sigma_rate);
    std::printf("    theta: base=%.3f, scale=%.2f, rate=%.2f\n\n", gt.theta_base, gt.theta_scale, gt.theta_rate);

    std::vector<float> h_obs, true_z, true_h;
    generate_synthetic_data(T, data_seed, gt, h_obs, true_z, true_h);
    
    std::printf("  Generated T=%d observations\n\n", T);

    init_constants();

    // Allocate memory
    float *d_obs;
    CUDA_CHECK(cudaMalloc(&d_obs, T * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_obs, h_obs.data(), T * sizeof(float), cudaMemcpyHostToDevice));

    ChainParams d_curr, d_prop;
    
    #define ALLOC_PARAM(name) \
        CUDA_CHECK(cudaMalloc(&d_curr.name, n_chains * sizeof(float))); \
        CUDA_CHECK(cudaMalloc(&d_prop.name, n_chains * sizeof(float)));
    
    ALLOC_PARAM(rho) ALLOC_PARAM(sigma_z)
    ALLOC_PARAM(mu_base) ALLOC_PARAM(mu_scale) ALLOC_PARAM(mu_rate)
    ALLOC_PARAM(sigma_base) ALLOC_PARAM(sigma_scale) ALLOC_PARAM(sigma_rate)
    ALLOC_PARAM(theta_base) ALLOC_PARAM(theta_scale) ALLOC_PARAM(theta_rate)
    #undef ALLOC_PARAM

    std::vector<float> h_init(n_chains);
    std::mt19937 init_rng(12345);
    
    #define INIT_PARAM(name) \
        for (int c = 0; c < n_chains; c++) { \
            std::uniform_real_distribution<float> dist(prior_cfg.name##_min, prior_cfg.name##_max); \
            h_init[c] = dist(init_rng); \
        } \
        CUDA_CHECK(cudaMemcpy(d_curr.name, h_init.data(), n_chains * sizeof(float), cudaMemcpyHostToDevice));
    
    INIT_PARAM(rho)
#if FIX_SIGMA_Z
    // Fix sigma_z to ground truth
    for (int c = 0; c < n_chains; c++) h_init[c] = gt.sigma_z;
    CUDA_CHECK(cudaMemcpy(d_curr.sigma_z, h_init.data(), n_chains * sizeof(float), cudaMemcpyHostToDevice));
#else
    INIT_PARAM(sigma_z)
#endif
    INIT_PARAM(mu_base) INIT_PARAM(mu_scale) INIT_PARAM(mu_rate)
    INIT_PARAM(sigma_base) INIT_PARAM(sigma_scale) INIT_PARAM(sigma_rate)
    #undef INIT_PARAM

#if FIX_THETA_CURVE
    // Fix theta curve to ground truth (like CPU test)
    #define INIT_FIXED(name, value) \
        for (int c = 0; c < n_chains; c++) h_init[c] = value; \
        CUDA_CHECK(cudaMemcpy(d_curr.name, h_init.data(), n_chains * sizeof(float), cudaMemcpyHostToDevice));
    
    INIT_FIXED(theta_base, gt.theta_base)
    INIT_FIXED(theta_scale, gt.theta_scale)
    INIT_FIXED(theta_rate, gt.theta_rate)
    #undef INIT_FIXED
#else
    #define INIT_PARAM(name) \
        for (int c = 0; c < n_chains; c++) { \
            std::uniform_real_distribution<float> dist(prior_cfg.name##_min, prior_cfg.name##_max); \
            h_init[c] = dist(init_rng); \
        } \
        CUDA_CHECK(cudaMemcpy(d_curr.name, h_init.data(), n_chains * sizeof(float), cudaMemcpyHostToDevice));
    INIT_PARAM(theta_base) INIT_PARAM(theta_scale) INIT_PARAM(theta_rate)
    #undef INIT_PARAM
#endif

    const int64_t T_noise = T + 1;  // v3.1 FIX: Extra slot for init jitter
    const int64_t TN = (int64_t)T_noise * N;
    const int64_t total_noise = (int64_t)n_chains * TN;
    
    float2 *d_curr_zh;
    float *d_ll_prop, *d_ll_curr;
    int *d_acc_count, *d_acc_flag, *d_prop_valid;
    
    // Batched RNG buffers - allocate RNG_BATCH_SIZE iterations worth
    float2 *d_fresh_zh_batch;
    float *d_u_batch, *d_u_mh_batch, *d_prop_noise_batch;
    
    CUDA_CHECK(cudaMalloc(&d_curr_zh, total_noise * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&d_fresh_zh_batch, (int64_t)RNG_BATCH_SIZE * total_noise * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&d_u_batch, (int64_t)RNG_BATCH_SIZE * n_chains * T * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ll_prop, n_chains * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ll_curr, n_chains * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_u_mh_batch, (int64_t)RNG_BATCH_SIZE * n_chains * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_prop_noise_batch, (int64_t)RNG_BATCH_SIZE * n_chains * N_PARAMS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_acc_count, n_chains * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_acc_flag, n_chains * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_prop_valid, n_chains * sizeof(int)));
    
    CUDA_CHECK(cudaMemset(d_acc_count, 0, n_chains * sizeof(int)));

    float *d_param_trace;
    CUDA_CHECK(cudaMalloc(&d_param_trace, (int64_t)n_iter * n_chains * N_PARAMS * sizeof(float)));
    
    std::vector<float> h_param_trace((size_t)n_iter * n_chains * N_PARAMS);

    std::vector<float> h_ll_init(n_chains, -1e30f);
    CUDA_CHECK(cudaMemcpy(d_ll_curr, h_ll_init.data(), n_chains * sizeof(float), cudaMemcpyHostToDevice));

    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 12345ULL));

    CURAND_CHECK(curandGenerateNormal(gen, (float*)d_curr_zh, 2 * total_noise, 0.0f, 1.0f));

    // Main MCMC loop
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start));

    int final_iter = n_iter;
    float final_rhat = 999.0f;
    bool converged = false;

    std::printf("  Running MCMC (will stop when max R-hat < %.2f)...\n\n", RHAT_THRESHOLD);
    std::printf("  %8s  %10s  %8s  %s\n", "Iter", "Max R-hat", "AccRate", "Worst Params");
    std::printf("  ────────  ──────────  ────────  ─────────────────────────────────\n");

    float z_init = gt.z_floor;
    float h_init_val = gt.mu_base;
    
#if FIX_THETA_CURVE && FIX_SIGMA_Z
    const char* param_names[N_PARAMS] = {
        "rho", "mu_b", "mu_s", "mu_r", "sig_b", "sig_s", "sig_r"
    };
#elif FIX_THETA_CURVE
    const char* param_names[N_PARAMS] = {
        "rho", "sig_z", "mu_b", "mu_s", "mu_r", "sig_b", "sig_s", "sig_r"
    };
#elif FIX_SIGMA_Z
    const char* param_names[N_PARAMS] = {
        "rho", "mu_b", "mu_s", "mu_r", "sig_b", "sig_s", "sig_r", "th_b", "th_s", "th_r"
    };
#else
    const char* param_names[N_PARAMS] = {
        "rho", "sig_z", "mu_b", "mu_s", "mu_r",
        "sig_b", "sig_s", "sig_r", "th_b", "th_s", "th_r"
    };
#endif

    for (int it = 0; it < n_iter; it++) {
        // Generate batched RNG at start of each batch
        int batch_idx = it % RNG_BATCH_SIZE;
        if (batch_idx == 0) {
            int iters_remaining = n_iter - it;
            int batch_count = (iters_remaining < RNG_BATCH_SIZE) ? iters_remaining : RNG_BATCH_SIZE;
            CURAND_CHECK(curandGenerateNormal(gen, (float*)d_fresh_zh_batch, 
                (size_t)2 * batch_count * total_noise, 0.0f, 1.0f));
            CURAND_CHECK(curandGenerateUniform(gen, d_u_batch, 
                (size_t)batch_count * n_chains * T));
            CURAND_CHECK(curandGenerateUniform(gen, d_u_mh_batch, 
                (size_t)batch_count * n_chains));
            CURAND_CHECK(curandGenerateNormal(gen, d_prop_noise_batch, 
                (size_t)batch_count * n_chains * N_PARAMS, 0.0f, 1.0f));
        }
        
        // Pointers to this iteration's RNG data
        float2* d_fresh_zh = d_fresh_zh_batch + batch_idx * total_noise;
        float* d_u = d_u_batch + batch_idx * n_chains * T;
        float* d_u_mh = d_u_mh_batch + batch_idx * n_chains;
        float* d_prop_noise = d_prop_noise_batch + batch_idx * n_chains * N_PARAMS;

        int prop_threads = 256;
        int prop_blocks = (n_chains + prop_threads - 1) / prop_threads;
        propose_params_kernel<<<prop_blocks, prop_threads>>>(
            d_curr.rho, d_curr.sigma_z, d_curr.mu_base, d_curr.mu_scale, d_curr.mu_rate,
            d_curr.sigma_base, d_curr.sigma_scale, d_curr.sigma_rate,
            d_curr.theta_base, d_curr.theta_scale, d_curr.theta_rate,
            d_prop.rho, d_prop.sigma_z, d_prop.mu_base, d_prop.mu_scale, d_prop.mu_rate,
            d_prop.sigma_base, d_prop.sigma_scale, d_prop.sigma_rate,
            d_prop.theta_base, d_prop.theta_scale, d_prop.theta_rate,
            d_prop_noise, d_prop_valid, n_chains);
        KERNEL_CHECK();

        // v3 FIX [A]: Pass jitter stds to PF kernel
        pf_kernel_with_params<<<n_chains, BLOCK_SIZE>>>(
            d_obs, d_curr_zh, d_fresh_zh, d_u, d_ll_prop,
            d_prop.rho, d_prop.sigma_z, d_prop.mu_base, d_prop.mu_scale, d_prop.mu_rate,
            d_prop.sigma_base, d_prop.sigma_scale, d_prop.sigma_rate,
            d_prop.theta_base, d_prop.theta_scale, d_prop.theta_rate,
            d_prop_valid,
            cpmmh_rho, cpmmh_scale, T, N, gt.z_floor, gt.z_ceil,
            z_init, h_init_val,
            z_init_jitter_std, h_init_jitter_std,  // v3 FIX [A]
            logN);
        KERNEL_CHECK();

        mh_accept_reject_kernel<<<prop_blocks, prop_threads>>>(
            d_ll_prop, d_ll_curr, d_u_mh, d_prop_valid,
            d_curr.rho, d_curr.sigma_z, d_curr.mu_base, d_curr.mu_scale, d_curr.mu_rate,
            d_curr.sigma_base, d_curr.sigma_scale, d_curr.sigma_rate,
            d_curr.theta_base, d_curr.theta_scale, d_curr.theta_rate,
            d_prop.rho, d_prop.sigma_z, d_prop.mu_base, d_prop.mu_scale, d_prop.mu_rate,
            d_prop.sigma_base, d_prop.sigma_scale, d_prop.sigma_rate,
            d_prop.theta_base, d_prop.theta_scale, d_prop.theta_rate,
            d_acc_count, d_acc_flag, d_param_trace, it, n_chains, n_iter);
        KERNEL_CHECK();

        int upd_threads = 256;
        int64_t upd_total = (int64_t)n_chains * T_noise * N;  // T+1 noise slots
        int64_t upd_blocks = (upd_total + upd_threads - 1) / upd_threads;
        update_curr_noise_if_accept_f2<<<(int)upd_blocks, upd_threads>>>(
            d_curr_zh, d_fresh_zh, d_acc_flag, cpmmh_rho, cpmmh_scale, T_noise, N, n_chains);
        KERNEL_CHECK();

        // v3 FIX [B]: Periodic sync for debugging
        #if DEBUG_SYNC_INTERVAL > 0
        if ((it + 1) % DEBUG_SYNC_INTERVAL == 0) {
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        #endif

        // R-hat check
        if ((it + 1) >= MIN_ITERS_BEFORE_RHAT && 
            ((it + 1) % RHAT_CHECK_INTERVAL == 0 || it == n_iter - 1)) {
            
            // v3 FIX [B]: Sync before R-hat computation
            CUDA_CHECK(cudaDeviceSynchronize());
            
            size_t bytes = (size_t)(it + 1) * n_chains * N_PARAMS * sizeof(float);
            CUDA_CHECK(cudaMemcpy(h_param_trace.data(), d_param_trace, bytes, cudaMemcpyDeviceToHost));
            
            // Get acceptance count so far
            std::vector<int> h_acc_tmp(n_chains);
            CUDA_CHECK(cudaMemcpy(h_acc_tmp.data(), d_acc_count, n_chains * sizeof(int), cudaMemcpyDeviceToHost));
            int total_acc_tmp = 0;
            for (int c = 0; c < n_chains; c++) total_acc_tmp += h_acc_tmp[c];
            float acc_rate = (it > 0) ? (float)(total_acc_tmp - n_chains) / (float)(n_chains * it) : 0.0f;
            
            // Per-parameter R-hat
            float per_param_rhat[N_PARAMS];
            float max_rhat = compute_max_rhat(h_param_trace.data(), n_chains, it + 1, per_param_rhat);
            final_rhat = max_rhat;
            
            // Find worst 3 params
            std::vector<std::pair<float, int>> rhat_idx(N_PARAMS);
            for (int p = 0; p < N_PARAMS; p++) rhat_idx[p] = {per_param_rhat[p], p};
            std::sort(rhat_idx.rbegin(), rhat_idx.rend());
            
            char worst_str[128];
            std::snprintf(worst_str, sizeof(worst_str), "%s=%.2f %s=%.2f %s=%.2f",
                param_names[rhat_idx[0].second], rhat_idx[0].first,
                param_names[rhat_idx[1].second], rhat_idx[1].first,
                param_names[rhat_idx[2].second], rhat_idx[2].first);
            
            std::printf("  %8d  %10.4f  %7.1f%%  %s\n", it + 1, max_rhat, 100.0f * acc_rate, worst_str);
            
            if (max_rhat < RHAT_THRESHOLD) {
                final_iter = it + 1;
                converged = true;
                break;
            }
        }
    }

    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    std::vector<int> h_acc(n_chains);
    CUDA_CHECK(cudaMemcpy(h_acc.data(), d_acc_count, n_chains * sizeof(int), cudaMemcpyDeviceToHost));
    int total_accept = 0;
    for (int c = 0; c < n_chains; c++) total_accept += h_acc[c];
    
    // v3 FIX [C]: Guard division by zero
    float accept_rate = (final_iter > 1) 
        ? (float)(total_accept - n_chains) / (float)(n_chains * (final_iter - 1))
        : 0.0f;

    std::vector<float> h_final_params(n_chains);
    float post_mean[N_PARAMS] = {0};
    
#if FIX_THETA_CURVE && FIX_SIGMA_Z
    const char* param_names_full[N_PARAMS] = {
        "rho", "mu_base", "mu_scale", "mu_rate",
        "sigma_base", "sigma_scale", "sigma_rate"
    };
    float truth[N_PARAMS] = {
        gt.rho, gt.mu_base, gt.mu_scale, gt.mu_rate,
        gt.sigma_base, gt.sigma_scale, gt.sigma_rate
    };
#elif FIX_THETA_CURVE
    const char* param_names_full[N_PARAMS] = {
        "rho", "sigma_z", "mu_base", "mu_scale", "mu_rate",
        "sigma_base", "sigma_scale", "sigma_rate"
    };
    float truth[N_PARAMS] = {
        gt.rho, gt.sigma_z, gt.mu_base, gt.mu_scale, gt.mu_rate,
        gt.sigma_base, gt.sigma_scale, gt.sigma_rate
    };
#elif FIX_SIGMA_Z
    const char* param_names_full[N_PARAMS] = {
        "rho", "mu_base", "mu_scale", "mu_rate",
        "sigma_base", "sigma_scale", "sigma_rate",
        "theta_base", "theta_scale", "theta_rate"
    };
    float truth[N_PARAMS] = {
        gt.rho, gt.mu_base, gt.mu_scale, gt.mu_rate,
        gt.sigma_base, gt.sigma_scale, gt.sigma_rate,
        gt.theta_base, gt.theta_scale, gt.theta_rate
    };
#else
    const char* param_names_full[N_PARAMS] = {
        "rho", "sigma_z", "mu_base", "mu_scale", "mu_rate",
        "sigma_base", "sigma_scale", "sigma_rate",
        "theta_base", "theta_scale", "theta_rate"
    };
    float truth[N_PARAMS] = {
        gt.rho, gt.sigma_z, gt.mu_base, gt.mu_scale, gt.mu_rate,
        gt.sigma_base, gt.sigma_scale, gt.sigma_rate,
        gt.theta_base, gt.theta_scale, gt.theta_rate
    };
#endif

    #define GET_POST_MEAN(name, idx) \
        CUDA_CHECK(cudaMemcpy(h_final_params.data(), d_curr.name, n_chains * sizeof(float), cudaMemcpyDeviceToHost)); \
        for (int c = 0; c < n_chains; c++) post_mean[idx] += h_final_params[c]; \
        post_mean[idx] /= n_chains;
    
    GET_POST_MEAN(rho, 0)
#if FIX_SIGMA_Z
    GET_POST_MEAN(mu_base, 1) GET_POST_MEAN(mu_scale, 2) GET_POST_MEAN(mu_rate, 3)
    GET_POST_MEAN(sigma_base, 4) GET_POST_MEAN(sigma_scale, 5) GET_POST_MEAN(sigma_rate, 6)
  #if !FIX_THETA_CURVE
    GET_POST_MEAN(theta_base, 7) GET_POST_MEAN(theta_scale, 8) GET_POST_MEAN(theta_rate, 9)
  #endif
#else
    GET_POST_MEAN(sigma_z, 1)
    GET_POST_MEAN(mu_base, 2) GET_POST_MEAN(mu_scale, 3) GET_POST_MEAN(mu_rate, 4)
    GET_POST_MEAN(sigma_base, 5) GET_POST_MEAN(sigma_scale, 6) GET_POST_MEAN(sigma_rate, 7)
  #if !FIX_THETA_CURVE
    GET_POST_MEAN(theta_base, 8) GET_POST_MEAN(theta_scale, 9) GET_POST_MEAN(theta_rate, 10)
  #endif
#endif
    #undef GET_POST_MEAN

    std::printf("\n");
    std::printf("  ┌─────────────────────────────────────────────────────────────────┐\n");
    std::printf("  │ Results                                                         │\n");
    std::printf("  ├─────────────────────────────────────────────────────────────────┤\n");
    std::printf("  │   Converged:         %-3s                                       │\n", converged ? "YES" : "NO");
    std::printf("  │   Max R-hat:         %8.4f                                   │\n", final_rhat);
    std::printf("  │   Iterations:        %8d / %d                              │\n", final_iter, n_iter);
    std::printf("  │   Total GPU time:    %8.1f ms                                │\n", elapsed_ms);
    std::printf("  │   Time per iter:     %8.3f ms                                │\n", elapsed_ms / final_iter);
    std::printf("  │   Accept rate:       %8.1f%% (excl. init)                     │\n", 100.0f * accept_rate);
    std::printf("  └─────────────────────────────────────────────────────────────────┘\n");
    std::printf("\n");
    std::printf("  Parameter Recovery:\n");
    std::printf("  %-12s  %10s  %10s  %10s\n", "Parameter", "Truth", "Estimate", "Error%%");
    std::printf("  ──────────────────────────────────────────────────────────────────\n");
    for (int p = 0; p < N_PARAMS; p++) {
        float err = std::fabs(post_mean[p] - truth[p]) / std::fabs(truth[p]) * 100.0f;
        std::printf("  %-12s  %10.4f  %10.4f  %9.1f%%\n", param_names_full[p], truth[p], post_mean[p], err);
    }
    std::printf("\n");

    // Cleanup
    CURAND_CHECK(curandDestroyGenerator(gen));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    
    cudaFree(d_obs);
    cudaFree(d_curr_zh); cudaFree(d_fresh_zh_batch);
    cudaFree(d_u_batch); cudaFree(d_ll_prop); cudaFree(d_ll_curr);
    cudaFree(d_u_mh_batch); cudaFree(d_prop_noise_batch);
    cudaFree(d_acc_count); cudaFree(d_acc_flag); cudaFree(d_prop_valid);
    cudaFree(d_param_trace);
    
    #define FREE_PARAM(name) cudaFree(d_curr.name); cudaFree(d_prop.name);
    FREE_PARAM(rho) FREE_PARAM(sigma_z)
    FREE_PARAM(mu_base) FREE_PARAM(mu_scale) FREE_PARAM(mu_rate)
    FREE_PARAM(sigma_base) FREE_PARAM(sigma_scale) FREE_PARAM(sigma_rate)
    FREE_PARAM(theta_base) FREE_PARAM(theta_scale) FREE_PARAM(theta_rate)
    #undef FREE_PARAM

    return 0;
}
