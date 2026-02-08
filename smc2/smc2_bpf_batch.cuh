/**
 * @file smc2_bpf_batch.cuh
 * @brief SMC² with BPF Inner Filter for SV Parameter Learning
 *
 * Learns θ = (ρ, σ_z, μ) for the stochastic volatility model:
 *
 *   h_t = μ + ρ·(h_{t-1} − μ) + σ_z·ε_t      ε_t ~ N(0,1)
 *   y_t = exp(h_t / 2) · η_t                    η_t ~ Student-t(ν)
 *
 * The inner filter is a Bootstrap Particle Filter with Student-t observation
 * likelihood — the SAME model as the production BPF worker. This ensures
 * learned parameters are optimal for the actual filter, not a surrogate.
 *
 * Architecture:
 *   Outer layer:  N_theta parameter particles with SMC² weighting
 *   Inner layer:  N_inner BPF particles per θ (state: h only)
 *   Rejuvenation: CPMMH with correlated noise for low-variance MH ratios
 *   Proposals:    Adaptive Haario with Cholesky factor (3×3)
 *   Checkpoints:  Fixed-lag for O(L) replay instead of O(T)
 *
 * Designed as a batch learner: processes windows of ~1000 ticks,
 * then injects learned parameters into the production BPF.
 */

#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdint.h>

/*═══════════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════════*/

#define SMC2_N_PARAMS 3   /* ρ, σ_z, μ   (ν fixed externally) */

#ifndef SMC2_SORT_EVERY_K
#define SMC2_SORT_EVERY_K 4
#endif

/*═══════════════════════════════════════════════════════════════════════════════
 * STRUCTURES
 *═══════════════════════════════════════════════════════════════════════════════*/

/** Prior: independent Gaussians (truncated by bounds) */
typedef struct {
    float mean[SMC2_N_PARAMS];  /* [ρ, σ_z, μ] */
    float std[SMC2_N_PARAMS];
} SVPrior3;

/** Hard bounds for rejection / proposal clamping */
typedef struct {
    float lo[SMC2_N_PARAMS];
    float hi[SMC2_N_PARAMS];
} SVBounds3;

/** SoA layout for θ-particles + inner BPF state */
typedef struct {
    /* Per θ-particle [N_theta] */
    float* rho;
    float* sigma_z;
    float* mu;
    float* log_weight;
    float* weight;
    float* log_likelihood;
    float* ess_inner;

    /* Per inner particle [N_theta × N_inner] */
    float* inner_h;
    float* inner_log_w;
    curandState* rng_states;
} ThetaSoA;

/** Full algorithm state */
typedef struct {
    int N_theta;
    int N_inner;

    ThetaSoA d_particles;
    ThetaSoA d_particles_temp;   /* Ping-pong for resampling + CPMMH scratch */

    /* Observation history (device) */
    float* d_y_history;
    int    y_history_len;
    int    y_history_capacity;
    int    t_current;

    /* Scratch arrays */
    int*   d_ancestors;
    float* d_uniform;
    float* d_ess;
    int*   d_accepts;
    int*   d_swap_flags;

    /* CPMMH noise buffers [2] — ping-pong */
    float* d_h_noise[2];     /* [N_theta × (capacity+1) × N_inner] */
    float* d_u0_noise[2];    /* [N_theta × (capacity+1)] */
    int    noise_buf;         /* Current buffer index (0 or 1) */
    int    noise_capacity;    /* Max timesteps before realloc */
    float  cpmmh_rho;         /* Noise correlation ρ ∈ [0.95, 0.999] */

    /* Fixed-lag checkpoint */
    int    fixed_lag_L;
    int    t_checkpoint;
    float* d_checkpoint_h;
    float* d_checkpoint_log_w;
    float* d_checkpoint_ll;

    /* Model */
    SVPrior3  prior;
    SVBounds3 bounds;
    float     nu_obs;         /* Student-t df (fixed, shared with prod BPF) */

    /* Proposals */
    float proposal_std[SMC2_N_PARAMS];
    float proposal_chol[SMC2_N_PARAMS * SMC2_N_PARAMS];  /* 3×3 lower tri */
    bool  use_adaptive_proposals;

    /* Adaptive proposal scratch (device) */
    float* d_temp_mean;       /* [3] */
    float* d_temp_cov;        /* [9] */

    /* Tuning */
    float ess_threshold_outer;
    int   K_rejuv;

    /* Diagnostics */
    int n_resamples;
    int n_rejuv_accepts;
    int n_rejuv_total;

    /* Host RNG */
    uint64_t host_rng_state;
    uint64_t user_seed;
} SMC2BPFState;

/*═══════════════════════════════════════════════════════════════════════════════
 * DEVICE HELPERS — Warp-Shuffle Reductions
 *
 * Same hybrid strategy as production BPF PTX:
 *   Inter-warp: shared memory + __syncthreads
 *   Intra-warp: __shfl_xor_sync (no shared memory, no barrier)
 *═══════════════════════════════════════════════════════════════════════════════*/

__device__ __forceinline__ float warp_reduce_sum(float v) {
    v += __shfl_xor_sync(0xFFFFFFFF, v, 16);
    v += __shfl_xor_sync(0xFFFFFFFF, v, 8);
    v += __shfl_xor_sync(0xFFFFFFFF, v, 4);
    v += __shfl_xor_sync(0xFFFFFFFF, v, 2);
    v += __shfl_xor_sync(0xFFFFFFFF, v, 1);
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, 16));
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, 8));
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, 4));
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, 2));
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, 1));
    return v;
}

/**
 * Block-level sum reduction.  smem must have at least 32 floats.
 * Returns result in thread 0 only. Caller broadcasts via shared if needed.
 */
__device__ inline float block_reduce_sum(float val, float* smem) {
    int tid  = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    float ws = warp_reduce_sum(val);
    if (lane == 0) smem[warp] = ws;
    __syncthreads();

    int nw = (blockDim.x + 31) >> 5;
    float v = (tid < nw) ? smem[tid] : 0.0f;
    if (warp == 0) v = warp_reduce_sum(v);
    return v;   /* valid in tid 0 */
}

/**
 * Block-level max reduction.  smem must have at least 32 floats.
 */
__device__ inline float block_reduce_max(float val, float* smem) {
    int tid  = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    float wm = warp_reduce_max(val);
    if (lane == 0) smem[warp] = wm;
    __syncthreads();

    int nw = (blockDim.x + 31) >> 5;
    float v = (tid < nw) ? smem[tid] : -INFINITY;
    if (warp == 0) v = warp_reduce_max(v);
    return v;
}

/**
 * In-place inclusive scan (serial, thread 0).
 * Fine for N ≤ 1024 — runs once per resample.
 */
__device__ inline void block_inclusive_scan(float* data, int N) {
    if (threadIdx.x != 0) return;
    for (int i = 1; i < N; i++) data[i] += data[i - 1];
}

/*═══════════════════════════════════════════════════════════════════════════════
 * DEVICE HELPERS — Student-t Log-Likelihood
 *
 * log p(y | h, ν) = C(ν) − h/2 − ((ν+1)/2) · log(1 + y²·exp(−h)/ν)
 *
 * where C(ν) = lgamma((ν+1)/2) − lgamma(ν/2) − 0.5·log(νπ)
 *═══════════════════════════════════════════════════════════════════════════════*/

__device__ __forceinline__ float student_t_C(float nu) {
    return lgammaf(0.5f * (nu + 1.0f))
         - lgammaf(0.5f * nu)
         - 0.5f * __logf(nu * 3.14159265f);
}

__device__ __forceinline__ float student_t_log_lik(float y, float h, float nu, float C) {
    float half_nu_p1 = 0.5f * (nu + 1.0f);
    float exp_neg_h  = __expf(-h);
    float arg = 1.0f + (y * y * exp_neg_h) / nu;
    return C - 0.5f * h - half_nu_p1 * __logf(arg);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * DEVICE HELPERS — Noise Uniform from Gaussian
 *
 * Map stored Gaussian noise to U(0,1) for resampling via Φ(z).
 * erff-based CDF for standard normal.
 *═══════════════════════════════════════════════════════════════════════════════*/

__device__ __forceinline__ float u0_from_noise(float z) {
    float u = 0.5f * (1.0f + erff(z * 0.7071067812f));
    return fminf(fmaxf(u, 1e-7f), 1.0f - 1e-7f);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * SHARED MEMORY SIZING
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * Shared memory for inner BPF step kernel.
 * Layout: s_reduction[32] | s_h[N] | s_cdf[N]
 */
template<int N>
__host__ __device__ constexpr int bpf_step_smem() {
    return (32 + 2 * N) * sizeof(float);
}

/**
 * Shared memory for CPMMH rejuvenation kernel.
 * Layout: s_reduction[32] | s_h[N] | s_cdf[N] | CUB_temp
 * CUB BlockRadixSort<float, N, 1>::TempStorage is ≤ 4*N bytes typically.
 */
template<int N>
__host__ __device__ constexpr int cpmmh_smem() {
    return (32 + 2 * N) * sizeof(float) + 4 * N;  /* Conservative CUB estimate */
}

/*═══════════════════════════════════════════════════════════════════════════════
 * HOST API
 *═══════════════════════════════════════════════════════════════════════════════*/

#ifdef __cplusplus
extern "C" {
#endif

/** Allocate and return state. N_inner must be 64, 128, 256, or 512. */
SMC2BPFState* smc2_bpf_alloc(int N_theta, int N_inner);

/** Free all GPU memory and state. */
void smc2_bpf_free(SMC2BPFState* s);

/** Initialize particles from prior. Call after configuring prior/bounds/nu. */
void smc2_bpf_init(SMC2BPFState* s);

/**
 * Process one observation. Returns outer ESS.
 * Internally triggers resample + CPMMH rejuvenation when ESS drops.
 */
float smc2_bpf_update(SMC2BPFState* s, float y_obs);

/**
 * Process a batch window of observations.
 * Resets state, runs all T observations, returns final outer ESS.
 */
float smc2_bpf_learn_window(SMC2BPFState* s, const float* y, int T);

/** Extract weighted mean of θ = (ρ, σ_z, μ). */
void smc2_bpf_get_theta_mean(SMC2BPFState* s, float out[SMC2_N_PARAMS]);

/** Extract weighted std of θ. */
void smc2_bpf_get_theta_std(SMC2BPFState* s, float out[SMC2_N_PARAMS]);

/** Get current outer ESS. */
float smc2_bpf_get_outer_ess(SMC2BPFState* s);

/* Configuration setters */
void smc2_bpf_set_seed(SMC2BPFState* s, uint64_t seed);
void smc2_bpf_set_nu_obs(SMC2BPFState* s, float nu);
void smc2_bpf_set_cpmmh_rho(SMC2BPFState* s, float rho);
void smc2_bpf_set_fixed_lag(SMC2BPFState* s, int L);
void smc2_bpf_set_proposal_std(SMC2BPFState* s, const float std[SMC2_N_PARAMS]);

#ifdef __cplusplus
}
#endif
