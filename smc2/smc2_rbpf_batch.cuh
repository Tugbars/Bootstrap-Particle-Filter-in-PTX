/**
 * @file smc2_rbpf_cuda.cuh
 * @brief SMC² with RBPF Inner Filter and CPMMH Rejuvenation - CUDA Implementation
 * 
 * @author TUGBARS
 * @date 2025
 * 
 * ═══════════════════════════════════════════════════════════════════════════════
 * ALGORITHM OVERVIEW
 * ═══════════════════════════════════════════════════════════════════════════════
 * 
 * This implements SMC² (Chopin et al. 2013) for online Bayesian parameter
 * learning in a regime-switching stochastic volatility model.
 * 
 * Three-Level Structure:
 * ----------------------
 * 
 *   ┌─────────────────────────────────────────────────────────────────┐
 *   │ OUTER: SMC² over θ-particles (N_theta = 256)                │
 *   │   - Learned parameters: ρ, σ_z, μ_base, σ_base              │
 *   │   - Fixed curve shapes: μ_scale, μ_rate, σ_scale, σ_rate    │
 *   │   - Weights: accumulated likelihood p̂(y_{1:t} | θ)          │
 *   │   - Resample when ESS < threshold                           │
 *   │   - Rejuvenate via CPMMH moves                              │
 *   └─────────────────────────────────────────────────────────────────┘
 *                              │
 *                              ▼
 *   ┌─────────────────────────────────────────────────────────────────┐
 *   │ INNER: RBPF over (z, h) state (N_inner = 256 per θ)         │
 *   │   - Regime z̃: particle approximation (N_inner samples)      │
 *   │   - Log-vol h: Rao-Blackwellized (analytic Kalman moments)  │
 *   │   - OCSN 10-component mixture for observation likelihood    │
 *   └─────────────────────────────────────────────────────────────────┘
 *                              │
 *                              ▼
 *   ┌─────────────────────────────────────────────────────────────────┐
 *   │ CPMMH: Correlated Pseudo-Marginal MH for rejuvenation       │
 *   │   - Correlates noise: z' = ρ·z + √(1-ρ²)·ε  (ρ ≈ 0.99)      │
 *   │   - Bucket sort after resampling preserves coupling         │
 *   │   - Full-history replay for correct MH ratio                │
 *   └─────────────────────────────────────────────────────────────────┘
 * 
 * 
 * Parameter Space (4D learned + 4D fixed)
 * ========================================
 * 
 * Learned by SMC²:
 *   ρ        - AR(1) persistence of z̃ (dynamics)
 *   σ_z      - Innovation scale of z̃ (dynamics)
 *   μ_base   - Base mean level (curve level)
 *   σ_base   - Base vol-of-vol (curve level)
 * 
 * Fixed (calibrated offline):
 *   μ_scale, μ_rate     - Mean curve shape
 *   σ_scale, σ_rate     - Vol-of-vol curve shape
 * 
 * Rationale: curve LEVELS need to adapt online, but curve SHAPES are
 * slow-moving and can be calibrated offline. This halves the proposal
 * dimension from 8D to 4D, roughly doubling MH acceptance rates.
 * 
 * 
 * CPMMH Coupling: Why We Sort After Resampling
 * =============================================
 * 
 * Standard PMMH has high variance because p̂(y|θ) is noisy. CPMMH reduces
 * variance by correlating the random numbers between current and proposed:
 * 
 *     z_prop[i] = ρ · z_curr[i] + √(1-ρ²) · z_fresh[i]
 * 
 * With ρ = 0.99, proposed and current filters see nearly identical noise.
 * BUT this only helps if particle[i] represents the "same" state in both runs.
 * 
 * Problem: Resampling scrambles particle identities.
 * Solution: Sort particles by μ_h after resampling.
 * 
 * 
 * OCSN Mixture Approximation
 * ==========================
 * 
 * The SV observation equation is:
 *     y_t = exp(h_t/2) · ε_t,  ε_t ~ N(0,1)
 * Taking logs: log(y_t²) = h_t + log(χ²(1))
 * 
 * Omori, Chib, Shephard & Nakajima (2007) approximate log(χ²(1)) as a
 * 10-component Gaussian mixture, restoring approximate Kalman tractability.
 * 
 * 
 * Z-Space Transform
 * =================
 * 
 *     z̃ ∈ ℝ              (unconstrained, exact Gaussian AR(1))
 *     z = 1.5·(1 + tanh(z̃)) ∈ (0, 3)   (bounded, for curves)
 * 
 * 
 * References
 * ==========
 * 
 * [1] Chopin, Jacob, Papaspiliopoulos (2013). "SMC²." JRSS-B.
 * [2] Andrieu, Doucet, Holenstein (2010). "Particle MCMC." JRSS-B.
 * [3] Deligiannidis, Doucet, Pitt (2018). "Correlated Pseudo-Marginal." JRSS-B.
 * [4] Omori, Chib, Shephard, Nakajima (2007). "SV with Leverage." J. Econometrics.
 * 
 * ═══════════════════════════════════════════════════════════════════════════════
 */

#ifndef SMC2_RBPF_CUDA_CUH
#define SMC2_RBPF_CUDA_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <stdint.h>

#include "smc2_noise_precision.cuh"

/*═══════════════════════════════════════════════════════════════════════════════
 * SECTION 1: COMPILE-TIME CONFIGURATION
 *═══════════════════════════════════════════════════════════════════════════════*/

#define OCSN_K 10

/** Number of learned parameters in the SMC² outer layer */
#define N_PARAMS 4

#ifndef SORT_EVERY_K
#define SORT_EVERY_K  4
#endif

#define Z_CENTER 1.5f
#define Z_SCALE  1.5f

/*═══════════════════════════════════════════════════════════════════════════════
 * SECTION 2: DATA STRUCTURES
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Gaussian prior specification for learned θ parameters (4D)
 * 
 * Each parameter has independent N(mean, std²) prior.
 * Used in MH acceptance ratio: π(θ*)/π(θ).
 */
struct SVPrior {
    float rho_mean, rho_std;
    float sigma_z_mean, sigma_z_std;
    float mu_base_mean, mu_base_std;
    float sigma_base_mean, sigma_base_std;
};

/**
 * @brief Hard bounds for learned parameter support
 * 
 * Parameters outside bounds → log_prior = -∞ (instant rejection).
 */
struct SVBounds {
    float rho_min, rho_max;
    float sigma_z_min, sigma_z_max;
    float mu_base_min, mu_base_max;
    float sigma_base_min, sigma_base_max;
};

/**
 * @brief Fixed curve shape parameters (calibrated offline, not learned)
 * 
 * These define how μ(z) and σ_h(z) curves respond to regime z.
 * The BASE levels are learned; the SHAPES are fixed.
 */
struct SVFixedCurves {
    float mu_scale;       /**< μ(z) curve: scale */
    float mu_rate;        /**< μ(z) curve: rate */
    float sigma_scale;    /**< σ_h(z) curve: scale */
    float sigma_rate;     /**< σ_h(z) curve: rate */
};

/**
 * @brief Regime-dependent curve: f(z) = base + scale * (1 - exp(-rate * z))
 */
struct SVCurve {
    float base;
    float scale;
    float rate;
};

/**
 * @brief θ-particle population with embedded RBPF state (SoA layout)
 * 
 * Memory layout uses Structure-of-Arrays for coalesced GPU access.
 * 
 * Only 4 parameters are learned (rho, sigma_z, mu_base, sigma_base).
 * Curve shapes (mu_scale, mu_rate, sigma_scale, sigma_rate) live in
 * constant memory as SVFixedCurves.
 */
struct ThetaParticlesSoA {
    /* ═══ Learned θ-level arrays (N_theta elements) ═══ */
    float* rho;              /**< AR(1) coefficient for z̃ dynamics */
    float* sigma_z;          /**< Innovation std for z̃ */
    float* mu_base;          /**< μ(z) curve: base parameter (learned) */
    float* sigma_base;       /**< σ_h(z) curve: base parameter (learned) */
    
    float* log_weight;
    float* weight;
    float* log_likelihood;
    float* ess_inner;
    
    /* ═══ Inner RBPF arrays (N_theta × N_inner elements) ═══ */
    float* inner_z;
    float* inner_mu_h;
    float* inner_var_h;
    float* inner_log_w;
    curandState* rng_states;
};

/**
 * @brief Complete SMC² state container
 */
struct SMC2StateCUDA {
    /* ═══ Dimensions ═══ */
    int N_theta;
    int N_inner;
    
    /* ═══ Particle storage (double-buffered for resampling) ═══ */
    ThetaParticlesSoA d_particles;
    ThetaParticlesSoA d_particles_temp;
    
    /* ═══ Observation history ═══ */
    float* d_y_history;
    int y_history_len;
    int y_history_capacity;
    int t_current;
    
    /* ═══ CPMMH noise buffers ═══ */
    noise_t* d_z_noise[2];
    noise_t* d_u0_noise[2];
    int noise_buf;
    int noise_capacity;
    float cpmmh_rho;
    
    /* ═══ Scratch arrays ═══ */
    int* d_ancestors;
    float* d_uniform;
    float* d_ess;
    int* d_accepts;
    int* d_swap_flags;
    
    /* ═══ Model specification ═══ */
    SVPrior prior;
    SVBounds bounds;
    SVCurve theta_curve;             /**< θ(z) curve (fixed, not learned) */
    SVFixedCurves fixed_curves;      /**< Fixed curve shapes for μ and σ_h */
    float proposal_std[N_PARAMS];    /**< Random walk proposal std */
    
    /* ═══ Algorithm settings ═══ */
    float ess_threshold_outer;
    float ess_threshold_inner;
    int K_rejuv;
    
    /* ═══ Fixed-Lag PMMH ═══ */
    int fixed_lag_L;
    int t_checkpoint;
    float* d_checkpoint_z;
    float* d_checkpoint_mu_h;
    float* d_checkpoint_var_h;
    float* d_checkpoint_log_w;
    float* d_checkpoint_ll;
    
    /* ═══ Diagnostics ═══ */
    int n_resamples;
    int n_rejuv_accepts;
    int n_rejuv_total;
    
    /* ═══ Adaptive Proposals ═══ */
    float* d_temp_mean;                        /**< Scratch: particle mean [N_PARAMS] */
    float* d_temp_cov;                         /**< Scratch: particle covariance [N_PARAMS²] */
    bool use_adaptive_proposals;
    
    /* ═══ RNG ═══ */
    uint64_t user_seed;
    uint64_t host_rng_state;
};

/*═══════════════════════════════════════════════════════════════════════════════
 * SECTION 3-4: DEVICE HELPER FUNCTIONS
 *═══════════════════════════════════════════════════════════════════════════════*/

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ __forceinline__ 
float block_reduce_sum(float val, volatile float* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    int numWarps = (blockDim.x + 31) >> 5;
    
    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < numWarps) ? shared[threadIdx.x] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    
    if (threadIdx.x == 0) shared[0] = val;
    __syncthreads();
    return shared[0];
}

__device__ __forceinline__
float block_reduce_max(float val, volatile float* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    int numWarps = (blockDim.x + 31) >> 5;
    
    val = warp_reduce_max(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < numWarps) ? shared[threadIdx.x] : -1e30f;
    if (wid == 0) val = warp_reduce_max(val);
    
    if (threadIdx.x == 0) shared[0] = val;
    __syncthreads();
    return shared[0];
}

__device__ __forceinline__
void block_inclusive_scan(volatile float* data, int n) {
    int tid = threadIdx.x;
    
    for (int offset = 1; offset < n; offset *= 2) {
        float temp = 0.0f;
        if (tid >= offset && tid < n) {
            temp = data[tid - offset];
        }
        __syncthreads();
        
        if (tid >= offset && tid < n) {
            data[tid] += temp;
        }
        __syncthreads();
    }
}

__device__ __forceinline__
float eval_curve(float base, float scale, float rate, float z) {
    return base + scale * (1.0f - __expf(-rate * z));
}

__device__ __forceinline__ float z_tilde_to_z(float z_tilde) {
    return Z_CENTER * (1.0f + tanhf(z_tilde));
}

__device__ __forceinline__ float z_to_z_tilde(float z) {
    float normalized = (z - Z_CENTER) / Z_SCALE;
    normalized = fmaxf(-0.999f, fminf(0.999f, normalized));
    return atanhf(normalized);
}

__device__ __forceinline__ float u0_from_noise(float z_noise) {
    float u = normcdff(z_noise);
    return fmaxf(1e-7f, fminf(1.0f - 1e-7f, u));
}

#include "smc2_sorting.cuh"

/*═══════════════════════════════════════════════════════════════════════════════
 * SECTION 5: OCSN CONSTANTS
 *═══════════════════════════════════════════════════════════════════════════════*/

/* OCSN constant arrays and ocsn_kalman_update() defined in .cu file only.
 * Not declared extern here — MSVC/NVCC treats extern __constant__ as
 * a static definition, causing redefinition errors when the .cu provides
 * the actual definitions with initializers.
 * ocsn_kalman_update lives in .cu where the constants are visible. */

/**
 * @brief OCSN Kalman update — moment-matching over 10-component mixture
 * 
 * Defined in smc2_rbpf_cuda.cu alongside OCSN constant arrays.
 */
__device__ void ocsn_kalman_update(
    float y, float mu_pred, float var_pred,
    float* mu_post, float* var_post, float* log_lik
);

/*═══════════════════════════════════════════════════════════════════════════════
 * SECTION 6: KERNEL DECLARATIONS
 *═══════════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_init_rng(curandState* states, unsigned long long seed, int N);

__global__ void kernel_init_from_prior(
    ThetaParticlesSoA particles,
    int N_theta, int N_inner,
    noise_t* d_z_noise, noise_t* d_u0_noise,
    int noise_capacity
);

__global__ void kernel_rbpf_step(
    ThetaParticlesSoA particles,
    float y_obs,
    int N_theta, int N_inner,
    noise_t* d_z_noise, noise_t* d_u0_noise,
    int t_current, int noise_capacity
);

__global__ void kernel_compute_outer_ess(
    ThetaParticlesSoA particles, float* d_ess_out, int N_theta
);

__global__ void kernel_outer_resample(
    ThetaParticlesSoA particles, int* d_ancestors, float* d_uniform, int N_theta
);

__global__ void kernel_copy_theta_particles(
    ThetaParticlesSoA src, ThetaParticlesSoA dst, int* d_ancestors,
    int N_theta, int N_inner, unsigned long long resample_seed
);

__global__ void kernel_copy_noise_arrays(
    const noise_t* src_z, noise_t* dst_z,
    const noise_t* src_u0, noise_t* dst_u0,
    const int* d_ancestors, int N_theta, int N_inner,
    int t_current, int noise_capacity
);

__global__ void kernel_cpmmh_rejuvenate_fused(
    ThetaParticlesSoA particles, ThetaParticlesSoA particles_scratch,
    const float* y_history,
    noise_t* d_z_noise_curr, noise_t* d_z_noise_other,
    noise_t* d_u0_noise_curr, noise_t* d_u0_noise_other,
    int t_current, int N_theta, int N_inner,
    int noise_capacity, float cpmmh_rho,
    int* d_accepts, int* d_swap_flags,
    unsigned long long seed, int move_id, int block_id
);

__global__ void kernel_commit_accepted_noise(
    noise_t* d_z_noise_0, noise_t* d_z_noise_1,
    noise_t* d_u0_noise_0, noise_t* d_u0_noise_1,
    const int* d_swap_flags, int N_theta, int N_inner,
    int t_current, int noise_capacity, int t_start
);

/*═══════════════════════════════════════════════════════════════════════════════
 * SECTION 7: HOST API
 *═══════════════════════════════════════════════════════════════════════════════*/

#ifdef __cplusplus
extern "C" {
#endif

SMC2StateCUDA* smc2_cuda_alloc(int N_theta, int N_inner);
void smc2_cuda_free(SMC2StateCUDA* state);
void smc2_cuda_set_seed(SMC2StateCUDA* state, uint64_t seed);
void smc2_cuda_set_noise_capacity(SMC2StateCUDA* state, int capacity);
void smc2_cuda_set_fixed_lag(SMC2StateCUDA* state, int L);
void smc2_cuda_set_cpmmh_rho(SMC2StateCUDA* state, float rho);

/**
 * @brief Set proposal standard deviations
 * @param std  Array of N_PARAMS floats [rho, sigma_z, mu_base, sigma_base], or NULL for defaults
 */
void smc2_cuda_set_proposal_std(SMC2StateCUDA* state, const float* std);

void smc2_cuda_init_from_prior(SMC2StateCUDA* state);
float smc2_cuda_update(SMC2StateCUDA* state, float y_obs);

/**
 * @brief Get posterior mean of learned θ parameters
 * @param theta_mean  Output array of size N_PARAMS
 * Order: [rho, sigma_z, mu_base, sigma_base]
 */
void smc2_cuda_get_theta_mean(SMC2StateCUDA* state, float* theta_mean);
void smc2_cuda_get_theta_std(SMC2StateCUDA* state, float* theta_std);
float smc2_cuda_get_outer_ess(SMC2StateCUDA* state);

#ifdef __cplusplus
}
#endif

#endif /* SMC2_RBPF_CUDA_CUH */
