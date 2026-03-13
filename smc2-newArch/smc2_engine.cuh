/**
 * @file smc2_engine.cuh
 * @brief SMC² with RBPF Inner Filter — Unified Header
 *
 * Single header for the entire SMC² subsystem. Organized in dependency order:
 *
 *   §1  Compile-Time Constants
 *   §2  Noise Precision Layer     (FP16/FP32 abstraction for CPMMH noise)
 *   §3  Model Types               (SVPrior, SVBounds, SVCurve)
 *   §4  Default Configurations    (named functions, not buried in alloc())
 *   §5  Algorithm Types           (ThetaParticlesSoA, SMC2StateCUDA, SMC2Diagnostics)
 *   §6  Device Utilities          (reductions, scan, curve eval, z-transform)
 *   §7  OCSN Kalman Update        (declaration — defined in .cu with constants)
 *   §8  Sorting Backend           (bitonic or CUB, compile-time selectable)
 *   §9  Shared Memory Helpers
 *   §10 Host API Declarations
 *
 * OCSN Parameterization Note:
 *   This file uses the Omori et al. (2007) centered parameterization where
 *   y_t = log(r_t^2) and mixture means are mostly negative. The companion
 *   CPMMH code (cpmmh_gpu_learn_v3.cu) uses Kim et al. (1998) with
 *   y_t = log(r_t^2) - E[log(chi^2_1)] and different mean/variance values.
 *   Ensure observation preprocessing matches the parameterization used.
 *
 * Limitations:
 *   - N_theta must be <= 8192 (shared memory for outer resample CDF)
 *   - N_inner must be one of {64, 128, 256, 512}
 *
 * References:
 *   [1] Chopin, Jacob, Papaspiliopoulos (2013). "SMC²." JRSS-B.
 *   [2] Andrieu, Doucet, Holenstein (2010). "Particle MCMC." JRSS-B.
 *   [3] Deligiannidis, Doucet, Pitt (2018). "Correlated Pseudo-Marginal." JRSS-B.
 *   [4] Omori, Chib, Shephard, Nakajima (2007). "SV with Leverage." J. Econometrics.
 */

#ifndef SMC2_ENGINE_CUH
#define SMC2_ENGINE_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <stdint.h>


/*═══════════════════════════════════════════════════════════════════════════════
 * §1: COMPILE-TIME CONSTANTS
 *═══════════════════════════════════════════════════════════════════════════════*/

/** OCSN 10-component mixture approximation to log(χ²(1)) */
#define OCSN_K 10

/** Number of learned parameters in the SMC² outer layer.
 *  8D: rho, sigma_total, r_split, mu_base, mu_scale, mu_rate, sigma_scale, sigma_rate */
#define N_PARAMS 8

/** Sort particles every K steps for CPMMH coupling preservation */
#ifndef SORT_EVERY_K
#define SORT_EVERY_K 4
#endif

/** Z-space transform: z = Z_SCALE * (1 + tanh(z̃)) maps ℝ → (0, 2*Z_SCALE) */
#define Z_CENTER 1.5f
#define Z_SCALE  1.5f

/** Maximum N_theta — limited by shared memory for outer resample CDF */
#define MAX_N_THETA 8192

/** Maximum noise capacity to prevent OOM (128k timesteps) */
#define MAX_NOISE_CAPACITY 131072


/*═══════════════════════════════════════════════════════════════════════════════
 * §2: NOISE PRECISION LAYER
 *
 * Unified interface for FP16/FP32 noise storage with compile-time selection.
 *   - Default (FP32): Full precision, recommended for accuracy
 *   - Bandwidth-optimized (FP16): Define SMC2_NOISE_FP16 before including
 *     ~50% memory bandwidth, ~0.1% relative error per step.
 *     Use only when bandwidth is critical and T < 500.
 *
 * Build:
 *   nvcc -o smc2 smc2_engine.cu                          # FP32 (default)
 *   nvcc -DSMC2_NOISE_FP16 -o smc2_fp16 smc2_engine.cu   # FP16
 *═══════════════════════════════════════════════════════════════════════════════*/

#ifdef SMC2_NOISE_FP16

/* ── FP16 Implementation (bandwidth-optimized, opt-in) ── */

typedef half noise_t;
#define NOISE_SIZEOF sizeof(half)

__device__ __forceinline__ float noise_load(const noise_t* ptr, int64_t idx) {
    return __half2float(ptr[idx]);
}

__device__ __forceinline__ void noise_store(noise_t* ptr, int64_t idx, float val) {
    ptr[idx] = __float2half(val);
}

/**
 * @brief Store value and return the quantized value actually stored.
 * Critical for CPMMH correctness: the filter must use exactly what's stored,
 * including any quantization effects.
 */
__device__ __forceinline__ float noise_store_roundtrip(noise_t* ptr, int64_t idx, float val) {
    half h = __float2half(val);
    ptr[idx] = h;
    return __half2float(h);
}

#else  /* FP32 (default) */

/* ── FP32 Implementation (full precision, default) ── */

typedef float noise_t;
#define NOISE_SIZEOF sizeof(float)

__device__ __forceinline__ float noise_load(const noise_t* ptr, int64_t idx) {
    return ptr[idx];
}

__device__ __forceinline__ void noise_store(noise_t* ptr, int64_t idx, float val) {
    ptr[idx] = val;
}

/** For FP32, store_roundtrip is identity (no quantization). */
__device__ __forceinline__ float noise_store_roundtrip(noise_t* ptr, int64_t idx, float val) {
    ptr[idx] = val;
    return val;
}

#endif  /* SMC2_NOISE_FP16 */

/* ── Host-side noise helpers ── */

static inline size_t noise_array_bytes(int64_t count) {
    return (size_t)count * NOISE_SIZEOF;
}

static inline bool noise_is_fp32(void) {
#ifdef SMC2_NOISE_FP16
    return false;
#else
    return true;
#endif
}

static inline const char* noise_precision_str(void) {
#ifdef SMC2_NOISE_FP16
    return "FP16";
#else
    return "FP32";
#endif
}

/*═══════════════════════════════════════════════════════════════════════════════
 * §3: MODEL TYPES
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Gaussian prior specification for learned θ parameters (8D)
 *
 * Each parameter has independent N(mean, std²) prior.
 * Used in MH acceptance ratio: π(θ*)/π(θ).
 */
struct SVPrior {
    float rho_mean, rho_std;
    float sigma_total_mean, sigma_total_std;
    float r_split_mean, r_split_std;
    float mu_base_mean, mu_base_std;
    float mu_scale_mean, mu_scale_std;
    float mu_rate_mean, mu_rate_std;
    float sigma_scale_mean, sigma_scale_std;
    float sigma_rate_mean, sigma_rate_std;
};

/**
 * @brief Hard bounds for learned parameter support.
 * Parameters outside bounds → log_prior = -∞ (instant rejection).
 */
struct SVBounds {
    float rho_min, rho_max;
    float sigma_total_min, sigma_total_max;
    float r_split_min, r_split_max;
    float mu_base_min, mu_base_max;
    float mu_scale_min, mu_scale_max;
    float mu_rate_min, mu_rate_max;
    float sigma_scale_min, sigma_scale_max;
    float sigma_rate_min, sigma_rate_max;
};

/**
 * @brief Regime-dependent curve: f(z) = base + scale * (1 - exp(-rate * z))
 */
struct SVCurve {
    float base;
    float scale;
    float rate;
};

/*═══════════════════════════════════════════════════════════════════════════════
 * §4: DEFAULT CONFIGURATIONS
 *
 * Named functions instead of magic numbers buried in smc2_cuda_alloc().
 * Caller can use these or provide custom values.
 *═══════════════════════════════════════════════════════════════════════════════*/

static inline SVPrior sv_default_prior(void) {
    SVPrior p;
    /* ρ ∈ (0.8, 0.999): high persistence expected */
    p.rho_mean = 0.95f;            p.rho_std = 0.02f;
    /* σ_total ≈ √(σ_z² + σ_base²): total vol-of-vol */
    p.sigma_total_mean = 0.18f;    p.sigma_total_std = 0.1f;
    /* r = σ_z/σ_total ∈ (0,1): weakly identified */
    p.r_split_mean = 0.5f;         p.r_split_std = 0.2f;
    /* μ(z) curve parameters */
    p.mu_base_mean = -1.0f;        p.mu_base_std = 0.5f;
    p.mu_scale_mean = 0.5f;        p.mu_scale_std = 0.3f;
    p.mu_rate_mean = 1.0f;         p.mu_rate_std = 0.5f;
    /* σ_h(z) curve parameters */
    p.sigma_scale_mean = 0.1f;     p.sigma_scale_std = 0.05f;
    p.sigma_rate_mean = 1.0f;      p.sigma_rate_std = 0.5f;
    return p;
}

static inline SVBounds sv_default_bounds(void) {
    SVBounds b;
    b.rho_min = 0.8f;              b.rho_max = 0.999f;
    b.sigma_total_min = 0.01f;     b.sigma_total_max = 1.5f;
    b.r_split_min = 0.01f;         b.r_split_max = 0.99f;
    b.mu_base_min = -10.0f;        b.mu_base_max = 5.0f;
    b.mu_scale_min = 0.01f;        b.mu_scale_max = 5.0f;
    b.mu_rate_min = 0.1f;          b.mu_rate_max = 10.0f;
    b.sigma_scale_min = 0.001f;    b.sigma_scale_max = 1.0f;
    b.sigma_rate_min = 0.1f;       b.sigma_rate_max = 10.0f;
    return b;
}

static inline SVCurve sv_default_theta_curve(void) {
    SVCurve c;
    c.base = 0.02f;
    c.scale = 0.08f;
    c.rate = 1.5f;
    return c;
}

/** Default CPMMH proposal standard deviations (8D random walk) */
static inline void sv_default_proposal_std(float* out) {
    out[0] = 0.01f;   /* rho — tightly constrained */
    out[1] = 0.02f;   /* sigma_total */
    out[2] = 0.05f;   /* r_split — weakly identified, wider step */
    out[3] = 0.1f;    /* mu_base */
    out[4] = 0.05f;   /* mu_scale */
    out[5] = 0.1f;    /* mu_rate */
    out[6] = 0.02f;   /* sigma_scale */
    out[7] = 0.1f;    /* sigma_rate */
}

/** Default prior values for BPF fallback (before convergence) */
static inline void sv_default_prior_fallback(float* out) {
    out[0] = 0.85f;   /* ρ         */
    out[1] = 0.30f;   /* σ_total   */
    out[2] = 0.50f;   /* r_split   */
    out[3] = -10.0f;  /* μ_base    */
    out[4] = 3.00f;   /* μ_scale   */
    out[5] = 1.00f;   /* μ_rate    */
    out[6] = 0.50f;   /* σ_scale   */
    out[7] = 0.80f;   /* σ_rate    */
}

/*═══════════════════════════════════════════════════════════════════════════════
 * §5: ALGORITHM TYPES
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief θ-particle population with embedded RBPF state (SoA layout)
 *
 * Memory layout uses Structure-of-Arrays for coalesced GPU access.
 *
 * 8 parameters are learned:
 *   rho, sigma_total, r_split, mu_base, mu_scale, mu_rate, sigma_scale, sigma_rate
 *
 * Physical params derived per-particle:
 *   sigma_z    = r_split * sigma_total
 *   sigma_base = sqrt(1-r²) * sigma_total
 *
 * θ(z) speed curve remains fixed (in constant memory as SVCurve).
 */
struct ThetaParticlesSoA {
    /* ═══ Learned θ-level arrays (N_theta elements) ═══ */
    float* rho;
    float* sigma_total;
    float* r_split;
    float* mu_base;
    float* mu_scale;
    float* mu_rate;
    float* sigma_scale;
    float* sigma_rate;

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
    SVCurve theta_curve;
    float proposal_std[N_PARAMS];

    /* ═══ Fixed parameter mask ═══ */
    uint8_t fixed_mask[N_PARAMS];
    float   fixed_values[N_PARAMS];

    /* ═══ Algorithm settings ═══ */
    float ess_threshold_outer;
    float ess_threshold_inner;
    int K_rejuv;

    /* ═══ Batch update acceleration ═══ */
    cudaStream_t compute_stream;
    float* h_ess_pinned;
    int ess_check_interval;

    /* ═══ Fixed-Lag PMMH ═══ */
    int fixed_lag_L;
    int t_checkpoint;
    float* d_checkpoint_z;
    float* d_checkpoint_mu_h;
    float* d_checkpoint_var_h;
    float* d_checkpoint_log_w;
    float* d_checkpoint_ll;

    /** Dedicated scratch for checkpoint reindexing after outer resample.
     *  Separate from d_particles_temp to prevent corruption after the
     *  particle pointer swap. */
    float* d_checkpoint_scratch_z;
    float* d_checkpoint_scratch_mu_h;
    float* d_checkpoint_scratch_var_h;
    float* d_checkpoint_scratch_log_w;
    float* d_checkpoint_scratch_ll;

    /* ═══ Diagnostics ═══ */
    int n_resamples;
    int n_rejuv_accepts;
    int n_rejuv_total;

    /* ═══ Adaptive Proposals ═══ */
    float* d_temp_mean;
    float* d_temp_cov;
    bool use_adaptive_proposals;

    /* ═══ RNG ═══ */
    uint64_t user_seed;
    uint64_t host_rng_state;
};

/**
 * @brief Diagnostic snapshot — decouples consumers from SMC² internals
 *
 * The param_tracker and any other consumer reads diagnostics through this
 * struct instead of reaching into SMC2StateCUDA fields directly.
 */
struct SMC2Diagnostics {
    int   n_resamples;
    int   n_rejuv_accepts;
    int   n_rejuv_total;
    float outer_ess;
    float accept_rate;
};

/*═══════════════════════════════════════════════════════════════════════════════
 * §6: DEVICE UTILITIES
 *═══════════════════════════════════════════════════════════════════════════════*/

/* ── Warp-level primitives ── */

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

/* ── Block-level reductions ── */

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

/* ── Block-level inclusive scan ── */

__device__ __forceinline__
void block_inclusive_scan(volatile float* data, int n) {
    int tid = threadIdx.x;
    for (int offset = 1; offset < n; offset *= 2) {
        float temp = 0.0f;
        if (tid >= offset && tid < n) temp = data[tid - offset];
        __syncthreads();
        if (tid >= offset && tid < n) data[tid] += temp;
        __syncthreads();
    }
}

/* ── Model evaluation ── */

__device__ __forceinline__
float eval_curve(float base, float scale, float rate, float z) {
    return base + scale * (1.0f - __expf(-rate * z));
}

/** Host-side equivalent for param_tracker (no __expf) */
static inline float eval_curve_host(float base, float scale, float rate, float z) {
    return base + scale * (1.0f - expf(-rate * z));
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

/*═══════════════════════════════════════════════════════════════════════════════
 * §7: OCSN KALMAN UPDATE (declaration only)
 *
 * Defined in smc2_engine.cu alongside __constant__ OCSN arrays.
 * Not declared extern here — MSVC/NVCC extern __constant__ issues.
 *═══════════════════════════════════════════════════════════════════════════════*/

__device__ void ocsn_kalman_update(
    float y, float mu_pred, float var_pred,
    float* mu_post, float* var_post, float* log_lik
);

/*═══════════════════════════════════════════════════════════════════════════════
 * §8: SORTING BACKEND
 *
 * Deterministic sorting for CPMMH coupling preservation.
 *   -DSMC2_USE_CUB_SORT   Use CUB BlockRadixSort (deterministic, slower for small N)
 *   (default)             Use Bitonic sort (deterministic, fast for N≤1024)
 *═══════════════════════════════════════════════════════════════════════════════*/

#ifndef SMC2_USE_CUB_SORT

/* ── Bitonic Sort (default) ──
 * Optimal for small N (≤1024): zero library overhead, fully parallel
 * compare-swap network, deterministic fixed topology.
 * Complexity: O(N log²N) comparisons, log²N parallel stages. */

template<int BLOCK_SIZE>
__device__ __forceinline__
void cpmmh_sort(
    float* __restrict__ s_z,
    float* __restrict__ s_mu,
    float* __restrict__ s_var,
    int* __restrict__ s_idx,
    void* s_temp  /* unused */
) {
    static_assert((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0, "BLOCK_SIZE must be power of 2");
    static_assert(BLOCK_SIZE <= 1024, "BLOCK_SIZE must be <= 1024");
    (void)s_temp;

    int tid = threadIdx.x;
    s_idx[tid] = tid;
    __syncthreads();

    #pragma unroll 1
    for (int k = 2; k <= BLOCK_SIZE; k <<= 1) {
        #pragma unroll 1
        for (int j = k >> 1; j > 0; j >>= 1) {
            int partner = tid ^ j;
            if (partner > tid) {
                bool ascending = ((tid & k) == 0);
                float key_lo = s_mu[tid], key_hi = s_mu[partner];
                int idx_lo = s_idx[tid], idx_hi = s_idx[partner];
                bool should_swap = ascending ? (key_lo > key_hi) : (key_lo < key_hi);
                if (should_swap) {
                    s_mu[tid] = key_hi;      s_mu[partner] = key_lo;
                    s_idx[tid] = idx_hi;     s_idx[partner] = idx_lo;
                }
            }
            __syncthreads();
        }
    }

    int src_idx = s_idx[tid];
    float my_z = s_z[src_idx], my_var = s_var[src_idx];
    __syncthreads();
    s_z[tid] = my_z;
    s_var[tid] = my_var;
    __syncthreads();
}

template<int BLOCK_SIZE>
__host__ __device__ __forceinline__
constexpr size_t cpmmh_sort_smem_size() { return 0; }

#else  /* SMC2_USE_CUB_SORT */

/* ── CUB BlockRadixSort (opt-in) ──
 * Uses FP16 key for faster sorting (rank order preserved). */

#include <cub/cub.cuh>

template<int BLOCK_SIZE>
__device__ __forceinline__
void cpmmh_sort(
    float* __restrict__ s_z,
    float* __restrict__ s_mu,
    float* __restrict__ s_var,
    int* __restrict__ s_idx,
    void* s_temp
) {
    static_assert(BLOCK_SIZE <= 1024, "BLOCK_SIZE must be <= 1024");
    int tid = threadIdx.x;

    half h_key = __float2half(s_mu[tid]);
    unsigned short sort_key = *reinterpret_cast<unsigned short*>(&h_key);
    unsigned short mask = -((sort_key >> 15) & 1) | 0x8000;
    sort_key ^= mask;

    unsigned short keys[1] = { sort_key };
    int values[1] = { tid };

    typedef cub::BlockRadixSort<unsigned short, BLOCK_SIZE, 1, int> BlockSortT;
    typename BlockSortT::TempStorage& temp_storage =
        *reinterpret_cast<typename BlockSortT::TempStorage*>(s_temp);

    BlockSortT(temp_storage).Sort(keys, values);
    __syncthreads();

    s_idx[tid] = values[0];
    __syncthreads();

    int src_idx = s_idx[tid];
    float gathered_z = s_z[src_idx], gathered_mu = s_mu[src_idx], gathered_var = s_var[src_idx];
    __syncthreads();
    s_z[tid] = gathered_z;
    s_mu[tid] = gathered_mu;
    s_var[tid] = gathered_var;
    __syncthreads();
}

template<int BLOCK_SIZE>
__host__ __device__ __forceinline__
constexpr size_t cpmmh_sort_smem_size() {
    return sizeof(typename cub::BlockRadixSort<unsigned short, BLOCK_SIZE, 1, int>::TempStorage);
}

#endif  /* SMC2_USE_CUB_SORT */

/*═══════════════════════════════════════════════════════════════════════════════
 * §9: SHARED MEMORY SIZE HELPERS
 *
 * Layout for RBPF step:
 *   [0..31]          : Warp reduction scratch (32 floats)
 *   [32..32+N-1]     : s_z (N floats)
 *   [32+N..32+2N-1]  : s_mu (N floats)
 *   [32+2N..32+3N-1] : s_var (N floats)
 *   [32+3N..32+4N-1] : s_cumsum (N floats) — DEDICATED, not aliased
 *   [32+4N..32+5N-1] : s_idx (N ints)
 *   [32+5N..]        : Sort temp (if CUB)
 *═══════════════════════════════════════════════════════════════════════════════*/

template<int BLOCK_SIZE>
__host__ __device__ __forceinline__
constexpr size_t rbpf_shared_mem_size() {
    size_t base = (32 + 5 * BLOCK_SIZE) * sizeof(float);
    size_t sort_temp = cpmmh_sort_smem_size<BLOCK_SIZE>();
    return base + sort_temp;
}

template<int BLOCK_SIZE>
__host__ __device__ __forceinline__
constexpr size_t cpmmh_shared_mem_size() {
    size_t base = (32 + 5 * BLOCK_SIZE) * sizeof(float);
    size_t sort_temp = cpmmh_sort_smem_size<BLOCK_SIZE>();
    return base + sort_temp;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * §10: HOST API DECLARATIONS
 *═══════════════════════════════════════════════════════════════════════════════*/

#ifdef __cplusplus
extern "C" {
#endif

/* ── Lifecycle ── */

SMC2StateCUDA* smc2_cuda_alloc(int N_theta, int N_inner);
void           smc2_cuda_free(SMC2StateCUDA* state);

/* ── Configuration (call between alloc and init) ── */

void smc2_cuda_set_seed(SMC2StateCUDA* state, uint64_t seed);
void smc2_cuda_set_noise_capacity(SMC2StateCUDA* state, int capacity);
void smc2_cuda_set_fixed_lag(SMC2StateCUDA* state, int L);
void smc2_cuda_set_cpmmh_rho(SMC2StateCUDA* state, float rho);
void smc2_cuda_set_proposal_std(SMC2StateCUDA* state, const float* std);
void smc2_cuda_set_fixed_params(SMC2StateCUDA* state,
                                 const uint8_t* mask, const float* values);

/* ── Initialization ── */

void smc2_cuda_init_from_prior(SMC2StateCUDA* state);
void smc2_cuda_init_warm(SMC2StateCUDA* state,
                          const float* warm_mean, const float* warm_cov);

/* ── Update ── */

float smc2_cuda_update(SMC2StateCUDA* state, float y_obs);
float smc2_cuda_update_batch(SMC2StateCUDA* state, const float* y_batch, int n_obs);

/* ── Queries ── */

void  smc2_cuda_get_theta_mean(SMC2StateCUDA* state, float* theta_mean);
void  smc2_cuda_get_theta_std(SMC2StateCUDA* state, float* theta_std);
void  smc2_cuda_get_theta_cov(SMC2StateCUDA* state, float* theta_mean, float* theta_cov);
float smc2_cuda_get_z_mean(SMC2StateCUDA* state);
void  smc2_cuda_get_z_range(SMC2StateCUDA* state,
                             float* z_mean_out, float* z_min_out, float* z_max_out);
void  smc2_cuda_get_z_range_robust(SMC2StateCUDA* state,
                                    float* z_mean_out, float* z_min_out, float* z_max_out);
float smc2_cuda_get_outer_ess(SMC2StateCUDA* state);
void  smc2_cuda_get_diagnostics(SMC2StateCUDA* state, SMC2Diagnostics* diag);

/* ── Internal (exposed for param_tracker, not general use) ── */

void smc2_update_adaptive_covariance(SMC2StateCUDA* state);

#ifdef __cplusplus
}
#endif

#endif /* SMC2_ENGINE_CUH */
