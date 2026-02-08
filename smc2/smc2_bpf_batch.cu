/**
 * @file smc2_bpf_batch.cu
 * @brief SMC² with BPF Inner Filter — Kernel + Host Implementation
 *
 * See smc2_bpf_batch.cuh for algorithm documentation.
 *
 * Key differences from RBPF version:
 *   - Inner state is just h[i] (no z̃, μ_h, var_h)
 *   - Observation model: Student-t log-likelihood (no OCSN/Kalman)
 *   - Parameters: 3D (ρ, σ_z, μ) instead of 8D
 *   - CPMMH sort by h only (1 array to permute vs 3)
 *   - Same correlated noise infrastructure
 */

#include "smc2_bpf_batch.cuh"
#include <cub/block/block_radix_sort.cuh>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/*═══════════════════════════════════════════════════════════════════════════════
 * CUDA ERROR CHECKING
 *═══════════════════════════════════════════════════════════════════════════════*/

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

/*═══════════════════════════════════════════════════════════════════════════════
 * CONSTANT MEMORY
 *═══════════════════════════════════════════════════════════════════════════════*/

__constant__ SVPrior3  d_prior;
__constant__ SVBounds3 d_bounds;
__constant__ float     d_nu_obs;          /* Student-t df */
__constant__ float     d_C_obs;           /* Precomputed Student-t constant */
__constant__ float     d_half_nu_p1;      /* (ν+1)/2 */
__constant__ float     d_proposal_std[SMC2_N_PARAMS];
__constant__ float     d_proposal_chol[SMC2_N_PARAMS * SMC2_N_PARAMS];

/*═══════════════════════════════════════════════════════════════════════════════
 * LOG PRIOR EVALUATION
 *═══════════════════════════════════════════════════════════════════════════════*/

__device__ float log_prior_theta(float rho, float sigma_z, float mu) {
    float p[3] = {rho, sigma_z, mu};
    float lp = 0.0f;
    #pragma unroll
    for (int i = 0; i < SMC2_N_PARAMS; i++) {
        if (p[i] < d_bounds.lo[i] || p[i] > d_bounds.hi[i]) return -INFINITY;
        float d = (p[i] - d_prior.mean[i]) / d_prior.std[i];
        lp -= 0.5f * d * d;
    }
    return lp;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * CPMMH SORT — CUB BlockRadixSort by h value
 *
 * After resampling, all particles have equal weight. Sorting by h ensures
 * that noise index i maps to the same "rank" in particle space across
 * current and proposed runs, reducing CPMMH likelihood variance.
 *
 * Only h needs permuting (BPF has no other per-particle state).
 *═══════════════════════════════════════════════════════════════════════════════*/

template<int N>
__device__ void cpmmh_sort_h(float* s_h, void* cub_temp) {
    typedef cub::BlockRadixSort<float, N, 1> BlockSort;

    float key = s_h[threadIdx.x];
    BlockSort(*reinterpret_cast<typename BlockSort::TempStorage*>(cub_temp)).Sort(key);
    s_h[threadIdx.x] = key;
    __syncthreads();
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
 *
 * One block per θ-particle. Thread 0 samples θ from prior.
 * All threads initialize inner BPF particles from stationary distribution.
 *═══════════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_init_from_prior(
    ThetaSoA particles,
    int N_theta, int N_inner,
    float* d_h_noise, float* d_u0_noise,
    int noise_capacity
) {
    int theta_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    int global_idx = theta_idx * N_inner + inner_idx;

    if (theta_idx >= N_theta || inner_idx >= N_inner) return;

    curandState* rng = &particles.rng_states[global_idx];

    __shared__ float s_rho, s_sigma_z, s_mu;

    if (inner_idx == 0) {
        curandState rng0 = *rng;
        int valid = 0, attempts = 0;
        while (!valid && attempts < 1000) {
            s_rho     = d_prior.mean[0] + d_prior.std[0] * curand_normal(&rng0);
            s_sigma_z = d_prior.mean[1] + d_prior.std[1] * curand_normal(&rng0);
            s_mu      = d_prior.mean[2] + d_prior.std[2] * curand_normal(&rng0);
            valid = (s_rho     >= d_bounds.lo[0] && s_rho     <= d_bounds.hi[0] &&
                     s_sigma_z >= d_bounds.lo[1] && s_sigma_z <= d_bounds.hi[1] &&
                     s_mu      >= d_bounds.lo[2] && s_mu      <= d_bounds.hi[2]);
            attempts++;
        }
        particles.rho[theta_idx]     = s_rho;
        particles.sigma_z[theta_idx] = s_sigma_z;
        particles.mu[theta_idx]      = s_mu;
        particles.log_weight[theta_idx]     = 0.0f;
        particles.weight[theta_idx]         = 1.0f / N_theta;
        particles.log_likelihood[theta_idx] = 0.0f;
        particles.ess_inner[theta_idx]      = (float)N_inner;

        /* Store resampling noise for t=0 */
        float u0_raw = curand_normal(&rng0);
        int64_t u0_idx = (int64_t)theta_idx * (noise_capacity + 1);
        d_u0_noise[u0_idx] = u0_raw;
        *rng = rng0;
    }
    __syncthreads();

    /* Initialize h from stationary: N(μ, σ_z²/(1−ρ²)) */
    float rho = s_rho, sigma_z = s_sigma_z, mu = s_mu;
    float stat_std = sigma_z / sqrtf(fmaxf(1.0f - rho * rho, 1e-6f));

    float eps = curand_normal(rng);
    /* Store noise for CPMMH replay */
    int64_t h_noise_idx = (int64_t)theta_idx * N_inner * (noise_capacity + 1) + inner_idx;
    d_h_noise[h_noise_idx] = eps;

    float h = mu + stat_std * eps;
    particles.inner_h[global_idx]     = h;
    particles.inner_log_w[global_idx] = -__logf((float)N_inner);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: Inner BPF Step (Forward Filtering)
 *
 * Per-tick: resample → sort → propagate h → Student-t weight → accumulate LL
 * One block per θ-particle, N_INNER threads = inner BPF particles.
 * Always resamples (CPMMH determinism requirement).
 *═══════════════════════════════════════════════════════════════════════════════*/

template<int N_INNER>
__global__
__launch_bounds__(N_INNER)
void kernel_bpf_step_impl(
    ThetaSoA particles,
    float y_obs,
    int N_theta,
    float* d_h_noise, float* d_u0_noise,
    int t_current, int noise_capacity
) {
    static_assert(N_INNER <= 1024, "N_INNER must be <= 1024");

    int theta_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    int global_idx = theta_idx * N_INNER + inner_idx;

    if (theta_idx >= N_theta || inner_idx >= N_INNER) return;

    extern __shared__ char shared_raw[];
    float* s_red = reinterpret_cast<float*>(shared_raw);
    float* s_h   = &s_red[32];
    float* s_cdf = &s_h[N_INNER];

    __shared__ float s_rho, s_sigma_z, s_mu;
    __shared__ float s_log_max, s_sum_w, s_u0;

    if (inner_idx == 0) {
        s_rho     = particles.rho[theta_idx];
        s_sigma_z = particles.sigma_z[theta_idx];
        s_mu      = particles.mu[theta_idx];
    }
    __syncthreads();

    curandState local_rng = particles.rng_states[global_idx];

    float h     = particles.inner_h[global_idx];
    float log_w = particles.inner_log_w[global_idx];

    /* Noise indices */
    int64_t h_noise_base = (int64_t)theta_idx * N_INNER * (noise_capacity + 1);
    int64_t h_noise_idx  = h_noise_base + (int64_t)(t_current + 1) * N_INNER + inner_idx;
    int64_t u0_idx       = (int64_t)theta_idx * (noise_capacity + 1) + (t_current + 1);

    /* Generate and store noise */
    float eps_raw = curand_normal(&local_rng);
    d_h_noise[h_noise_idx] = eps_raw;

    if (inner_idx == 0) {
        float u0_raw = curand_normal(&local_rng);
        d_u0_noise[u0_idx] = u0_raw;
        s_u0 = u0_from_noise(u0_raw);
    }
    __syncthreads();

    /*───── RESAMPLE (always, for CPMMH determinism) ─────*/
    s_h[inner_idx] = h;
    __syncthreads();

    float lm = block_reduce_max(log_w, s_red);
    if (inner_idx == 0) s_log_max = lm;
    __syncthreads();

    float w_un = __expf(log_w - s_log_max);
    float sw   = block_reduce_sum(w_un, s_red);
    if (inner_idx == 0) s_sum_w = sw;
    __syncthreads();

    s_cdf[inner_idx] = w_un / s_sum_w;
    __syncthreads();
    block_inclusive_scan(s_cdf, N_INNER);
    if (inner_idx == N_INNER - 1) s_cdf[N_INNER - 1] = 1.0f;
    __syncthreads();

    float u = (s_u0 + (float)inner_idx) / (float)N_INNER;
    int lo = 0, hi = N_INNER - 1;
    while (lo < hi) { int mid = (lo + hi) / 2; if (s_cdf[mid] < u) lo = mid + 1; else hi = mid; }
    h = s_h[lo];
    log_w = -__logf((float)N_INNER);
    __syncthreads();

    /*───── CPMMH SORT ─────*/
    if ((t_current % SMC2_SORT_EVERY_K) == 0) {
        s_h[inner_idx] = h;
        __syncthreads();
        cpmmh_sort_h<N_INNER>(s_h,
            reinterpret_cast<void*>(&s_cdf[0]));  /* Reuse s_cdf as CUB temp after scan done */
        h = s_h[inner_idx];
        __syncthreads();
    }

    /*───── PROPAGATE ─────*/
    float h_new = s_mu + s_rho * (h - s_mu) + s_sigma_z * eps_raw;

    /*───── WEIGHT (Student-t) ─────*/
    float ll = student_t_log_lik(y_obs, h_new, d_nu_obs, d_C_obs);
    log_w += ll;

    /*───── NORMALIZE + ESS + LL INCREMENT ─────*/
    lm = block_reduce_max(log_w, s_red);
    if (inner_idx == 0) s_log_max = lm;
    __syncthreads();

    w_un = __expf(log_w - s_log_max);
    sw = block_reduce_sum(w_un, s_red);
    if (inner_idx == 0) s_sum_w = sw;
    __syncthreads();

    float wn = w_un / fmaxf(s_sum_w, 1e-30f);
    float wsq = wn * wn;
    float sum_wsq = block_reduce_sum(wsq, s_red);
    float ess = 1.0f / fmaxf(sum_wsq, 1e-30f);

    float ll_incr = s_log_max + __logf(fmaxf(s_sum_w, 1e-30f)) - __logf((float)N_INNER);

    /*───── STORE ─────*/
    particles.inner_h[global_idx]     = h_new;
    particles.inner_log_w[global_idx] = log_w;
    particles.rng_states[global_idx]  = local_rng;

    if (inner_idx == 0) {
        particles.ess_inner[theta_idx] = ess;
        particles.log_weight[theta_idx]     += ll_incr;
        particles.log_likelihood[theta_idx] += ll_incr;
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: Outer ESS
 *═══════════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_compute_outer_ess(ThetaSoA particles, float* d_ess_out, int N_theta) {
    extern __shared__ float s_data[];
    int idx = threadIdx.x;

    float lw = (idx < N_theta) ? particles.log_weight[idx] : -1e30f;
    float lm = block_reduce_max(lw, s_data);
    __shared__ float s_lm;
    if (idx == 0) s_lm = lm;
    __syncthreads();

    float w = (idx < N_theta) ? __expf(lw - s_lm) : 0.0f;
    float sw = block_reduce_sum(w, s_data);
    __shared__ float s_sw;
    if (idx == 0) s_sw = sw;
    __syncthreads();

    if (idx < N_theta) {
        w /= s_sw;
        particles.weight[idx] = w;
    }

    float wsq = (idx < N_theta) ? w * w : 0.0f;
    float sum_wsq = block_reduce_sum(wsq, s_data);
    if (idx == 0) *d_ess_out = 1.0f / fmaxf(sum_wsq, 1e-30f);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: Outer Resampling
 *═══════════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_outer_resample(
    ThetaSoA particles, int* d_ancestors, float* d_uniform, int N_theta
) {
    extern __shared__ float s_cum[];
    int idx = threadIdx.x;

    if (idx < N_theta) s_cum[idx] = particles.weight[idx];
    __syncthreads();

    if (idx == 0) {
        for (int i = 1; i < N_theta; i++) s_cum[i] += s_cum[i - 1];
        s_cum[N_theta - 1] = 1.0f;
    }
    __syncthreads();

    if (idx < N_theta) {
        float u = (*d_uniform + (float)idx) / (float)N_theta;
        int lo = 0, hi = N_theta - 1;
        while (lo < hi) { int mid = (lo + hi) / 2; if (s_cum[mid] < u) lo = mid + 1; else hi = mid; }
        d_ancestors[idx] = lo;
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: Copy θ-Particles After Resampling
 *═══════════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_copy_theta_particles(
    ThetaSoA src, ThetaSoA dst, int* d_ancestors,
    int N_theta, int N_inner, unsigned long long resample_seed
) {
    int theta_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    if (theta_idx >= N_theta) return;

    int anc = d_ancestors[theta_idx];

    if (inner_idx == 0) {
        dst.rho[theta_idx]     = src.rho[anc];
        dst.sigma_z[theta_idx] = src.sigma_z[anc];
        dst.mu[theta_idx]      = src.mu[anc];
        dst.log_weight[theta_idx]     = 0.0f;
        dst.weight[theta_idx]         = 1.0f / N_theta;
        dst.log_likelihood[theta_idx] = src.log_likelihood[anc];
        dst.ess_inner[theta_idx]      = src.ess_inner[anc];
    }

    if (inner_idx < N_inner) {
        int si = anc * N_inner + inner_idx;
        int di = theta_idx * N_inner + inner_idx;
        dst.inner_h[di]     = src.inner_h[si];
        dst.inner_log_w[di] = src.inner_log_w[si];
        curand_init(resample_seed, di, 0, &dst.rng_states[di]);
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: Copy Noise Arrays (Ping-Pong)
 *═══════════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_copy_noise_arrays(
    const float* src_h, float* dst_h,
    const float* src_u0, float* dst_u0,
    const int* d_ancestors,
    int N_theta, int N_inner, int t_current, int noise_capacity
) {
    int theta_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    if (theta_idx >= N_theta || inner_idx >= N_inner) return;

    int anc = d_ancestors[theta_idx];
    int64_t dst_base = (int64_t)theta_idx * N_inner * (noise_capacity + 1);
    int64_t src_base = (int64_t)anc       * N_inner * (noise_capacity + 1);

    for (int t = 0; t <= t_current + 1; t++) {
        int64_t si = src_base + (int64_t)t * N_inner + inner_idx;
        int64_t di = dst_base + (int64_t)t * N_inner + inner_idx;
        dst_h[di] = src_h[si];
    }

    if (inner_idx == 0) {
        int64_t dst_u0b = (int64_t)theta_idx * (noise_capacity + 1);
        int64_t src_u0b = (int64_t)anc       * (noise_capacity + 1);
        for (int t = 0; t <= t_current + 1; t++) {
            dst_u0[dst_u0b + t] = src_u0[src_u0b + t];
        }
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: Copy Checkpoint After Outer Resampling
 *═══════════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_copy_checkpoint(
    const float* src_h, const float* src_lw, const float* src_ll,
    float* dst_h, float* dst_lw, float* dst_ll,
    const int* d_ancestors, int N_theta, int N_inner
) {
    int theta_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    if (theta_idx >= N_theta || inner_idx >= N_inner) return;

    int anc = d_ancestors[theta_idx];
    int si = anc * N_inner + inner_idx;
    int di = theta_idx * N_inner + inner_idx;

    dst_h[di]  = src_h[si];
    dst_lw[di] = src_lw[si];
    if (inner_idx == 0) dst_ll[theta_idx] = src_ll[anc];
}

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: CPMMH Rejuvenation with BPF Inner Filter
 *
 * The big one. For each θ-particle:
 *   1. Propose θ* via adaptive random walk
 *   2. Initialize inner BPF from θ*'s stationary (or checkpoint)
 *   3. Replay observations with correlated noise
 *   4. Accept/reject by Metropolis-Hastings
 *═══════════════════════════════════════════════════════════════════════════════*/

template<int N_INNER>
__global__
__launch_bounds__(N_INNER)
void kernel_cpmmh_rejuvenate_impl(
    ThetaSoA particles,
    ThetaSoA scratch,
    const float* y_history,
    float* d_h_noise_curr, float* d_h_noise_other,
    float* d_u0_noise_curr, float* d_u0_noise_other,
    int t_current, int N_theta,
    int noise_capacity, float cpmmh_rho,
    int* d_accepts, int* d_swap_flags,
    /* Fixed-lag */
    int t_checkpoint,
    const float* d_cp_h, const float* d_cp_lw, const float* d_cp_ll
) {
    static_assert(N_INNER <= 1024, "");

    int theta_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    int global_idx = theta_idx * N_INNER + inner_idx;

    if (theta_idx >= N_theta || inner_idx >= N_INNER) return;

    extern __shared__ char shared_raw[];
    float* s_red = reinterpret_cast<float*>(shared_raw);
    float* s_h   = &s_red[32];
    float* s_cdf = &s_h[N_INNER];
    void*  s_cub = reinterpret_cast<void*>(&s_cdf[N_INNER]);

    __shared__ float s_log_max, s_sum_w;
    __shared__ float s_rho_c, s_sz_c, s_mu_c;
    __shared__ float s_rho_p, s_sz_p, s_mu_p;
    __shared__ float s_ll_curr, s_ll_prop, s_lp_curr, s_lp_prop;
    __shared__ int   s_accept, s_valid;
    __shared__ float s_u0;

    curandState local_rng = particles.rng_states[global_idx];

    /*───── PROPOSE θ* (thread 0) ─────*/
    if (inner_idx == 0) {
        s_rho_c = particles.rho[theta_idx];
        s_sz_c  = particles.sigma_z[theta_idx];
        s_mu_c  = particles.mu[theta_idx];
        s_ll_curr = particles.log_likelihood[theta_idx];
        s_lp_curr = log_prior_theta(s_rho_c, s_sz_c, s_mu_c);

        float z[SMC2_N_PARAMS];
        for (int i = 0; i < SMC2_N_PARAMS; i++) z[i] = curand_normal(&local_rng);

        float pert[SMC2_N_PARAMS] = {0};
        float mix_u = curand_uniform(&local_rng);

        if (mix_u > 0.05f) {
            /* 95%: Adaptive correlated proposal (Cholesky) */
            for (int i = 0; i < SMC2_N_PARAMS; i++) {
                float s = 0.0f;
                for (int j = 0; j <= i; j++)
                    s += d_proposal_chol[i * SMC2_N_PARAMS + j] * z[j];
                pert[i] = s;
            }
        } else {
            /* 5%: Fixed independent (ergodicity) */
            for (int i = 0; i < SMC2_N_PARAMS; i++)
                pert[i] = d_proposal_std[i] * z[i];
        }

        s_rho_p = s_rho_c + pert[0];
        s_sz_p  = s_sz_c  + pert[1];
        s_mu_p  = s_mu_c  + pert[2];

        s_lp_prop = log_prior_theta(s_rho_p, s_sz_p, s_mu_p);
        s_valid = isfinite(s_lp_prop) ? 1 : 0;
        s_accept = 0;
    }
    __syncthreads();

    if (!s_valid) {
        if (inner_idx == 0) d_swap_flags[theta_idx] = 0;
        particles.rng_states[global_idx] = local_rng;
        return;
    }

    float rho_p = s_rho_p, sz_p = s_sz_p, mu_p = s_mu_p;
    int64_t h_base = (int64_t)theta_idx * N_INNER * (noise_capacity + 1);
    float scale = sqrtf(1.0f - cpmmh_rho * cpmmh_rho);

    /*───── INITIALIZE INNER BPF ─────*/
    float h, log_w;
    float ll_accum = 0.0f;
    int t_start;

    if (t_checkpoint >= 0 && d_cp_h != nullptr) {
        /* Fixed-lag: load checkpoint */
        h     = d_cp_h[global_idx];
        log_w = d_cp_lw[global_idx];
        t_start = t_checkpoint + 1;
    } else {
        /* Full history: init from θ_prop stationary */
        float stat_std = sz_p / sqrtf(fmaxf(1.0f - rho_p * rho_p, 1e-6f));
        float eps_curr = d_h_noise_curr[h_base + inner_idx];
        float eps_fresh = curand_normal(&local_rng);
        float eps_prop = cpmmh_rho * eps_curr + scale * eps_fresh;
        d_h_noise_other[h_base + inner_idx] = eps_prop;

        h = mu_p + stat_std * eps_prop;
        log_w = -__logf((float)N_INNER);
        t_start = 0;
    }

    /*───── REPLAY OBSERVATIONS ─────*/
    for (int t = t_start; t <= t_current; t++) {
        float y_obs = y_history[t];

        /* Normalize + CDF */
        float lm = block_reduce_max(log_w, s_red);
        if (inner_idx == 0) s_log_max = lm;
        __syncthreads();
        float w_un = __expf(log_w - s_log_max);
        float sw = block_reduce_sum(w_un, s_red);
        if (inner_idx == 0) s_sum_w = sw;
        __syncthreads();

        s_h[inner_idx] = h;
        s_cdf[inner_idx] = w_un / fmaxf(s_sum_w, 1e-30f);
        __syncthreads();
        block_inclusive_scan(s_cdf, N_INNER);
        if (inner_idx == N_INNER - 1) s_cdf[N_INNER - 1] = 1.0f;
        __syncthreads();

        /* Correlate noise for t+1 */
        int64_t idx_t1 = h_base + (int64_t)(t + 1) * N_INNER + inner_idx;
        float eps_curr = d_h_noise_curr[idx_t1];
        float eps_fresh = curand_normal(&local_rng);
        float eps_prop = cpmmh_rho * eps_curr + scale * eps_fresh;
        d_h_noise_other[idx_t1] = eps_prop;

        /* Correlate resampling noise */
        if (inner_idx == 0) {
            int64_t u0_idx = (int64_t)theta_idx * (noise_capacity + 1) + (t + 1);
            float u0_curr = d_u0_noise_curr[u0_idx];
            float u0_fresh = curand_normal(&local_rng);
            float u0_prop = cpmmh_rho * u0_curr + scale * u0_fresh;
            d_u0_noise_other[u0_idx] = u0_prop;
            s_u0 = u0_from_noise(u0_prop);
        }
        __syncthreads();

        /* Systematic resample */
        float u = (s_u0 + (float)inner_idx) / (float)N_INNER;
        int lo = 0, hi = N_INNER - 1;
        while (lo < hi) { int mid = (lo + hi) / 2; if (s_cdf[mid] < u) lo = mid + 1; else hi = mid; }
        h = s_h[lo];
        log_w = -__logf((float)N_INNER);
        __syncthreads();

        /* CPMMH sort */
        if ((t % SMC2_SORT_EVERY_K) == 0) {
            s_h[inner_idx] = h;
            __syncthreads();
            cpmmh_sort_h<N_INNER>(s_h, s_cub);
            h = s_h[inner_idx];
            __syncthreads();
        }

        /* Propagate */
        float h_new = mu_p + rho_p * (h - mu_p) + sz_p * eps_prop;

        /* Weight */
        float ll = student_t_log_lik(y_obs, h_new, d_nu_obs, d_C_obs);
        log_w += ll;

        /* Likelihood increment */
        lm = block_reduce_max(log_w, s_red);
        if (inner_idx == 0) s_log_max = lm;
        __syncthreads();
        w_un = __expf(log_w - s_log_max);
        sw = block_reduce_sum(w_un, s_red);
        if (inner_idx == 0) s_sum_w = sw;
        __syncthreads();

        ll_accum += s_log_max + __logf(fmaxf(s_sum_w, 1e-30f)) - __logf((float)N_INNER);
        h = h_new;
    }

    /* Store proposed state to scratch */
    scratch.inner_h[global_idx]     = h;
    scratch.inner_log_w[global_idx] = log_w;

    __shared__ float s_ll_base;

    if (inner_idx == 0) {
        float ll_base = (t_checkpoint >= 0 && d_cp_ll) ? d_cp_ll[theta_idx] : 0.0f;
        s_ll_base = ll_base;
        s_ll_prop = ll_base + ll_accum;
    }
    __syncthreads();

    /*───── MH ACCEPT / REJECT ─────*/
    if (inner_idx == 0) {
        float ll_curr_win = s_ll_curr - s_ll_base;
        float ll_prop_win = ll_accum;
        float log_alpha = (ll_prop_win + s_lp_prop) - (ll_curr_win + s_lp_curr);

        float u = curand_uniform(&local_rng);
        s_accept = (__logf(u) < log_alpha) ? 1 : 0;

        if (s_accept) {
            particles.rho[theta_idx]     = s_rho_p;
            particles.sigma_z[theta_idx] = s_sz_p;
            particles.mu[theta_idx]      = s_mu_p;
            particles.log_likelihood[theta_idx] = s_ll_prop;
            atomicAdd(d_accepts, 1);
        }
        d_swap_flags[theta_idx] = s_accept;
    }
    __syncthreads();

    if (s_accept) {
        particles.inner_h[global_idx]     = scratch.inner_h[global_idx];
        particles.inner_log_w[global_idx] = scratch.inner_log_w[global_idx];
    }

    particles.rng_states[global_idx] = local_rng;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: Commit Accepted Noise
 *═══════════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_commit_noise(
    float* d_h0, float* d_h1, float* d_u0_0, float* d_u0_1,
    const int* d_swap_flags,
    int N_theta, int N_inner, int t_current, int noise_capacity, int t_start
) {
    int theta_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    if (theta_idx >= N_theta || inner_idx >= N_inner) return;
    if (!d_swap_flags[theta_idx]) return;

    int64_t base = (int64_t)theta_idx * N_inner * (noise_capacity + 1);
    for (int t = t_start; t <= t_current + 1; t++) {
        int64_t i = base + (int64_t)t * N_inner + inner_idx;
        d_h0[i] = d_h1[i];
    }
    if (inner_idx == 0) {
        int64_t u0b = (int64_t)theta_idx * (noise_capacity + 1);
        for (int t = t_start; t <= t_current + 1; t++)
            d_u0_0[u0b + t] = d_u0_1[u0b + t];
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: Save / Reset
 *═══════════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_save_checkpoint(
    const ThetaSoA p,
    float* cp_h, float* cp_lw, float* cp_ll,
    int N_theta, int N_inner
) {
    int ti = blockIdx.x, ii = threadIdx.x;
    int gi = ti * N_inner + ii;
    if (ti >= N_theta || ii >= N_inner) return;
    cp_h[gi]  = p.inner_h[gi];
    cp_lw[gi] = p.inner_log_w[gi];
    if (ii == 0) cp_ll[ti] = p.log_likelihood[ti];
}

__global__ void kernel_reset_outer_weights(ThetaSoA p, int N_theta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_theta) return;
    p.log_weight[idx]     = 0.0f;
    p.weight[idx]         = 1.0f / (float)N_theta;
    p.log_likelihood[idx] = 0.0f;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: Compute Particle Moments (for adaptive Haario proposals)
 *═══════════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_compute_moments(
    ThetaSoA p, float* d_mean, float* d_cov, int N_theta
) {
    extern __shared__ float s[];
    int tid = threadIdx.x;

    float par[SMC2_N_PARAMS];
    if (tid < N_theta) {
        par[0] = p.rho[tid];
        par[1] = p.sigma_z[tid];
        par[2] = p.mu[tid];
    } else {
        par[0] = par[1] = par[2] = 0.0f;
    }

    __shared__ float s_mean[SMC2_N_PARAMS];
    for (int i = 0; i < SMC2_N_PARAMS; i++) {
        s[tid] = (tid < N_theta) ? par[i] : 0.0f;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) s[tid] += s[tid + stride];
            __syncthreads();
        }
        if (tid == 0) s_mean[i] = s[0] / (float)N_theta;
        __syncthreads();
    }
    if (tid < SMC2_N_PARAMS) d_mean[tid] = s_mean[tid];

    float c[SMC2_N_PARAMS];
    for (int i = 0; i < SMC2_N_PARAMS; i++) c[i] = par[i] - s_mean[i];

    float inv_N1 = 1.0f / (float)(N_theta - 1);
    for (int i = 0; i < SMC2_N_PARAMS; i++) {
        for (int j = 0; j <= i; j++) {
            s[tid] = (tid < N_theta) ? c[i] * c[j] : 0.0f;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (tid < stride) s[tid] += s[tid + stride];
                __syncthreads();
            }
            if (tid == 0) {
                float cv = s[0] * inv_N1;
                d_cov[i * SMC2_N_PARAMS + j] = cv;
                d_cov[j * SMC2_N_PARAMS + i] = cv;
            }
            __syncthreads();
        }
    }
}

/*═══════════════════════════════════════════════════════════════════════════════
 * HOST HELPERS
 *═══════════════════════════════════════════════════════════════════════════════*/

static inline uint64_t xorshift64star(uint64_t* s) {
    uint64_t x = *s;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static inline float xorshift_uniform(uint64_t* s) {
    return (float)((xorshift64star(s) >> 11) + 1) * (1.0f / 9007199254740994.0f);
}

#define ADAPTIVE_SCALE_3D 1.88853f  /* 2.38² / 3 */

static void update_adaptive_covariance(SMC2BPFState* s) {
    if (!s->use_adaptive_proposals) return;

    float h_cov[9], h_chol[9] = {0};
    int bs = 1;
    while (bs < s->N_theta && bs < 1024) bs *= 2;

    kernel_compute_moments<<<1, bs, bs * sizeof(float)>>>(
        s->d_particles, s->d_temp_mean, s->d_temp_cov, s->N_theta);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_cov, s->d_temp_cov, 9 * sizeof(float), cudaMemcpyDeviceToHost));

    /* Regularize + Cholesky */
    for (int i = 0; i < 3; i++) h_cov[i * 3 + i] += 1e-6f;

    bool ok = true;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j <= i; j++) {
            float sum = 0.0f;
            for (int k = 0; k < j; k++) sum += h_chol[i * 3 + k] * h_chol[j * 3 + k];
            if (i == j) {
                float val = h_cov[i * 3 + i] - sum;
                if (val <= 0.0f) { ok = false; val = 1e-8f; }
                h_chol[i * 3 + j] = sqrtf(val);
            } else {
                float d = h_chol[j * 3 + j];
                h_chol[i * 3 + j] = (d > 1e-10f) ? (h_cov[i * 3 + j] - sum) / d : 0.0f;
            }
        }
    }

    if (!ok) {
        memset(h_chol, 0, 9 * sizeof(float));
        for (int i = 0; i < 3; i++)
            h_chol[i * 3 + i] = sqrtf(fmaxf(h_cov[i * 3 + i], 1e-8f));
    }

    float sc = sqrtf(ADAPTIVE_SCALE_3D);
    for (int i = 0; i < 9; i++) h_chol[i] *= sc;

    CUDA_CHECK(cudaMemcpyToSymbol(d_proposal_chol, h_chol, 9 * sizeof(float)));
}

/*═══════════════════════════════════════════════════════════════════════════════
 * ALLOCATE θ-PARTICLE SoA
 *═══════════════════════════════════════════════════════════════════════════════*/

static void alloc_soa(ThetaSoA* p, int N_theta, int N_inner) {
    int N_total = N_theta * N_inner;
    CUDA_CHECK(cudaMalloc(&p->rho,       N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&p->sigma_z,   N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&p->mu,        N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&p->log_weight,     N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&p->weight,         N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&p->log_likelihood, N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&p->ess_inner,      N_theta * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&p->inner_h,     N_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&p->inner_log_w, N_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&p->rng_states,  N_total * sizeof(curandState)));
}

static void free_soa(ThetaSoA* p) {
    cudaFree(p->rho); cudaFree(p->sigma_z); cudaFree(p->mu);
    cudaFree(p->log_weight); cudaFree(p->weight);
    cudaFree(p->log_likelihood); cudaFree(p->ess_inner);
    cudaFree(p->inner_h); cudaFree(p->inner_log_w);
    cudaFree(p->rng_states);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * HOST API
 *═══════════════════════════════════════════════════════════════════════════════*/

SMC2BPFState* smc2_bpf_alloc(int N_theta, int N_inner) {
    SMC2BPFState* s = (SMC2BPFState*)calloc(1, sizeof(SMC2BPFState));
    s->N_theta = N_theta;
    s->N_inner = N_inner;
    s->ess_threshold_outer = 0.5f;
    s->K_rejuv = 5;

    alloc_soa(&s->d_particles, N_theta, N_inner);
    alloc_soa(&s->d_particles_temp, N_theta, N_inner);

    int N_total = N_theta * N_inner;
    kernel_init_rng<<<(N_total + 255) / 256, 256>>>(s->d_particles.rng_states, 12345ULL, N_total);
    CUDA_CHECK(cudaDeviceSynchronize());

    s->y_history_capacity = 8000;
    CUDA_CHECK(cudaMalloc(&s->d_y_history, s->y_history_capacity * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&s->d_ancestors,   N_theta * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s->d_uniform,     sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->d_ess,         sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->d_accepts,     sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s->d_swap_flags,  N_theta * sizeof(int)));

    s->noise_capacity = 2048;
    s->cpmmh_rho = 0.99f;
    s->noise_buf = 0;
    s->host_rng_state = 0x853C49E6748FEA9BULL ^ (uint64_t)time(NULL);

    int64_t h_sz  = (int64_t)N_theta * N_inner * (s->noise_capacity + 1);
    int64_t u0_sz = (int64_t)N_theta * (s->noise_capacity + 1);
    CUDA_CHECK(cudaMalloc(&s->d_h_noise[0],  h_sz  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->d_h_noise[1],  h_sz  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->d_u0_noise[0], u0_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->d_u0_noise[1], u0_sz * sizeof(float)));

    /* Defaults: prior, bounds, nu */
    s->prior.mean[0] = 0.95f;  s->prior.std[0] = 0.02f;   /* ρ */
    s->prior.mean[1] = 0.15f;  s->prior.std[1] = 0.10f;   /* σ_z */
    s->prior.mean[2] = -1.0f;  s->prior.std[2] = 0.50f;   /* μ */

    s->bounds.lo[0] = 0.80f;   s->bounds.hi[0] = 0.999f;
    s->bounds.lo[1] = 0.01f;   s->bounds.hi[1] = 1.0f;
    s->bounds.lo[2] = -10.0f;  s->bounds.hi[2] = 5.0f;

    s->nu_obs = 5.0f;

    s->proposal_std[0] = 0.005f;
    s->proposal_std[1] = 0.01f;
    s->proposal_std[2] = 0.05f;
    s->use_adaptive_proposals = true;

    /* Fixed-lag (disabled by default) */
    s->fixed_lag_L = 0;
    s->t_checkpoint = -1;
    CUDA_CHECK(cudaMalloc(&s->d_checkpoint_h,     N_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->d_checkpoint_log_w, N_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->d_checkpoint_ll,    N_theta * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&s->d_temp_mean, SMC2_N_PARAMS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->d_temp_cov,  SMC2_N_PARAMS * SMC2_N_PARAMS * sizeof(float)));

    return s;
}

void smc2_bpf_free(SMC2BPFState* s) {
    if (!s) return;
    free_soa(&s->d_particles);
    free_soa(&s->d_particles_temp);
    cudaFree(s->d_y_history);
    cudaFree(s->d_ancestors); cudaFree(s->d_uniform);
    cudaFree(s->d_ess); cudaFree(s->d_accepts); cudaFree(s->d_swap_flags);
    cudaFree(s->d_h_noise[0]); cudaFree(s->d_h_noise[1]);
    cudaFree(s->d_u0_noise[0]); cudaFree(s->d_u0_noise[1]);
    cudaFree(s->d_checkpoint_h); cudaFree(s->d_checkpoint_log_w); cudaFree(s->d_checkpoint_ll);
    cudaFree(s->d_temp_mean); cudaFree(s->d_temp_cov);
    free(s);
}

void smc2_bpf_set_seed(SMC2BPFState* s, uint64_t seed) {
    s->user_seed = seed;
    if (seed) s->host_rng_state = 0x853C49E6748FEA9BULL ^ seed;
}

void smc2_bpf_set_nu_obs(SMC2BPFState* s, float nu) { s->nu_obs = nu; }
void smc2_bpf_set_cpmmh_rho(SMC2BPFState* s, float rho) { s->cpmmh_rho = rho; }

void smc2_bpf_set_fixed_lag(SMC2BPFState* s, int L) {
    s->fixed_lag_L = L;
    s->t_checkpoint = -1;
}

void smc2_bpf_set_proposal_std(SMC2BPFState* s, const float std[SMC2_N_PARAMS]) {
    memcpy(s->proposal_std, std, SMC2_N_PARAMS * sizeof(float));
    CUDA_CHECK(cudaMemcpyToSymbol(d_proposal_std, s->proposal_std, SMC2_N_PARAMS * sizeof(float)));
}

void smc2_bpf_init(SMC2BPFState* s) {
    /* Upload model constants */
    CUDA_CHECK(cudaMemcpyToSymbol(d_prior,  &s->prior,  sizeof(SVPrior3)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_bounds, &s->bounds, sizeof(SVBounds3)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_nu_obs, &s->nu_obs, sizeof(float)));

    float C = lgammaf(0.5f * (s->nu_obs + 1.0f))
            - lgammaf(0.5f * s->nu_obs)
            - 0.5f * logf(s->nu_obs * 3.14159265f);
    CUDA_CHECK(cudaMemcpyToSymbol(d_C_obs, &C, sizeof(float)));

    float hnp1 = 0.5f * (s->nu_obs + 1.0f);
    CUDA_CHECK(cudaMemcpyToSymbol(d_half_nu_p1, &hnp1, sizeof(float)));

    CUDA_CHECK(cudaMemcpyToSymbol(d_proposal_std, s->proposal_std, SMC2_N_PARAMS * sizeof(float)));

    /* Init Cholesky to diagonal */
    float h_chol[9] = {0};
    float sc = sqrtf(ADAPTIVE_SCALE_3D);
    for (int i = 0; i < 3; i++) h_chol[i * 3 + i] = s->proposal_std[i] * sc;
    CUDA_CHECK(cudaMemcpyToSymbol(d_proposal_chol, h_chol, 9 * sizeof(float)));

    /* Init RNG */
    int N_total = s->N_theta * s->N_inner;
    unsigned long long rng_seed = s->user_seed ? s->user_seed : 12345ULL;
    kernel_init_rng<<<(N_total + 255) / 256, 256>>>(s->d_particles.rng_states, rng_seed, N_total);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Init particles from prior */
    kernel_init_from_prior<<<s->N_theta, s->N_inner>>>(
        s->d_particles, s->N_theta, s->N_inner,
        s->d_h_noise[s->noise_buf], s->d_u0_noise[s->noise_buf],
        s->noise_capacity);
    CUDA_CHECK(cudaDeviceSynchronize());

    s->n_resamples = s->n_rejuv_accepts = s->n_rejuv_total = 0;
    s->y_history_len = 0;
    s->t_current = -1;
    s->t_checkpoint = -1;
}

float smc2_bpf_update(SMC2BPFState* s, float y_obs) {
    /* Store observation */
    if (s->y_history_len >= s->y_history_capacity) {
        int nc = s->y_history_capacity * 2;
        float* nh; CUDA_CHECK(cudaMalloc(&nh, nc * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(nh, s->d_y_history, s->y_history_len * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaFree(s->d_y_history);
        s->d_y_history = nh;
        s->y_history_capacity = nc;
    }
    CUDA_CHECK(cudaMemcpy(&s->d_y_history[s->y_history_len], &y_obs, sizeof(float), cudaMemcpyHostToDevice));
    s->y_history_len++;
    s->t_current++;

    /* Grow noise if needed */
    if (s->t_current >= s->noise_capacity) {
        int nc = s->noise_capacity * 2;
        int64_t new_h  = (int64_t)s->N_theta * s->N_inner * (nc + 1);
        int64_t old_h  = (int64_t)s->N_theta * s->N_inner * (s->noise_capacity + 1);
        int64_t new_u0 = (int64_t)s->N_theta * (nc + 1);
        int64_t old_u0 = (int64_t)s->N_theta * (s->noise_capacity + 1);
        for (int b = 0; b < 2; b++) {
            float *nh, *nu;
            CUDA_CHECK(cudaMalloc(&nh, new_h * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&nu, new_u0 * sizeof(float)));
            if (b == s->noise_buf) {
                CUDA_CHECK(cudaMemcpy(nh, s->d_h_noise[b], old_h * sizeof(float), cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(nu, s->d_u0_noise[b], old_u0 * sizeof(float), cudaMemcpyDeviceToDevice));
            }
            cudaFree(s->d_h_noise[b]); cudaFree(s->d_u0_noise[b]);
            s->d_h_noise[b] = nh; s->d_u0_noise[b] = nu;
        }
        s->noise_capacity = nc;
    }

    /* Inner BPF step */
    #define DISPATCH_BPF(N) \
        kernel_bpf_step_impl<N><<<s->N_theta, N, bpf_step_smem<N>()>>>( \
            s->d_particles, y_obs, s->N_theta, \
            s->d_h_noise[s->noise_buf], s->d_u0_noise[s->noise_buf], \
            s->t_current, s->noise_capacity)

    switch (s->N_inner) {
        case 64:  DISPATCH_BPF(64);  break;
        case 128: DISPATCH_BPF(128); break;
        case 256: DISPATCH_BPF(256); break;
        case 512: DISPATCH_BPF(512); break;
        default: fprintf(stderr, "N_inner=%d unsupported\n", s->N_inner); exit(1);
    }
    #undef DISPATCH_BPF
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Outer ESS */
    kernel_compute_outer_ess<<<1, s->N_theta, 32 * sizeof(float)>>>(
        s->d_particles, s->d_ess, s->N_theta);
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_ess;
    CUDA_CHECK(cudaMemcpy(&h_ess, s->d_ess, sizeof(float), cudaMemcpyDeviceToHost));

    /* Resample + rejuvenate if ESS low */
    if (h_ess < s->ess_threshold_outer * s->N_theta) {
        s->n_resamples++;

        float h_u = xorshift_uniform(&s->host_rng_state);
        CUDA_CHECK(cudaMemcpy(s->d_uniform, &h_u, sizeof(float), cudaMemcpyHostToDevice));

        kernel_outer_resample<<<1, s->N_theta, s->N_theta * sizeof(float)>>>(
            s->d_particles, s->d_ancestors, s->d_uniform, s->N_theta);
        CUDA_CHECK(cudaDeviceSynchronize());

        unsigned long long rseed = time(NULL) * 1000ULL + s->n_resamples * 12345ULL;
        kernel_copy_theta_particles<<<s->N_theta, s->N_inner>>>(
            s->d_particles, s->d_particles_temp, s->d_ancestors,
            s->N_theta, s->N_inner, rseed);
        CUDA_CHECK(cudaDeviceSynchronize());

        int ob = 1 - s->noise_buf;
        kernel_copy_noise_arrays<<<s->N_theta, s->N_inner>>>(
            s->d_h_noise[s->noise_buf], s->d_h_noise[ob],
            s->d_u0_noise[s->noise_buf], s->d_u0_noise[ob],
            s->d_ancestors, s->N_theta, s->N_inner, s->t_current, s->noise_capacity);
        CUDA_CHECK(cudaDeviceSynchronize());
        s->noise_buf = ob;

        /* Swap particle buffers */
        ThetaSoA tmp = s->d_particles;
        s->d_particles = s->d_particles_temp;
        s->d_particles_temp = tmp;

        /* Copy checkpoint if fixed-lag */
        if (s->fixed_lag_L > 0 && s->t_checkpoint >= 0) {
            kernel_copy_checkpoint<<<s->N_theta, s->N_inner>>>(
                s->d_checkpoint_h, s->d_checkpoint_log_w, s->d_checkpoint_ll,
                s->d_particles_temp.inner_h, s->d_particles_temp.inner_log_w,
                s->d_particles_temp.log_likelihood,
                s->d_ancestors, s->N_theta, s->N_inner);
            CUDA_CHECK(cudaDeviceSynchronize());

            int Nt = s->N_theta * s->N_inner;
            CUDA_CHECK(cudaMemcpy(s->d_checkpoint_h, s->d_particles_temp.inner_h,
                                  Nt * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(s->d_checkpoint_log_w, s->d_particles_temp.inner_log_w,
                                  Nt * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(s->d_checkpoint_ll, s->d_particles_temp.log_likelihood,
                                  s->N_theta * sizeof(float), cudaMemcpyDeviceToDevice));
        }

        /* Fixed-lag parameters */
        int t_cp_use = -1;
        const float *cp_h = nullptr, *cp_lw = nullptr, *cp_ll = nullptr;
        if (s->fixed_lag_L > 0 && s->t_checkpoint >= 0) {
            int steps = s->t_current - s->t_checkpoint;
            if (steps > 0 && steps <= 2 * s->fixed_lag_L) {
                t_cp_use = s->t_checkpoint;
                cp_h  = s->d_checkpoint_h;
                cp_lw = s->d_checkpoint_log_w;
                cp_ll = s->d_checkpoint_ll;
            }
        }

        /* Adaptive covariance */
        update_adaptive_covariance(s);

        /* CPMMH rejuvenation */
        #define DISPATCH_CPMMH(N) \
            kernel_cpmmh_rejuvenate_impl<N><<<s->N_theta, N, cpmmh_smem<N>()>>>( \
                s->d_particles, s->d_particles_temp, s->d_y_history, \
                cn, on, cu, ou, \
                s->t_current, s->N_theta, s->noise_capacity, s->cpmmh_rho, \
                s->d_accepts, s->d_swap_flags, \
                t_cp_use, cp_h, cp_lw, cp_ll)

        for (int k = 0; k < s->K_rejuv; k++) {
            int ha = 0;
            CUDA_CHECK(cudaMemcpy(s->d_accepts, &ha, sizeof(int), cudaMemcpyHostToDevice));

            float* cn = s->d_h_noise[s->noise_buf];
            float* on = s->d_h_noise[1 - s->noise_buf];
            float* cu = s->d_u0_noise[s->noise_buf];
            float* ou = s->d_u0_noise[1 - s->noise_buf];

            switch (s->N_inner) {
                case 64:  DISPATCH_CPMMH(64);  break;
                case 128: DISPATCH_CPMMH(128); break;
                case 256: DISPATCH_CPMMH(256); break;
                case 512: DISPATCH_CPMMH(512); break;
                default: exit(1);
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            int t_start = (t_cp_use >= 0) ? (t_cp_use + 1) : 0;
            kernel_commit_noise<<<s->N_theta, s->N_inner>>>(
                cn, on, cu, ou, s->d_swap_flags,
                s->N_theta, s->N_inner, s->t_current, s->noise_capacity, t_start);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(&ha, s->d_accepts, sizeof(int), cudaMemcpyDeviceToHost));
            s->n_rejuv_accepts += ha;
            s->n_rejuv_total += s->N_theta;
        }
        #undef DISPATCH_CPMMH

        /* Reset weights */
        kernel_reset_outer_weights<<<(s->N_theta + 255) / 256, 256>>>(s->d_particles, s->N_theta);
        CUDA_CHECK(cudaDeviceSynchronize());

        kernel_compute_outer_ess<<<1, s->N_theta, 32 * sizeof(float)>>>(
            s->d_particles, s->d_ess, s->N_theta);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_ess, s->d_ess, sizeof(float), cudaMemcpyDeviceToHost));
    }

    /* Fixed-lag: save checkpoint */
    if (s->fixed_lag_L > 0) {
        int target = (s->t_current / s->fixed_lag_L) * s->fixed_lag_L;
        if (target > s->t_checkpoint && s->t_current > 0) {
            kernel_save_checkpoint<<<s->N_theta, s->N_inner>>>(
                s->d_particles, s->d_checkpoint_h, s->d_checkpoint_log_w,
                s->d_checkpoint_ll, s->N_theta, s->N_inner);
            CUDA_CHECK(cudaDeviceSynchronize());
            s->t_checkpoint = target;
        }
    }

    return h_ess;
}

float smc2_bpf_learn_window(SMC2BPFState* s, const float* y, int T) {
    smc2_bpf_init(s);
    float ess = 0.0f;
    for (int t = 0; t < T; t++) ess = smc2_bpf_update(s, y[t]);
    return ess;
}

void smc2_bpf_get_theta_mean(SMC2BPFState* s, float out[SMC2_N_PARAMS]) {
    float* hw = (float*)malloc(s->N_theta * sizeof(float));
    float* hp[3];
    for (int i = 0; i < 3; i++) hp[i] = (float*)malloc(s->N_theta * sizeof(float));

    CUDA_CHECK(cudaMemcpy(hw,    s->d_particles.weight,  s->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hp[0], s->d_particles.rho,     s->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hp[1], s->d_particles.sigma_z, s->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hp[2], s->d_particles.mu,      s->N_theta * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 3; i++) out[i] = 0.0f;
    for (int j = 0; j < s->N_theta; j++)
        for (int i = 0; i < 3; i++) out[i] += hw[j] * hp[i][j];

    free(hw);
    for (int i = 0; i < 3; i++) free(hp[i]);
}

void smc2_bpf_get_theta_std(SMC2BPFState* s, float out[SMC2_N_PARAMS]) {
    float mean[3];
    smc2_bpf_get_theta_mean(s, mean);

    float* hw = (float*)malloc(s->N_theta * sizeof(float));
    float* hp[3];
    for (int i = 0; i < 3; i++) hp[i] = (float*)malloc(s->N_theta * sizeof(float));

    CUDA_CHECK(cudaMemcpy(hw,    s->d_particles.weight,  s->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hp[0], s->d_particles.rho,     s->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hp[1], s->d_particles.sigma_z, s->N_theta * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hp[2], s->d_particles.mu,      s->N_theta * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 3; i++) out[i] = 0.0f;
    for (int j = 0; j < s->N_theta; j++)
        for (int i = 0; i < 3; i++) {
            float d = hp[i][j] - mean[i];
            out[i] += hw[j] * d * d;
        }
    for (int i = 0; i < 3; i++) out[i] = sqrtf(out[i]);

    free(hw);
    for (int i = 0; i < 3; i++) free(hp[i]);
}

float smc2_bpf_get_outer_ess(SMC2BPFState* s) {
    float e;
    CUDA_CHECK(cudaMemcpy(&e, s->d_ess, sizeof(float), cudaMemcpyDeviceToHost));
    return e;
}
