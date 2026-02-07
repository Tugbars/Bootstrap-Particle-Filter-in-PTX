/**
 * @file gpu_bpf_full.cuh
 * @brief GPU Bootstrap PF — Pure PTX edition
 *
 * Identical API to gpu_bpf.cuh but:
 *   - PCG32 PRNG instead of curandState (16 bytes vs ~48 bytes per particle)
 *   - All 13 BPF kernels in hand-written PTX
 *   - Zero cuRAND dependency for BPF
 *   - APF/IMM still use cuRAND (nvcc compiled)
 *
 * The BPF struct uses uint64_t* for RNG state instead of curandState*.
 * APF struct is unchanged (still needs cuRAND).
 */

#ifndef GPU_BPF_FULL_CUH
#define GPU_BPF_FULL_CUH

#include <curand_kernel.h>   // Still needed for APF
#include <stdint.h>

#define IMM_MAX_MODELS 64

// =============================================================================
// Common result type
// =============================================================================

typedef struct {
    float h_mean;
    float log_lik;
} BpfResult;

// =============================================================================
// Bootstrap PF — Pure PTX (PCG32 RNG)
// =============================================================================

typedef struct {
    float* d_h;
    float* d_h2;
    float* d_log_w;
    float* d_w;
    float* d_cdf;
    float* d_wh;
    uint64_t* d_rng;         // PCG32: 2 x u64 per particle [state, inc]
    float* d_scalars;        // [4]: max_lw, sum_w, h_est, log_lik
    float* d_noise;          // [N]: standard normal samples for Silverman
    float* d_var;            // [1]: variance accumulator
    float* d_chi2;           // [N]: pre-generated chi2 variates (or NULL)
    float* d_chi2_normals;   // [N*nu_int]: temp normals for chi2 gen
    void* curand_gen;        // curandGenerator_t (void* for header compat)
    float C_obs;             // precomputed lgamma constant for Student-t obs
    int nu_int;              // integer nu_state for chi2 generation
    int n_particles;
    int block, grid;
    cudaStream_t stream;
    float rho, sigma_z, mu, nu_state, nu_obs;
    float silverman_shrink;  // 0.0 = off, 0.5 = conservative
    unsigned long long host_rng_state;
    int timestep;
} GpuBpfState;

GpuBpfState* gpu_bpf_create(int n_particles, float rho, float sigma_z, float mu,
                              float nu_state, float nu_obs, int seed);
void gpu_bpf_destroy(GpuBpfState* state);
BpfResult gpu_bpf_step(GpuBpfState* state, float y_t);
void gpu_bpf_step_async(GpuBpfState* state, float y_t);
BpfResult gpu_bpf_get_result(GpuBpfState* state);

double gpu_bpf_run_rmse(
    const double* returns, const double* true_h, int n_ticks,
    int n_particles, float rho, float sigma_z, float mu,
    float nu_state, float nu_obs, int seed
);

// =============================================================================
// Auxiliary Particle Filter (unchanged — still cuRAND)
// =============================================================================

typedef struct {
    float* d_h;
    float* d_h2;
    float* d_mu_pred;
    float* d_log_v;
    float* d_v;
    float* d_log_w;
    float* d_w;
    float* d_cdf;
    float* d_wh;
    float* d_mu_pred2;
    curandState* d_rng;
    float* d_scalars;
    int n_particles;
    int block, grid;
    cudaStream_t stream;
    float rho, sigma_z, mu, nu_state, nu_obs;
    unsigned long long host_rng_state;
    int timestep;
} GpuApfState;

GpuApfState* gpu_apf_create(int n_particles, float rho, float sigma_z, float mu,
                              float nu_state, float nu_obs, int seed);
void gpu_apf_destroy(GpuApfState* state);
BpfResult gpu_apf_step(GpuApfState* state, float y_t);

double gpu_apf_run_rmse(
    const double* returns, const double* true_h, int n_ticks,
    int n_particles, float rho, float sigma_z, float mu,
    float nu_state, float nu_obs, int seed
);

// =============================================================================
// IMM — Interacting Multiple Model
// =============================================================================

typedef struct {
    float rho, sigma_z, mu, nu_state, nu_obs;
} ImmModelParams;

typedef struct {
    float h_mean, vol, log_lik;
    int best_model;
    float best_prob;
} ImmResult;

typedef struct {
    GpuBpfState** filters;
    int n_models, n_particles_per_model;
    double *log_pi, *log_pi_pred, *log_T;
    int timestep;
} GpuImmState;

GpuImmState* gpu_imm_create(const ImmModelParams* models, int n_models,
                              int n_particles_per_model,
                              const float* transition_matrix, int seed);
ImmResult gpu_imm_step(GpuImmState* state, float y_t);
float gpu_imm_get_prob(const GpuImmState* state, int k);
void gpu_imm_get_probs(const GpuImmState* state, float* probs_out);
void gpu_imm_destroy(GpuImmState* state);

ImmModelParams* gpu_imm_build_grid(
    const float* rhos, int n_rho, const float* sigma_zs, int n_sigma,
    const float* mus, int n_mu, float nu_state, float nu_obs, int* out_n_models
);

#endif // GPU_BPF_FULL_CUH
