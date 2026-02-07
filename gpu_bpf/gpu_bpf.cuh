/**
 * @file gpu_bpf.cuh
 * @brief GPU Bootstrap PF, Auxiliary PF, and IMM
 *
 * Layer 1a: GpuBpfState — Bootstrap SIR PF (prior proposal)
 * Layer 1b: GpuApfState — Auxiliary PF (Pitt & Shephard, predictive mean proposal)
 * Layer 2:  GpuImmState — K models with Markov switching
 */

#ifndef GPU_BPF_CUH
#define GPU_BPF_CUH

#include <curand_kernel.h>

#define IMM_MAX_MODELS 64

// =============================================================================
// Common result type
// =============================================================================

typedef struct {
    float h_mean;
    float log_lik;
} BpfResult;

// =============================================================================
// Bootstrap PF
// =============================================================================

typedef struct {
    float* d_h;
    float* d_h2;
    float* d_log_w;
    float* d_w;
    float* d_cdf;
    float* d_wh;
    curandState* d_rng;
    float* d_scalars;       // [4]: max_lw, sum_w, h_est, log_lik
    int n_particles;
    int block, grid;
    cudaStream_t stream;
    float rho, sigma_z, mu, nu_state, nu_obs;
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
// Auxiliary Particle Filter (Pitt & Shephard 1999)
//
// Per tick:
//   1. First-stage: log v_i = log p(y_t | mu_pred_i)  [predictive mean]
//   2. Resample by first-stage weights
//   3. Propagate resampled particles through transition
//   4. Second-stage: log w_i = log p(y_t | h_new_i) - log p(y_t | mu_pred_i)
//   5. Estimate: weighted mean with second-stage weights
// =============================================================================

typedef struct {
    float* d_h;
    float* d_h2;
    float* d_mu_pred;       // predictive means: mu + rho*(h - mu)
    float* d_log_v;         // first-stage log weights
    float* d_v;             // first-stage weights (exp)
    float* d_log_w;         // second-stage log weights
    float* d_w;             // second-stage weights (exp)
    float* d_cdf;
    float* d_wh;
    float* d_mu_pred2;      // resampled predictive means (for correction)
    curandState* d_rng;
    float* d_scalars;       // [4]: max, sum, h_est, log_lik
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

#endif // GPU_BPF_CUH
