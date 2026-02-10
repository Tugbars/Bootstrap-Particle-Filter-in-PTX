/**
 * @file multi_bpf.cuh
 * @brief K-Instance Bootstrap PF for volatility tracking
 *
 * K independent BPFs, each with own parameters, running on separate CUDA streams.
 * Host-side Bayesian model averaging mixes outputs by predictive likelihood.
 *
 * Each instance: predict → weight → resample. No bands, no Silverman.
 * All instances run every tick (state continuity). Mixing selects the best.
 */

#ifndef MULTI_BPF_CUH
#define MULTI_BPF_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>

#define MBPF_MAX_K 8

#ifdef __cplusplus
extern "C" {
#endif

// Parameters for one BPF instance
typedef struct {
    float rho;
    float sigma_z;
    float mu;
    float nu_state;   // state transition df (0 = Gaussian)
    float nu_obs;     // observation df (0 = Gaussian)
} MbpfParams;

// Per-tick result from one instance
typedef struct {
    float h_mean;
    float log_lik;
} MbpfInstanceResult;

// Per-tick result from the mixture
typedef struct {
    float h_mean;       // pi-weighted mean
    float vol;          // exp(h_mean / 2)
    float log_lik;      // log sum_k pi_k * p_k(y_t)
    int   best_k;       // argmax pi
    float best_prob;    // max pi
} MbpfResult;

// Internal state for one BPF instance
typedef struct {
    float* d_h;
    float* d_h2;        // swap buffer for resample
    float* d_log_w;
    float* d_w;
    float* d_cdf;
    float* d_wh;
    curandState* d_rng;
    float* d_scalars;   // [4]: max_lw, sum_w, h_est, log_lik
    cudaStream_t stream;
    MbpfParams params;
    int timestep;
} MbpfInstance;

// Top-level multi-BPF state
typedef struct {
    int K;                          // number of instances
    int n_particles;
    int block;
    int grid;
    MbpfInstance inst[MBPF_MAX_K];
    double log_pi[MBPF_MAX_K];     // mixing weights (log)
    double log_T[MBPF_MAX_K * MBPF_MAX_K];  // transition matrix (log)
    unsigned long long host_rng;
} MbpfState;

// Create / destroy
MbpfState* mbpf_create(
    int K,
    int n_particles,
    const MbpfParams* params,       // [K] parameter sets
    float p_stay,                   // transition matrix diagonal (e.g. 0.98)
    int seed
);
void mbpf_destroy(MbpfState* s);

// Run one tick — all K instances async, then mix
MbpfResult mbpf_step(MbpfState* s, float y_t);

// Update parameters for instance k (e.g. from backprop learning)
void mbpf_set_params(MbpfState* s, int k, const MbpfParams* p);

// Read mixing weights
void mbpf_get_probs(const MbpfState* s, float* probs_out);

// Batch RMSE evaluation
double mbpf_run_rmse(
    const double* returns, const double* true_h, int n_ticks,
    int K, int n_particles,
    const MbpfParams* params, float p_stay, int seed
);

#ifdef __cplusplus
}
#endif

#endif // MULTI_BPF_CUH
