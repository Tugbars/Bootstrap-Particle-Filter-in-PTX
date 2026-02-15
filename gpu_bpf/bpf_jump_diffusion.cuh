/**
 * @file bpf_jump_diffusion.cuh
 * @brief Mixture Innovation Model for BPF — Bernoulli Jump Perturbation
 *
 * Each tick, every particle independently draws J_t ~ Bernoulli(lambda).
 * If J_t = 1:  h[i] += sigma_J * N(0,1)
 * If J_t = 0:  h[i] unchanged (normal diffusion only)
 *
 * Implements a two-component MIM where resampling performs posterior
 * component selection. Jumpers are uniformly distributed across all
 * particle indices, so they naturally overlap with adaptive sigma_z
 * bands — no placement tuning needed.
 *
 * Integration: call jump_perturb() on d_h AFTER propagation, BEFORE weighting.
 * One extra kernel launch per tick, full N threads.
 */

#ifndef BPF_JUMP_DIFFUSION_CUH
#define BPF_JUMP_DIFFUSION_CUH

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct JumpState {
    int    n_particles;
    float  lambda;              /**< Jump probability per particle per tick */
    float  sigma_J;             /**< Jump size std dev                     */
    unsigned int* d_seeds;      /**< [N] per-particle PRNG seeds           */
} JumpState;

/** Create jump state. Allocates device memory for N particles. */
JumpState* jump_create(int n_particles, float lambda, float sigma_J,
                       int seed, cudaStream_t stream);

/** Free device memory. */
void jump_destroy(JumpState* js);

/**
 * Apply jump perturbation to propagated h values.
 * Call AFTER propagation kernel, BEFORE weight kernel.
 * Launches ceil(N/256) blocks — full particle count.
 */
void jump_perturb(JumpState* js, float* d_h, cudaStream_t stream);

/** Setters for parameter sweeps */
void jump_set_lambda(JumpState* js, float lambda);
void jump_set_sigma_J(JumpState* js, float sigma_J);

#ifdef __cplusplus
}
#endif

#endif /* BPF_JUMP_DIFFUSION_CUH */
