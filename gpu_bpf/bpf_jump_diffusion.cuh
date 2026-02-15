/**
 * @file bpf_jump_diffusion.cuh
 * @brief Minimal Jump-Diffusion Perturbation for BPF
 *
 * Fixed lambda, fixed sigma_J. No learning. Just the kernel.
 *
 * State transition becomes:
 *   h_t = mu + rho*(h_{t-1} - mu) + sigma_z*eps + J_t*sigma_J*eta
 *   J_t ~ Bernoulli(lambda), eta ~ N(0,1)
 *
 * Integration: call jump_perturb() on d_h AFTER propagation, BEFORE weighting.
 * That's it. One extra kernel launch per tick.
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
    float* d_jump_ind;          /**< [N] jump indicators (for diagnostics) */
} JumpState;

/** Create jump state. Allocates device memory. */
JumpState* jump_create(int n_particles, float lambda, float sigma_J,
                       int seed, cudaStream_t stream);

/** Free device memory. */
void jump_destroy(JumpState* js);

/**
 * Apply jump perturbation to propagated h values.
 * Call AFTER propagation kernel, BEFORE weight kernel.
 */
void jump_perturb(JumpState* js, float* d_h, cudaStream_t stream);

/** Setters for parameter sweeps */
void jump_set_lambda(JumpState* js, float lambda);
void jump_set_sigma_J(JumpState* js, float sigma_J);

#ifdef __cplusplus
}
#endif

#endif /* BPF_JUMP_DIFFUSION_CUH */
