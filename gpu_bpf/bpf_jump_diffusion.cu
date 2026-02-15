/**
 * @file bpf_jump_diffusion.cu
 * @brief Minimal Jump-Diffusion — CUDA kernel + host logic
 */

#include "bpf_jump_diffusion.cuh"
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * Device: LCG PRNG
 * ═══════════════════════════════════════════════════════════════════════════ */

__device__ __forceinline__
unsigned int jd_lcg(unsigned int s) {
    return s * 1103515245u + 12345u;
}

__device__ __forceinline__
float jd_uniform(unsigned int s) {
    return (float)((s >> 16) & 0x7FFF) / 32768.0f;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Kernel: Jump Perturbation
 *
 * Per particle:
 *   1. Bernoulli(lambda) from LCG
 *   2. If jump: Box-Muller normal, h[i] += sigma_J * normal
 * ═══════════════════════════════════════════════════════════════════════════ */

__global__ void jump_perturb_kernel(
    float*        d_h,
    float*        d_jump_ind,
    unsigned int* d_seeds,
    float         lambda,
    float         sigma_J,
    int           N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    unsigned int seed = d_seeds[tid];

    /* Bernoulli draw */
    seed = jd_lcg(seed);
    float u = jd_uniform(seed);

    if (u < lambda) {
        /* Box-Muller normal */
        seed = jd_lcg(seed);
        float u1 = jd_uniform(seed) + 1e-10f;
        seed = jd_lcg(seed);
        float u2 = jd_uniform(seed);
        float eta = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);

        d_h[tid] += sigma_J * eta;
        d_jump_ind[tid] = 1.0f;
    } else {
        d_jump_ind[tid] = 0.0f;
    }

    d_seeds[tid] = seed;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Host: Create / Destroy
 * ═══════════════════════════════════════════════════════════════════════════ */

JumpState* jump_create(int n_particles, float lambda, float sigma_J,
                       int seed, cudaStream_t stream) {
    JumpState* js = (JumpState*)calloc(1, sizeof(*js));
    js->n_particles = n_particles;
    js->lambda      = lambda;
    js->sigma_J     = sigma_J;

    cudaMalloc(&js->d_seeds,    n_particles * sizeof(unsigned int));
    cudaMalloc(&js->d_jump_ind, n_particles * sizeof(float));

    /* Initialize per-particle seeds */
    unsigned int* h_seeds = (unsigned int*)malloc(n_particles * sizeof(unsigned int));
    unsigned int s = (unsigned int)seed;
    for (int i = 0; i < n_particles; i++) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        h_seeds[i] = s;
    }
    cudaMemcpyAsync(js->d_seeds, h_seeds, n_particles * sizeof(unsigned int),
                    cudaMemcpyHostToDevice, stream);
    free(h_seeds);

    return js;
}

void jump_destroy(JumpState* js) {
    if (!js) return;
    cudaFree(js->d_seeds);
    cudaFree(js->d_jump_ind);
    free(js);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Host: Perturb
 * ═══════════════════════════════════════════════════════════════════════════ */

void jump_perturb(JumpState* js, float* d_h, cudaStream_t stream) {
    if (!js) return;
    int threads = 256;
    int blocks  = (js->n_particles + threads - 1) / threads;

    jump_perturb_kernel<<<blocks, threads, 0, stream>>>(
        d_h, js->d_jump_ind, js->d_seeds,
        js->lambda, js->sigma_J, js->n_particles
    );
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Host: Setters
 * ═══════════════════════════════════════════════════════════════════════════ */

void jump_set_lambda(JumpState* js, float lambda) {
    if (js) js->lambda = lambda;
}

void jump_set_sigma_J(JumpState* js, float sigma_J) {
    if (js) js->sigma_J = sigma_J;
}
