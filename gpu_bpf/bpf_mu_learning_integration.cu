// =============================================================================
// HOST INTEGRATION — Online mu learning for PTX BPF
// =============================================================================
//
// This file contains all the pieces to integrate into your existing code.
// NOT a standalone file — copy sections into the right places.
//
// Changes needed:
//   1. Header: add fields to GpuBpfState
//   2. PTX loader: add kernel 14 function pointer
//   3. gpu_bpf_create: init new fields, expand d_scalars to 6
//   4. gpu_bpf_step_async: insert gradient kernel + Adam logic
//   5. New API: gpu_bpf_enable_mu_learning(), gpu_bpf_get_mu()
// =============================================================================


// ─────────────────────────────────────────────────────────────────────────────
// 1. HEADER ADDITIONS (gpu_bpf_full.cuh) — add to GpuBpfState
// ─────────────────────────────────────────────────────────────────────────────

/*
    // Online mu learning (add after existing fields)
    int     learn_mu;           // 0=off, 1=on
    int     update_K;           // Adam step every K ticks (e.g. 50)
    float   lr;                 // learning rate (e.g. 0.003)
    float   grad_clip;          // gradient clipping norm (e.g. 1.0)
    int     grad_count;         // ticks since last Adam step
    float   m_alpha;            // Adam first moment
    float   v_alpha;            // Adam second moment
    int     adam_step;          // Adam step counter
*/

// And add these API declarations:
/*
    void  gpu_bpf_enable_mu_learning(GpuBpfState* s, float lr, int update_K, float grad_clip);
    void  gpu_bpf_disable_mu_learning(GpuBpfState* s);
    float gpu_bpf_get_mu(GpuBpfState* s);
*/


// ─────────────────────────────────────────────────────────────────────────────
// 2. PTX LOADER — add kernel 14 to PtxFunctions and extract
// ─────────────────────────────────────────────────────────────────────────────

/*
    In PtxFunctions struct, add:
        CUfunction grad_alpha;

    In ensure_ptx_loaded(), after extracting kernel 13, add:
        cuModuleGetFunction(&g_ptx.grad_alpha, g_ptx_module, "bpf_grad_alpha");

    Update the log message:
        fprintf(stderr, "[PTX-FULL] 14 kernels loaded\n");
*/


// ─────────────────────────────────────────────────────────────────────────────
// 3. gpu_bpf_create — changes
// ─────────────────────────────────────────────────────────────────────────────

/*
    Change d_scalars allocation from 5 to 6 floats:
        cudaMalloc(&s->d_scalars, 6 * sizeof(float));

    Scalars layout:
        [0] max_lw
        [1] sum_w
        [2] h_est
        [3] log_lik
        [4] sum_w_sq  (ESS)
        [5] grad_alpha_accum   ← NEW

    Init learning fields (off by default):
        s->learn_mu   = 0;
        s->update_K   = 50;
        s->lr         = 0.003f;
        s->grad_clip  = 1.0f;
        s->grad_count = 0;
        s->m_alpha    = 0.0f;
        s->v_alpha    = 0.0f;
        s->adam_step  = 0;
*/


// ─────────────────────────────────────────────────────────────────────────────
// 4. gpu_bpf_step_async — insert after ESS computation, before resampling
// ─────────────────────────────────────────────────────────────────────────────

// Find this block (after step 8b ESS):
//
//     // ─── Conditional resampling decision ───
//     bool do_resample = true;
//
// Insert BEFORE it:

static void bpf_grad_and_adam(GpuBpfState* s, float y_t,
                               CUdeviceptr dh, CUdeviceptr dw,
                               CUdeviceptr dscal, CUdeviceptr dh2,
                               int n, int g, int b, cudaStream_t st,
                               size_t smem)
{
    if (!s->learn_mu) return;

    // Kernel 14: scratch[i] = w[i] * dlw_dh[i]
    // Reuse d_h2 as scratch (ESS done, resampling hasn't started)
    {
        void* params[] = { &dh2, &dw, &dh, &y_t, &s->nu_obs, &n };
        ptx_launch(g_ptx.grad_alpha, st, g, b, 0, params);
    }

    // Reduce into d_scalars[5] (device-side accumulator, NOT reset each tick)
    {
        CUdeviceptr ptr = dscal + 5 * sizeof(float);
        void* p2[] = { &dh2, &ptr, &n };
        ptx_launch(g_ptx.reduce_sum, st, g, b, smem, p2);
    }

    s->grad_count++;

    // Adam update every K ticks
    if (s->grad_count >= s->update_K) {
        // Sync + read accumulated gradient from device
        cudaStreamSynchronize(st);
        float g_alpha;
        cudaMemcpy(&g_alpha, s->d_scalars + 5, sizeof(float), cudaMemcpyDeviceToHost);

        // Mean gradient over K ticks, negate (maximize ll → minimize loss)
        float grad = -g_alpha / (float)s->grad_count;

        // Clip
        if (s->grad_clip > 0.0f) {
            float ag = fabsf(grad);
            if (ag > s->grad_clip)
                grad *= s->grad_clip / ag;
        }

        // Adam
        s->adam_step++;
        float bc1 = 1.0f - powf(0.9f,   (float)s->adam_step);
        float bc2 = 1.0f - powf(0.999f,  (float)s->adam_step);
        s->m_alpha = 0.9f   * s->m_alpha + 0.1f   * grad;
        s->v_alpha = 0.999f * s->v_alpha + 0.001f * grad * grad;
        float alpha = s->mu * (1.0f - s->rho);
        alpha -= s->lr * (s->m_alpha / bc1) / (sqrtf(s->v_alpha / bc2) + 1e-8f);

        // alpha → mu, clamp
        float mu_new = alpha / fmaxf(1.0f - s->rho, 1e-6f);
        mu_new = fmaxf(fminf(mu_new, 2.0f), -5.0f);
        s->mu = mu_new;

        // Reset accumulator on device
        float zero = 0.0f;
        cudaMemcpy(s->d_scalars + 5, &zero, sizeof(float), cudaMemcpyHostToDevice);
        s->grad_count = 0;
    }
}

// In gpu_bpf_step_async, after ESS block and before "Conditional resampling decision":
//
//     bpf_grad_and_adam(s, y_t, dh, dw, dscal, dh2, n, g, b, st, smem);
//
// NOTE: dh2 is reused as scratch. This is safe because:
//   - ESS used dh2 for w² but that reduction is complete
//   - Resampling writes to dh2 AFTER this point
//
// IMPORTANT: The Adam sync only happens every K ticks (e.g. every 50).
//   On non-Adam ticks, the only added cost is kernel 14 launch + reduce_sum
//   launch = 2 kernel launches, no sync.


// ─────────────────────────────────────────────────────────────────────────────
// 5. API FUNCTIONS
// ─────────────────────────────────────────────────────────────────────────────

void gpu_bpf_enable_mu_learning(GpuBpfState* s, float lr, int update_K, float grad_clip) {
    s->learn_mu   = 1;
    s->lr         = lr;
    s->update_K   = update_K;
    s->grad_clip  = grad_clip;
    s->grad_count = 0;
    s->m_alpha    = 0.0f;
    s->v_alpha    = 0.0f;
    s->adam_step  = 0;

    // Zero the device accumulator
    float zero = 0.0f;
    cudaMemcpy(s->d_scalars + 5, &zero, sizeof(float), cudaMemcpyHostToDevice);
}

void gpu_bpf_disable_mu_learning(GpuBpfState* s) {
    s->learn_mu = 0;
}

float gpu_bpf_get_mu(GpuBpfState* s) {
    return s->mu;
}

// Called by SMC² when it wants to override mu
// Resets Adam state so online learning starts fresh from new value
void gpu_bpf_set_mu(GpuBpfState* s, float mu) {
    s->mu = mu;
    s->m_alpha  = 0.0f;
    s->v_alpha  = 0.0f;
    s->adam_step = 0;
    s->grad_count = 0;

    float zero = 0.0f;
    cudaMemcpy(s->d_scalars + 5, &zero, sizeof(float), cudaMemcpyHostToDevice);
}


// ─────────────────────────────────────────────────────────────────────────────
// 6. COST ANALYSIS
// ─────────────────────────────────────────────────────────────────────────────
//
// Per tick (learn_mu=1):
//   +2 kernel launches: bpf_grad_alpha + bpf_reduce_sum
//   Both trivial: element-wise product and standard reduction
//   No sync added (device accumulates across ticks)
//
// Every K ticks:
//   +1 cudaStreamSynchronize (was already happening for ESS check)
//   +1 cudaMemcpy D→H (4 bytes)
//   +1 cudaMemcpy H→D (4 bytes, reset accumulator)
//   +Adam arithmetic on host (negligible)
//
// Estimated overhead: ~2-5% per tick (two small kernel launches)
//
// ─────────────────────────────────────────────────────────────────────────────
// 7. BANDS COMPATIBILITY
// ─────────────────────────────────────────────────────────────────────────────
//
// Adaptive bands scale sigma_z in kernel 3 (propagation).
// Gradient kernel 14 only reads h[i] and w[i] from the observation model.
// dlw/dh depends on y_t, h[i], nu_obs — NOT sigma_z.
// Therefore: bands and mu learning are fully orthogonal. No collision.
//
