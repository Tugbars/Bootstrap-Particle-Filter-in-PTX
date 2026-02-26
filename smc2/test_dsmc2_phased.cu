/*═══════════════════════════════════════════════════════════════════════════════
 * @file test_dsmc2_phased.cu
 * @brief End-to-end test of the dSMC² pipeline with phased parameter learning
 *
 * Generates a ~60K tick multi-regime DGP that exercises all 3 phases:
 *
 *   Segment 0: Calm warmup              (5000 ticks)  — Phase 1, low z
 *   Segment 1: Moderate stress           (3000 ticks)  — z rises
 *   Segment 2: Recovery                  (4000 ticks)  — z falls back
 *   Segment 3: FIRST CRISIS              (3000 ticks)  — z spikes, Phase 2→3
 *   Segment 4: Deep calm 1               (5000 ticks)  — valve should lock back
 *   Segment 5: Calm plateau              (5000 ticks)  — valve stays locked
 *   Segment 6: Spike gauntlet            (2000 ticks)  — rapid vol spikes, re-unlock
 *   Segment 7: SECOND CRISIS (deeper)    (4000 ticks)  — tests ceiling at extremes
 *   Segment 8: Post-crisis calm          (6000 ticks)  — valve locks again
 *   Segment 9: Dead calm                 (5000 ticks)  — VIX-at-12 territory
 *   Segment 10: Crypto chaos             (4000 ticks)  — t-distributed shocks
 *   Segment 11: Final calm               (5000 ticks)  — parameter accuracy check
 *
 * Four configurations compared:
 *   A. ratchet  — dSMC² with phased learning (one-way: Phase 1→2→3 only)
 *   B. valve    — dSMC² with bidirectional phased learning (can lock back)
 *   C. fixed4   — SMC² locked to 4 params forever (no ceiling/rate learning)
 *   D. all8     — SMC² with all 8 params free from tick 0 (ridge problem)
 *
 * Z source for phased controller:
 *   The phased controller needs real z-range to trigger phase transitions.
 *   The pipeline's internal phased_observe_z_from_smc2() reads z from RBPF
 *   inner particles — wrong signal (depends on parameter quality, lags badly).
 *   Instead, we steal the phased pointer and feed true DGP z from the test
 *   loop, matching the OLD working code. When BPF goes 2D, this becomes
 *   BpfResult.z_mean.
 *
 * Build:
 *   nvcc -O3 test_dsmc2_phased.cu smc2_rbpf_cuda.cu gpu_bpf_ptx_full.cu \
 *        -o test_dsmc2_phased -lcuda -lcurand
 *
 * Usage:
 *   ./test_dsmc2_phased [n_bpf_particles]
 *═══════════════════════════════════════════════════════════════════════════════*/

#include "smc2_bpf_pipeline.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>

/* ── PRNG ────────────────────────────────────────────────────────────────── */

static unsigned int g_seed = 42;

static inline float randf() {
    g_seed = g_seed * 1103515245 + 12345;
    return (float)((g_seed >> 16) & 0x7FFF) / 32768.0f;
}

static inline float randn() {
    float u1 = randf() + 1e-10f;
    float u2 = randf();
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

static inline float rand_t(float nu) {
    if (nu <= 0.0f || nu > 100.0f) return randn();
    float z = randn();
    float chi2 = 0.0f;
    for (int k = 0; k < (int)nu; k++) { float g = randn(); chi2 += g * g; }
    return z * sqrtf(nu / fmaxf(chi2, 1e-8f));
}

/* ── DGP: 2-layer SV with sat_exp curves (matching the RBPF model) ──────
 *
 * True model:
 *   z̃_t = ρ · z̃_{t-1} + σ_z · ε_t          (z̃ ∈ ℝ, ε ~ N(0,1))
 *   z_t  = 1.5 · (1 + tanh(z̃_t))             (z ∈ [0, 3])
 *   μ(z) = μ_base + μ_scale · (1 - exp(-μ_rate · z))
 *   σ_h(z) = σ_base_h + σ_scale_h · (1 - exp(-σ_rate_h · z))
 *   h_t  = μ(z_t) + φ · (h_{t-1} - μ(z_t)) + σ_h(z_t) · η_t
 *   y_t  = exp(h_t / 2) · ξ_t                 (ξ ~ N(0,1) or t(ν))
 *
 * For testing, we simplify to a 1-layer SV where z̃ is the only latent:
 *   h_t = μ(z_t)  (perfect curve, no additional vol-of-vol noise)
 * This isolates the curve learning from state filtering noise.
 * ──────────────────────────────────────────────────────────────────────── */

struct TrueDGP {
    /* z̃ dynamics */
    float rho;
    float sigma_z;

    /* μ(z) curve: floor + scale · (1 - exp(-rate · z)) */
    float mu_floor;
    float mu_ceiling;
    float mu_rate;

    /* σ_h(z) curve */
    float sigma_h_floor;
    float sigma_h_ceiling;
    float sigma_h_rate;

    /* Observation model */
    float nu_obs;       /* 0 = Gaussian, >0 = Student-t */
};

static TrueDGP default_dgp() {
    TrueDGP d;
    d.rho            = 0.98f;
    d.sigma_z        = 0.15f;

    d.mu_floor       = -4.5f;    /* calm: exp(-4.5/2) ≈ 10.5% annualized vol */
    d.mu_ceiling     = -1.5f;    /* crisis: exp(-1.5/2) ≈ 47% vol */
    d.mu_rate        = 0.5f;

    d.sigma_h_floor  = 0.1f;
    d.sigma_h_ceiling = 0.6f;
    d.sigma_h_rate   = 0.3f;

    d.nu_obs         = 0.0f;     /* Gaussian obs */
    return d;
}

static inline float sat_exp(float base, float scale, float rate, float z) {
    return base + scale * (1.0f - expf(-rate * z));
}

/* ── Segment specification ───────────────────────────────────────────────── */

struct Segment {
    const char* name;
    int         ticks;
    float       rho_override;      /* -1 = use default */
    float       sigma_z_override;  /* -1 = use default */
    float       z_bias;            /* Added to z̃ to force regime */
    float       nu_state;          /* >0 = t-distributed state noise */
    float       nu_obs;            /* >0 = t-distributed obs noise */
    int         spike_count;       /* Random spikes injected */
    float       spike_mag;         /* Spike magnitude in z̃ units */
};

/* ── Generate data ───────────────────────────────────────────────────────── */

struct GeneratedData {
    std::vector<float> returns;
    std::vector<float> true_h;
    std::vector<float> true_z;
    std::vector<int>   segment_starts;
    std::vector<std::string> segment_names;
};

static GeneratedData generate_multregime(const TrueDGP& dgp,
                                          const std::vector<Segment>& segments) {
    GeneratedData gd;
    float z_tilde = 0.0f;
    float h = dgp.mu_floor;

    for (size_t s = 0; s < segments.size(); s++) {
        const Segment& seg = segments[s];
        gd.segment_starts.push_back((int)gd.returns.size());
        gd.segment_names.push_back(seg.name);

        float rho   = (seg.rho_override > 0)     ? seg.rho_override   : dgp.rho;
        float sig_z = (seg.sigma_z_override > 0)  ? seg.sigma_z_override : dgp.sigma_z;

        for (int t = 0; t < seg.ticks; t++) {
            /* State noise */
            float eps = (seg.nu_state > 0) ? rand_t(seg.nu_state) : randn();
            z_tilde = rho * z_tilde + sig_z * eps + seg.z_bias * (1.0f - rho);

            /* Random spikes */
            if (seg.spike_count > 0 && randf() < (float)seg.spike_count / seg.ticks) {
                z_tilde += seg.spike_mag * (randf() > 0.5f ? 1.0f : -1.0f);
            }

            /* Map to z */
            float z = 1.5f * (1.0f + tanhf(z_tilde));

            /* Evaluate curves */
            float mu_scale = dgp.mu_ceiling - dgp.mu_floor;
            float sh_scale = dgp.sigma_h_ceiling - dgp.sigma_h_floor;

            float mu_z    = sat_exp(dgp.mu_floor,       mu_scale, dgp.mu_rate,      z);
            float sigma_h = sat_exp(dgp.sigma_h_floor,  sh_scale, dgp.sigma_h_rate, z);

            /* h dynamics (AR(1) around mu(z)) */
            float phi = 0.8f;  /* Local persistence of h around mu(z) */
            h = mu_z + phi * (h - mu_z) + sigma_h * randn();

            /* Observation */
            float nu = (seg.nu_obs > 0) ? seg.nu_obs : dgp.nu_obs;
            float xi = (nu > 0) ? rand_t(nu) : randn();
            float y = expf(h * 0.5f) * xi;

            gd.returns.push_back(y);
            gd.true_h.push_back(h);
            gd.true_z.push_back(z);
        }
    }

    return gd;
}

/* ── Metrics per segment ─────────────────────────────────────────────────── */

struct SegmentMetrics {
    double rmse;
    double bias;
    double max_err;
    float  z_min, z_max, z_mean;
    int    ticks;
};

static SegmentMetrics compute_segment_metrics(
    const float* est_h, const float* true_h, const float* true_z,
    int start, int end
) {
    SegmentMetrics m = {};
    m.ticks = end - start;
    m.z_min = 1e6f;
    m.z_max = -1e6f;

    double sum_sq = 0, sum_bias = 0, worst = 0, z_sum = 0;
    int count = 0;

    for (int t = start; t < end; t++) {
        if (std::isnan(est_h[t]) || std::isinf(est_h[t])) continue;
        double err = (double)est_h[t] - (double)true_h[t];
        sum_sq += err * err;
        sum_bias += err;
        if (fabs(err) > worst) worst = fabs(err);

        float z = true_z[t];
        if (z < m.z_min) m.z_min = z;
        if (z > m.z_max) m.z_max = z;
        z_sum += z;
        count++;
    }

    if (count > 0) {
        m.rmse    = sqrt(sum_sq / count);
        m.bias    = sum_bias / count;
        m.max_err = worst;
        m.z_mean  = (float)(z_sum / count);
    }
    return m;
}

/* ── Run modes ───────────────────────────────────────────────────────────── */

enum TestMode {
    MODE_PHASED,        /* dSMC² with phased learning (one-way ratchet) */
    MODE_PHASED_BIDIR,  /* dSMC² with bidirectional valve */
    MODE_FIXED4,        /* 4-param locked forever */
    MODE_ALL8           /* All 8 from tick 0 */
};

static const char* mode_name(TestMode m) {
    switch (m) {
        case MODE_PHASED:       return "Ratchet";
        case MODE_PHASED_BIDIR: return "Valve";
        case MODE_FIXED4:       return "Fixed4";
        case MODE_ALL8:         return "All8";
    }
    return "?";
}

/* ── Run one configuration ───────────────────────────────────────────────── */

struct RunResult {
    std::vector<float>          est_h;
    std::vector<SegmentMetrics> seg_metrics;
    double                      total_rmse;
    int                         final_phase;
    int                         phase2_tick;
    int                         phase3_tick;
};

static RunResult run_pipeline(
    const GeneratedData& gd,
    const TrueDGP& dgp,
    int n_bpf,
    int n_theta,
    int n_inner,
    TestMode mode,
    int seed
) {
    RunResult result;
    result.final_phase = 1;
    result.phase2_tick = -1;
    result.phase3_tick = -1;
    int N = (int)gd.returns.size();
    result.est_h.resize(N, 0.0f);

    /* ── Create SMC² ─────────────────────────────────────────────────── */
    SMC2StateCUDA* smc2 = smc2_cuda_alloc(n_theta, n_inner);

    /* Set priors centered on true values but with uncertainty */
    smc2->prior.rho_mean         = dgp.rho;            smc2->prior.rho_std         = 0.05f;
    smc2->prior.sigma_total_mean = dgp.sigma_z * 1.5f; smc2->prior.sigma_total_std = 0.1f;
    smc2->prior.r_split_mean     = 0.5f;               smc2->prior.r_split_std     = 0.2f;
    smc2->prior.mu_base_mean     = dgp.mu_floor;        smc2->prior.mu_base_std     = 1.0f;
    smc2->prior.mu_scale_mean    = dgp.mu_ceiling - dgp.mu_floor;
    smc2->prior.mu_scale_std     = 1.5f;
    smc2->prior.mu_rate_mean     = 0.5f;                smc2->prior.mu_rate_std     = 0.3f;
    smc2->prior.sigma_scale_mean = 0.5f;                smc2->prior.sigma_scale_std = 0.3f;
    smc2->prior.sigma_rate_mean  = 0.3f;                smc2->prior.sigma_rate_std  = 0.2f;

    /* ── Set parameter mask based on mode ────────────────────────────── */
    uint8_t mask[N_PARAMS] = {0};
    float   vals[N_PARAMS] = {0};

    float true_mu_scale    = dgp.mu_ceiling - dgp.mu_floor;
    float true_sigma_scale = dgp.sigma_h_ceiling - dgp.sigma_h_floor;

    if (mode == MODE_FIXED4) {
        /* Lock scales and rates forever */
        mask[P_MU_SCALE]    = 1; vals[P_MU_SCALE]    = true_mu_scale;
        mask[P_MU_RATE]     = 1; vals[P_MU_RATE]     = dgp.mu_rate;
        mask[P_SIGMA_SCALE] = 1; vals[P_SIGMA_SCALE] = true_sigma_scale;
        mask[P_SIGMA_RATE]  = 1; vals[P_SIGMA_RATE]  = dgp.sigma_h_rate;
    }
    /* MODE_ALL8: all zeros (nothing fixed)
     * MODE_PHASED: controller manages masks */

    smc2_cuda_set_fixed_params(smc2, mask, vals);
    smc2_cuda_init_from_prior(smc2);

    /* ── Create dBPF ─────────────────────────────────────────────────── */
    float init_mu  = dgp.mu_floor + 0.5f;  /* Slightly misspecified */
    float init_rho = dgp.rho - 0.05f;
    float init_sz  = dgp.sigma_z * 0.8f;
    GpuBpfState* bpf = gpu_bpf_create(n_bpf, init_rho, init_sz, init_mu,
                                        0.0f, 0.0f, seed);

    /* Enable Kalman learning */
    gpu_bpf_enable_mu_learning(bpf, 50,
                                5e-8f, 0.01f,    /* Q_mu, P0_mu */
                                5e-9f, 0.001f);   /* Q_rho, P0_rho */
    gpu_bpf_enable_rho_learning(bpf, 1);

    /* ── Create pipeline ─────────────────────────────────────────────── */
    PipelineConfig pc = pipeline_default_config();
    pc.window_size = 3000;
    pc.stride      = 1500;
    pc.sync_mode   = 1;   /* Synchronous for deterministic testing */

    SMC2BpfPipeline* pipe = pipeline_create(bpf, smc2, pc);

    /* Disable phased learning for non-phased modes */
    if (mode != MODE_PHASED && mode != MODE_PHASED_BIDIR) {
        phased_destroy(pipe->phased);
        pipe->phased = NULL;
    }

    /* ── Steal phased pointer — feed z from DGP, not RBPF ────────────
     *
     * The pipeline's internal phased_observe_z_from_smc2() reads z from
     * RBPF inner particles — wrong signal (depends on parameter quality,
     * lags badly early on). We steal the pointer so the pipeline doesn't
     * call its broken path, and drive phased from here with true DGP z.
     *
     * When BPF goes 2D, this becomes BpfResult.z_mean.
     * ──────────────────────────────────────────────────────────────── */
    PhasedLearner* phased = pipe->phased;
    pipe->phased = NULL;  /* prevent pipeline's broken smc2 z-read */

    /* Set backward transition flag based on mode */
    if (phased) {
        phased->config.enable_backward = (mode == MODE_PHASED_BIDIR) ? 1 : 0;
    }

    float z_min_w = 1e6f, z_max_w = -1e6f;
    double z_sum_w = 0.0; int z_cnt_w = 0;
    int prev_pushes = 0;

    /* ── Run ─────────────────────────────────────────────────────────── */
    for (int t = 0; t < N; t++) {
        BpfResult r = pipeline_step(pipe, gd.returns[t]);
        result.est_h[t] = r.h_mean;

        /* Accumulate true z for phased controller */
        if (phased) {
            float z = gd.true_z[t];
            if (z < z_min_w) z_min_w = z;
            if (z > z_max_w) z_max_w = z;
            z_sum_w += z; z_cnt_w++;

            /* Window boundary: pipeline just pushed → feed phased */
            if (pipe->n_pushes > prev_pushes) {
                phased_observe_z(phased, (float)(z_sum_w / z_cnt_w),
                                 z_min_w, z_max_w);
                phased_update(phased);
                z_min_w = 1e6f; z_max_w = -1e6f;
                z_sum_w = 0.0; z_cnt_w = 0;
                prev_pushes = pipe->n_pushes;
            }
        }
    }

    /* ── Collect phase info ──────────────────────────────────────────── */
    if (phased) {
        result.final_phase = phased->phase;
        result.phase2_tick = (phased->phase2_entered_at >= 0)
            ? phased->phase2_entered_at * pc.stride : -1;
        result.phase3_tick = (phased->phase3_entered_at >= 0)
            ? phased->phase3_entered_at * pc.stride : -1;
    }

    /* ── Compute per-segment metrics ─────────────────────────────────── */
    int n_seg = (int)gd.segment_starts.size();
    for (int s = 0; s < n_seg; s++) {
        int start = gd.segment_starts[s];
        int end   = (s + 1 < n_seg) ? gd.segment_starts[s + 1] : N;
        SegmentMetrics sm = compute_segment_metrics(
            result.est_h.data(), gd.true_h.data(), gd.true_z.data(),
            start, end);
        result.seg_metrics.push_back(sm);
    }

    /* Total RMSE */
    double sum_sq = 0; int count = 0;
    for (int t = 0; t < N; t++) {
        if (!std::isnan(result.est_h[t])) {
            double err = (double)result.est_h[t] - (double)gd.true_h[t];
            sum_sq += err * err;
            count++;
        }
    }
    result.total_rmse = (count > 0) ? sqrt(sum_sq / count) : 999.0;

    /* ── Cleanup — put phased back so pipeline_destroy frees it ──────── */
    pipe->phased = phased;
    pipeline_destroy(pipe);
    gpu_bpf_destroy(bpf);
    smc2_cuda_free(smc2);

    return result;
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(int argc, char** argv) {
    int n_bpf   = (argc > 1) ? atoi(argv[1]) : 30000;
    int n_theta = 1024;
    int n_inner = 512;
    int seed    = 42;
    g_seed      = 12345;

    TrueDGP dgp = default_dgp();

    /* ── Define the 12-segment gauntlet ─────────────────────────────── */
    /*                                                                    */
    /*  z_bias shifts the OU mean, making z̃ orbit that level.            */
    /*  Positive z_bias → high z → crisis. Zero → calm.                  */
    /*                                                                    */
    std::vector<Segment> segments = {
        /* name               ticks  rho   sig_z  z_bias nu_s nu_o spk  mag */
        {"Calm warmup",       10000, -1,    0.08f,-0.5f,  0,   0,   0,   0  },
        {"Moderate stress",    3000, -1,   -1,    1.0f,  0,   0,   0,   0  },
        {"Recovery 1",         4000, -1,   -1,    0.0f,  0,   0,   0,   0  },
        {"CRISIS 1",           3000, -1,    0.25f, 2.5f, 0,   0,   2,   1.5f},
        {"Deep calm 1",       10000, -1,    0.06f,-1.5f,  0,   0,   0,   0  },
        {"Calm plateau",       8000, -1,    0.06f,-1.0f,  0,   0,   0,   0  },
        {"Spike gauntlet",     2000, -1,    0.20f, 0.5f, 0,   0,   8,   2.0f},
        {"CRISIS 2 (deep)",    4000, -1,    0.30f, 3.0f, 0,   0,   3,   1.0f},
        {"Post-crisis calm",  10000, -1,    0.06f,-1.5f,  0,   0,   0,   0  },
        {"Dead calm",          8000, -1,    0.04f,-2.0f,  0,   0,   0,   0  },
        {"Crypto chaos",       4000, -1,    0.25f, 1.5f, 3,   3,   5,   2.5f},
        {"Final calm",         8000, -1,    0.06f,-1.0f,  0,   0,   0,   0  },
    };

    GeneratedData gd = generate_multregime(dgp, segments);
    int N = (int)gd.returns.size();

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║   dSMC² Phased Learning — End-to-End Test                           ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════════╣\n");
    printf("║  Total ticks: %d                                                 ║\n", N);
    printf("║  BPF particles: %dK  SMC² θ: %d × %d                           ║\n",
           n_bpf / 1000, n_theta, n_inner);
    printf("║  True DGP: ρ=%.2f σ_z=%.2f μ∈[%.1f,%.1f] rate=%.1f             ║\n",
           dgp.rho, dgp.sigma_z, dgp.mu_floor, dgp.mu_ceiling, dgp.mu_rate);
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n\n");

    /* Print data summary */
    printf("  Segment layout:\n");
    printf("  %-25s %8s %8s %8s %8s\n", "Name", "Ticks", "z_min", "z_max", "z_mean");
    printf("  ─────────────────────── ──────── ──────── ──────── ────────\n");
    int n_seg = (int)segments.size();
    for (int s = 0; s < n_seg; s++) {
        int start = gd.segment_starts[s];
        int end   = (s + 1 < n_seg) ? gd.segment_starts[s + 1] : N;
        float zmin = 1e6f, zmax = -1e6f, zsum = 0;
        for (int t = start; t < end; t++) {
            if (gd.true_z[t] < zmin) zmin = gd.true_z[t];
            if (gd.true_z[t] > zmax) zmax = gd.true_z[t];
            zsum += gd.true_z[t];
        }
        printf("  %-25s %8d %8.2f %8.2f %8.2f\n",
               segments[s].name, end - start, zmin, zmax, zsum / (end - start));
    }
    printf("\n");

    /* ── Run all four modes ──────────────────────────────────────────── */
    TestMode modes[] = {MODE_PHASED, MODE_PHASED_BIDIR, MODE_FIXED4, MODE_ALL8};
    RunResult results[4];

    for (int m = 0; m < 4; m++) {
        printf("  Running %s...\n", mode_name(modes[m]));
        results[m] = run_pipeline(gd, dgp, n_bpf, n_theta, n_inner, modes[m], seed);
        printf("    Total RMSE: %.4f", results[m].total_rmse);
        if (modes[m] == MODE_PHASED || modes[m] == MODE_PHASED_BIDIR) {
            printf("  (final phase: %d", results[m].final_phase);
            if (results[m].phase2_tick >= 0) printf(", P2 at tick %d", results[m].phase2_tick);
            if (results[m].phase3_tick >= 0) printf(", P3 at tick %d", results[m].phase3_tick);
            printf(")");
        }
        printf("\n");
    }

    /* ── Per-segment comparison table ─────────────────────────────────── */
    printf("\n");
    printf("  ════════════════════════════════════════════════════════════════════════════════════════════\n");
    printf("  Per-segment RMSE comparison\n");
    printf("  ────────────────────────────────────────────────────────────────────────────────────────────\n");
    printf("  %-25s %6s %6s %6s | %8s %8s %8s %8s | Winner\n",
           "Segment", "z_min", "z_max", "z_avg", "Ratchet", "Valve", "Fixed4", "All8");
    printf("  ─────────────────────── ────── ────── ────── | ──────── ──────── ──────── ──────── | ──────\n");

    int wins[4] = {0, 0, 0, 0};

    for (int s = 0; s < n_seg; s++) {
        SegmentMetrics* sm[4] = {
            &results[0].seg_metrics[s],
            &results[1].seg_metrics[s],
            &results[2].seg_metrics[s],
            &results[3].seg_metrics[s]
        };

        /* Find winner */
        int best = 0;
        for (int m = 1; m < 4; m++) {
            if (sm[m]->rmse < sm[best]->rmse) best = m;
        }
        wins[best]++;

        const char* winner_names[] = {"Ratchet", "Valve", "Fixed4", "All8"};

        printf("  %-25s %6.2f %6.2f %6.2f | %8.4f %8.4f %8.4f %8.4f | %s\n",
               gd.segment_names[s].c_str(),
               sm[0]->z_min, sm[0]->z_max, sm[0]->z_mean,
               sm[0]->rmse, sm[1]->rmse, sm[2]->rmse, sm[3]->rmse,
               winner_names[best]);
    }

    /* ── Grand summary ───────────────────────────────────────────────── */
    printf("\n");
    printf("  ════════════════════════════════════════════════════════════════════════════════════════════\n");
    printf("  GRAND SUMMARY\n");
    printf("  ────────────────────────────────────────────────────────────────────────────────────────────\n");
    printf("  %-20s %12s %12s %12s %12s\n", "", "Ratchet", "Valve", "Fixed4", "All8");
    printf("  %-20s %12.4f %12.4f %12.4f %12.4f\n", "Total RMSE",
           results[0].total_rmse, results[1].total_rmse,
           results[2].total_rmse, results[3].total_rmse);
    printf("  %-20s %12d %12d %12d %12d\n", "Segment wins",
           wins[0], wins[1], wins[2], wins[3]);

    printf("\n  Key comparisons:\n");
    printf("    Valve vs Ratchet:  %+.1f%% RMSE  (%s)\n",
           100.0 * (results[1].total_rmse / results[0].total_rmse - 1.0),
           results[1].total_rmse < results[0].total_rmse ? "valve helps" : "ratchet better");
    printf("    Ratchet vs Fixed4: %+.1f%% RMSE\n",
           100.0 * (results[0].total_rmse / results[2].total_rmse - 1.0));
    printf("    Valve vs Fixed4:   %+.1f%% RMSE\n",
           100.0 * (results[1].total_rmse / results[2].total_rmse - 1.0));
    printf("    All8 vs Fixed4:    %+.1f%% RMSE\n",
           100.0 * (results[3].total_rmse / results[2].total_rmse - 1.0));

    /* Phase transition details for both phased modes */
    for (int m = 0; m < 2; m++) {
        printf("\n  %s transitions:\n", mode_name(modes[m]));
        if (results[m].phase2_tick >= 0) {
            printf("    Phase 1→2 (ceilings unlocked): tick %d\n", results[m].phase2_tick);
        }
        if (results[m].phase3_tick >= 0) {
            printf("    Phase 2→3 (rates unlocked):    tick %d\n", results[m].phase3_tick);
        } else {
            printf("    Phase 2→3: never reached\n");
        }
        printf("    Final phase: %d\n", results[m].final_phase);
    }

    printf("\n  Expected behavior:\n");
    printf("    • Ratchet ≤ Fixed4 — phased learning adapts to regimes\n");
    printf("    • Valve ≤ Ratchet on calm segments — locks out unidentifiable params\n");
    printf("    • Valve final_phase < Ratchet final_phase — valve retreats during calm\n");
    printf("    • All8 worst early — ridge degeneracy before identification\n");
    printf("  ════════════════════════════════════════════════════════════════════════════════════════════\n\n");

    return 0;
}
