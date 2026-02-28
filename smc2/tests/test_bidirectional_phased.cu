/*═══════════════════════════════════════════════════════════════════════════════
 * @file test_bidirectional_phased.cu
 * @brief Tests whether bidirectional phased learning preserves crisis knowledge
 *
 * The core question: after a long calm period, does the valve's saved memory
 * from a previous crisis give the cloud a head start when the next crisis hits?
 *
 * DGP structure:
 *
 *   Crisis 1 (teach)  →  Long calm (degenerate)  →  Crisis 2 (score)
 *     ~12K ticks           ~45K ticks                 ~12K ticks
 *     z ∈ [0, 2.8]         z ≈ 0                      z ∈ [0, 2.5]
 *     Ceilings learnable   Ceilings unidentifiable     Different from Crisis 1
 *
 * Three modes on identical data:
 *
 *   A. BIDIR   — Bidirectional valve. Learns ceilings in Crisis 1, locks them
 *                during calm (saving to posterior mean), unlocks at Crisis 2
 *                starting from saved values.
 *
 *   B. ONEWAY  — One-way ratchet. Learns ceilings in Crisis 1, leaves them
 *                free during calm. Cloud degenerates along ridge. Enters
 *                Crisis 2 with corrupted estimates.
 *
 *   C. PHASE1  — Always Phase 1. Never unlocks ceilings. Baseline.
 *                Cloud stays healthy but can't track crisis vol accurately.
 *
 * RMSE scored only on Crisis 2 — the moment of truth.
 *
 * Expected results:
 *   - BIDIR < ONEWAY on Crisis 2 (preserved memory beats degenerated cloud)
 *   - BIDIR < PHASE1 on Crisis 2 (learned ceilings beat prior defaults)
 *   - ONEWAY may be worse than PHASE1 if degeneracy during calm is severe
 *
 * Build:
 *   nvcc -O3 test_bidirectional_phased.cu smc2_rbpf_cuda.cu smc2_param_tracker.cu \
 *        gpu_bpf_ptx_full.cu -o test_bidirectional_phased -lcuda -lcurand
 *═══════════════════════════════════════════════════════════════════════════════*/

#include "gpu_bpf_full.cuh"
#include "smc2_phased_learning.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

/* ── PRNG ────────────────────────────────────────────────────────────────── */

struct PRNG {
    unsigned int state;
};

static inline PRNG prng_create(unsigned int seed) {
    PRNG p;
    p.state = seed ? seed : 1;
    return p;
}

static inline float prng_randf(PRNG* p) {
    p->state = p->state * 1103515245 + 12345;
    return (float)((p->state >> 16) & 0x7FFF) / 32768.0f;
}

static inline float prng_randn(PRNG* p) {
    float u1 = prng_randf(p) + 1e-10f;
    float u2 = prng_randf(p);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

/* ── DGP ─────────────────────────────────────────────────────────────────── */

struct TrueDGP {
    float rho;
    float sigma_z;
    float mu_base;
    float mu_scale;
    float mu_rate;
    float sigma_h_base;
    float sigma_h_scale;
    float sigma_h_rate;
    float theta_base;      /* θ(z) mean-reversion speed curve */
    float theta_scale;
    float theta_rate;
};

static inline float sat_exp(float base, float scale, float rate, float z) {
    return base + scale * (1.0f - expf(-rate * z));
}

static TrueDGP default_dgp() {
    TrueDGP d;
    d.rho            = 0.98f;
    d.sigma_z        = 0.15f;
    d.mu_base        = -4.5f;
    d.mu_scale       = 3.0f;
    d.mu_rate        = 0.5f;
    d.sigma_h_base   = 0.10f;
    d.sigma_h_scale  = 0.50f;
    d.sigma_h_rate   = 0.30f;
    /* θ(z) curve — must match smc2->theta_curve for model consistency.
     * h dynamics: h = (1-θ(z))·h + θ(z)·μ(z) + σ_h(z)·ε */
    d.theta_base     = 0.02f;
    d.theta_scale    = 0.08f;
    d.theta_rate     = 1.5f;
    return d;
}

/* ── Segment ─────────────────────────────────────────────────────────────── */

struct Segment {
    const char* name;
    int         ticks;
    float       z_bias;
    int         score;     /* 1 = include in RMSE */
};

/* ── Generated data ──────────────────────────────────────────────────────── */

struct GeneratedData {
    std::vector<float> returns;        /* raw r_t — for production BPF */
    std::vector<float> log_returns_sq; /* log(r_t²) — for SMC²/RBPF (OCSN) */
    std::vector<float> true_h;
    std::vector<float> true_z;
    std::vector<int>   segment_starts;
    std::vector<const char*> segment_names;
    std::vector<int>   segment_score;
    int N;
    int score_start;
};

static GeneratedData generate_data(const TrueDGP& dgp,
                                    const std::vector<Segment>& segments,
                                    PRNG* rng) {
    GeneratedData gd;
    gd.score_start = -1;

    float z_tilde = 0.0f;
    float h       = dgp.mu_base;

    for (size_t s = 0; s < segments.size(); s++) {
        const Segment& seg = segments[s];
        int start = (int)gd.returns.size();
        gd.segment_starts.push_back(start);
        gd.segment_names.push_back(seg.name);
        gd.segment_score.push_back(seg.score);

        if (seg.score && gd.score_start < 0)
            gd.score_start = start;

        for (int t = 0; t < seg.ticks; t++) {
            /* z̃ dynamics: AR(1) with constant ρ */
            float eps = prng_randn(rng);
            z_tilde = dgp.rho * z_tilde + dgp.sigma_z * eps
                      + seg.z_bias * (1.0f - dgp.rho);

            float z = 1.5f * (1.0f + tanhf(z_tilde));

            /* Curves evaluated at z */
            float mu_z    = sat_exp(dgp.mu_base,      dgp.mu_scale,      dgp.mu_rate,      z);
            float sigma_h = sat_exp(dgp.sigma_h_base,  dgp.sigma_h_scale, dgp.sigma_h_rate, z);
            float theta_z = sat_exp(dgp.theta_base,    dgp.theta_scale,   dgp.theta_rate,   z);

            /* h dynamics: matches RBPF inner model exactly
             * h = (1-θ(z))·h + θ(z)·μ(z) + σ_h(z)·ε */
            float phi = 1.0f - theta_z;
            h = phi * h + theta_z * mu_z + sigma_h * prng_randn(rng);

            float y = expf(h * 0.5f) * prng_randn(rng);

            gd.returns.push_back(y);
            gd.log_returns_sq.push_back(logf(y * y + 1e-20f));
            gd.true_h.push_back(h);
            gd.true_z.push_back(z);
        }
    }

    gd.N = (int)gd.returns.size();
    return gd;
}

/* ── Test modes ──────────────────────────────────────────────────────────── */

enum TestMode {
    MODE_BIDIR,      /* Bidirectional valve */
    MODE_ONEWAY,     /* One-way ratchet (never locks back) */
    MODE_PHASE1      /* Always Phase 1 (never unlocks) */
};

static const char* mode_name(TestMode m) {
    switch (m) {
        case MODE_BIDIR:  return "BIDIR (valve)";
        case MODE_ONEWAY: return "ONEWAY (ratchet)";
        case MODE_PHASE1: return "PHASE1 (baseline)";
    }
    return "?";
}

/* ── Run result ──────────────────────────────────────────────────────────── */

struct RunResult {
    std::vector<float> est_h;
    double             rmse_total;
    double             rmse_scored;      /* Crisis 2 only */
    int                n_pushes;
    float              last_mu_pushed;
    int                final_phase;
    int                n_transitions;
};

/* ── One-way ratchet shim ────────────────────────────────────────────────
 *
 * For MODE_ONEWAY, we use the same PhasedLearner but skip the backward
 * path by overriding phased_update with a version that only goes forward.
 * We do this by wrapping: call observe, then manually check only forward.
 * ──────────────────────────────────────────────────────────────────────── */

static inline int phased_update_oneway(PhasedLearner* pl) {
    LearningPhase prev = pl->phase;
    ZRangeTracker* zt = &pl->z_tracker;

    switch (pl->phase) {
    case PHASE_1_FLOORS:
        if (zt->high_z_streak >= pl->config.ceiling_z_sustained) {
            pl->phase = PHASE_2_CEILINGS;
            if (pl->phase2_entered_at < 0)
                pl->phase2_entered_at = zt->n_windows;
            phased_apply_mask(pl);
        }
        break;
    case PHASE_2_CEILINGS:
        if (zt->wide_range_streak >= pl->config.rate_range_sustained) {
            pl->phase = PHASE_3_RATES;
            if (pl->phase3_entered_at < 0)
                pl->phase3_entered_at = zt->n_windows;
            phased_apply_mask(pl);
        }
        /* No backward */
        break;
    case PHASE_3_RATES:
        /* Terminal — no backward */
        break;
    }

    int changed = (pl->phase != prev);
    if (changed) {
        printf("[ONEWAY] Phase %d → %d at window %d\n",
               prev, pl->phase, zt->n_windows);
    }
    return changed;
}

/* ── Run one mode ───────────────────────────────────────────────────────── */

static RunResult run_mode(
    const GeneratedData& gd,
    const TrueDGP& dgp,
    int n_bpf,
    int n_theta,
    int n_inner,
    int window_size,
    int stride,
    TestMode mode,
    unsigned int bpf_seed
) {
    RunResult result;
    int N = gd.N;
    result.est_h.resize(N, 0.0f);
    result.n_pushes = 0;
    result.last_mu_pushed = dgp.mu_base;
    result.n_transitions = 0;

    /* ── Create dBPF ─────────────────────────────────────────────────── */
    float init_mu  = dgp.mu_base;
    float init_rho = dgp.rho;
    float init_sz  = dgp.sigma_z;
    GpuBpfState* bpf = gpu_bpf_create(n_bpf, init_rho, init_sz, init_mu,
                                        0.0f, 0.0f, bpf_seed);

    /* Disable BPF online learning — only SMC² pushes */
    gpu_bpf_disable_mu_learning(bpf);
    gpu_bpf_enable_rho_learning(bpf, 0);

    /* ── Create SMC² ─────────────────────────────────────────────────── */
    SMC2StateCUDA* smc2 = smc2_cuda_alloc(n_theta, n_inner);

    /* Priors centered on truth */
    smc2->prior.rho_mean         = dgp.rho;         smc2->prior.rho_std         = 0.05f;
    smc2->prior.sigma_total_mean = dgp.sigma_z;     smc2->prior.sigma_total_std = 0.1f;
    smc2->prior.r_split_mean     = 0.5f;             smc2->prior.r_split_std     = 0.2f;
    smc2->prior.mu_base_mean     = dgp.mu_base;     smc2->prior.mu_base_std     = 1.0f;
    smc2->prior.mu_scale_mean    = dgp.mu_scale;    smc2->prior.mu_scale_std    = 1.5f;
    smc2->prior.mu_rate_mean     = dgp.mu_rate;     smc2->prior.mu_rate_std     = 0.3f;
    smc2->prior.sigma_scale_mean = dgp.sigma_h_scale;
    smc2->prior.sigma_scale_std  = 0.3f;
    smc2->prior.sigma_rate_mean  = dgp.sigma_h_rate;
    smc2->prior.sigma_rate_std   = 0.2f;

    /* θ(z) curve must match DGP exactly */
    smc2->theta_curve.base  = dgp.theta_base;
    smc2->theta_curve.scale = dgp.theta_scale;
    smc2->theta_curve.rate  = dgp.theta_rate;

    /* ── Create phased controller ────────────────────────────────────── */
    PhasedConfig pc = phased_default_config();
    PhasedLearner* pl = NULL;

    if (mode == MODE_PHASE1) {
        /* Lock ceilings + rates permanently. Use prior defaults. */
        uint8_t mask[N_PARAMS] = {0};
        float   vals[N_PARAMS] = {0};
        mask[P_MU_SCALE]    = 1;  vals[P_MU_SCALE]    = pc.fixed_mu_scale;
        mask[P_MU_RATE]     = 1;  vals[P_MU_RATE]     = pc.fixed_mu_rate;
        mask[P_SIGMA_SCALE] = 1;  vals[P_SIGMA_SCALE] = pc.fixed_sigma_scale;
        mask[P_SIGMA_RATE]  = 1;  vals[P_SIGMA_RATE]  = pc.fixed_sigma_rate;
        smc2_cuda_set_fixed_params(smc2, mask, vals);
    } else {
        pl = phased_create(smc2, pc);
    }

    /* ── Observation buffer (log(r²) for SMC²/RBPF/OCSN) ────────────── */
    smc2_cuda_init_from_prior(smc2);

    std::vector<float> obs_buffer;
    obs_buffer.reserve(N);

    /* ── Main tick loop ──────────────────────────────────────────────── */
    int tick = 0;
    int next_window = window_size;

    for (int t = 0; t < N; t++) {
        /* 1. Accumulate log(r²) for SMC² */
        obs_buffer.push_back(gd.log_returns_sq[t]);

        /* 2. Run BPF on raw returns */
        BpfResult r = gpu_bpf_step(bpf, gd.returns[t]);
        result.est_h[t] = r.h_mean;

        tick++;

        /* 3. Window boundary */
        if (tick >= next_window && (int)obs_buffer.size() >= window_size) {

            /* Extract last window_size obs */
            float* win_obs = &obs_buffer[obs_buffer.size() - window_size];

            /* Run SMC² on this window (cloud carries forward) */
            smc2_cuda_update_batch(smc2, win_obs, window_size);

            next_window += stride;

            /* Update phased controller */
            if (pl) {
                phased_observe_z_from_smc2(pl);
                if (mode == MODE_BIDIR) {
                    phased_update(pl);
                } else {
                    phased_update_oneway(pl);
                }
            }

            /* Push mu_base (RAW push — the traveling cloud's estimate) */
            float theta_mean[N_PARAMS];
            smc2_cuda_get_theta_mean(smc2, theta_mean);

            float mu_push = theta_mean[P_MU_BASE];
            gpu_bpf_set_mu(bpf, mu_push);
            gpu_bpf_set_rho(bpf, theta_mean[P_RHO]);

            result.last_mu_pushed = mu_push;
            result.n_pushes++;
        }
    }

    /* ── Record final state ──────────────────────────────────────────── */
    result.final_phase = pl ? pl->phase : 1;
    result.n_transitions = pl ? pl->n_transitions : 0;

    /* ── RMSE: total ─────────────────────────────────────────────────── */
    {
        double sum_sq = 0; int count = 0;
        for (int t = 0; t < N; t++) {
            if (!std::isnan(result.est_h[t]) && !std::isinf(result.est_h[t])) {
                double err = (double)result.est_h[t] - (double)gd.true_h[t];
                sum_sq += err * err;
                count++;
            }
        }
        result.rmse_total = (count > 0) ? sqrt(sum_sq / count) : 999.0;
    }

    /* ── RMSE: scored segments only (Crisis 2) ───────────────────────── */
    {
        double sum_sq = 0; int count = 0;
        int start = (gd.score_start >= 0) ? gd.score_start : 0;
        for (int t = start; t < N; t++) {
            if (!std::isnan(result.est_h[t]) && !std::isinf(result.est_h[t])) {
                double err = (double)result.est_h[t] - (double)gd.true_h[t];
                sum_sq += err * err;
                count++;
            }
        }
        result.rmse_scored = (count > 0) ? sqrt(sum_sq / count) : 999.0;
    }

    /* ── Cleanup ─────────────────────────────────────────────────────── */
    if (pl) phased_destroy(pl);
    smc2_cuda_free(smc2);
    gpu_bpf_destroy(bpf);

    return result;
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(int argc, char** argv) {
    int n_bpf   = (argc > 1) ? atoi(argv[1]) : 30000;
    int n_theta = 1024;
    int n_inner = 512;

    int window_size = 3000;
    int stride      = 1500;

    TrueDGP dgp = default_dgp();

    /* ── DGP segments ────────────────────────────────────────────────── */
    /*                                                                    */
    /* Key design:                                                        */
    /*   - Crisis 1 has alternating z_bias so each window sees wide       */
    /*     z-range → ceilings identifiable within windows                 */
    /*   - Calm is LONG (45K ticks = 30 windows) → real degeneracy       */
    /*   - Crisis 2 is DIFFERENT from Crisis 1 (different z-level)        */
    /*     so we test generalization, not memorization                    */
    /*                                                                    */
    std::vector<Segment> segments = {
        /* ── Warmup: let cloud settle ─────────────────────────────── */
        {"Warmup (calm)",          6000,  -3.0f,  0},

        /* ── Crisis 1: teach ceilings ─────────────────────────────── */
        /* Alternating z_bias within segments shorter than window size  */
        /* so each W=3000 window straddles low and high z               */
        {"Crisis 1: low",          2000,  -2.0f,  0},
        {"Crisis 1: high",         2000,   2.5f,  0},
        {"Crisis 1: low",          2000,  -2.0f,  0},
        {"Crisis 1: high",         2000,   2.5f,  0},
        {"Crisis 1: low",          2000,  -1.5f,  0},
        {"Crisis 1: high",         2000,   2.0f,  0},

        /* ── Long calm: cloud degenerates ─────────────────────────── */
        /* 45K ticks = 30 windows of nothing. Ridge kills diversity.    */
        {"Calm 1",                 9000,  -3.0f,  0},
        {"Calm 2",                 9000,  -3.0f,  0},
        {"Calm 3",                 9000,  -3.0f,  0},
        {"Calm 4",                 9000,  -3.0f,  0},
        {"Calm 5",                 9000,  -3.0f,  0},

        /* ── Crisis 2: the test ───────────────────────────────────── */
        /* Different z-levels from Crisis 1 to test generalization      */
        /* Alternating so windows see wide z-range                     */
        {"Crisis 2: low",          2000,  -1.5f,  1},
        {"Crisis 2: high",         2000,   2.0f,  1},
        {"Crisis 2: low",          2000,  -1.5f,  1},
        {"Crisis 2: high",         2000,   2.0f,  1},
        {"Crisis 2: low",          2000,  -1.0f,  1},
        {"Crisis 2: high",         2000,   1.5f,  1},
    };

    /* ── Generate data ───────────────────────────────────────────────── */
    PRNG dgp_rng = prng_create(98765);
    GeneratedData gd = generate_data(dgp, segments, &dgp_rng);
    int N = gd.N;

    /* ── Data summary ────────────────────────────────────────────────── */
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║   Bidirectional Phased Learning — Memory Preservation Test           ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════════╣\n");
    printf("║  Crisis 1 (teach) → Long calm (degenerate) → Crisis 2 (score)       ║\n");
    printf("║  BIDIR saves ceilings during calm · ONEWAY lets them degenerate     ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n\n");

    printf("  Configuration:\n");
    printf("    Total ticks: %d  |  Window: %d  |  Stride: %d\n",
           N, window_size, stride);
    printf("    BPF: %dK  |  SMC²: %d × %d\n",
           n_bpf / 1000, n_theta, n_inner);
    printf("    Score starts at tick %d\n\n", gd.score_start);

    printf("  True DGP:\n");
    printf("    ρ=%.2f  σ_z=%.2f  μ_base=%.2f  μ_scale=%.2f  μ_rate=%.2f\n",
           dgp.rho, dgp.sigma_z, dgp.mu_base, dgp.mu_scale, dgp.mu_rate);
    printf("    σ_h_base=%.2f  σ_h_scale=%.2f  σ_h_rate=%.2f\n\n",
           dgp.sigma_h_base, dgp.sigma_h_scale, dgp.sigma_h_rate);

    int n_seg = (int)segments.size();
    printf("  %-25s %6s %5s %6s %6s %6s\n",
           "Segment", "Ticks", "Score", "z_min", "z_max", "z_avg");
    printf("  ─────────────────────── ────── ───── ────── ────── ──────\n");
    for (int s = 0; s < n_seg; s++) {
        int start = gd.segment_starts[s];
        int end   = (s + 1 < n_seg) ? gd.segment_starts[s + 1] : N;
        float zmin = 1e6f, zmax = -1e6f, zsum = 0;
        for (int t = start; t < end; t++) {
            if (gd.true_z[t] < zmin) zmin = gd.true_z[t];
            if (gd.true_z[t] > zmax) zmax = gd.true_z[t];
            zsum += gd.true_z[t];
        }
        int n = end - start;
        printf("  %-25s %6d %5s %6.2f %6.2f %6.2f\n",
               segments[s].name, n, segments[s].score ? "YES" : "---",
               zmin, zmax, zsum / n);
    }
    printf("\n");

    /* ── Run all three modes ─────────────────────────────────────────── */
    unsigned int bpf_seed = 42;
    TestMode modes[] = {MODE_BIDIR, MODE_ONEWAY, MODE_PHASE1};
    RunResult results[3];

    for (int m = 0; m < 3; m++) {
        printf("  ════════════════════════════════════════════════════════\n");
        printf("  Running %-25s\n", mode_name(modes[m]));
        printf("  ────────────────────────────────────────────────────────\n");
        results[m] = run_mode(gd, dgp, n_bpf, n_theta, n_inner,
                               window_size, stride, modes[m], bpf_seed);
        printf("    Total RMSE: %.4f  |  Crisis 2 RMSE: %.4f\n",
               results[m].rmse_total, results[m].rmse_scored);
        printf("    Pushes: %d  |  last_μ: %.3f  |  final phase: %d  |  transitions: %d\n\n",
               results[m].n_pushes, results[m].last_mu_pushed,
               results[m].final_phase, results[m].n_transitions);
    }

    /* ── Per-segment RMSE for scored segments ────────────────────────── */
    printf("  ════════════════════════════════════════════════════════════════════\n");
    printf("  Per-segment RMSE (Crisis 2 only)\n");
    printf("  ────────────────────────────────────────────────────────────────────\n");
    printf("  %-25s %6s | %8s %8s %8s | Winner\n",
           "Segment", "z_avg", "BIDIR", "ONEWAY", "PHASE1");
    printf("  ─────────────────────── ────── | ──────── ──────── ──────── | ──────\n");

    int wins[3] = {0, 0, 0};

    for (int s = 0; s < n_seg; s++) {
        if (!segments[s].score) continue;

        int start = gd.segment_starts[s];
        int end   = (s + 1 < n_seg) ? gd.segment_starts[s + 1] : N;

        double rmse[3];
        for (int m2 = 0; m2 < 3; m2++) {
            double sum_sq = 0; int count = 0;
            for (int t = start; t < end; t++) {
                float est = results[m2].est_h[t];
                if (!std::isnan(est) && !std::isinf(est)) {
                    double err = (double)est - (double)gd.true_h[t];
                    sum_sq += err * err;
                    count++;
                }
            }
            rmse[m2] = (count > 0) ? sqrt(sum_sq / count) : 999.0;
        }

        int best = 0;
        for (int m2 = 1; m2 < 3; m2++) if (rmse[m2] < rmse[best]) best = m2;
        wins[best]++;

        float zsum = 0;
        for (int t = start; t < end; t++) zsum += gd.true_z[t];
        float zavg = zsum / (end - start);

        const char* winner[] = {"BIDIR", "ONEWAY", "PHASE1"};
        printf("  %-25s %6.2f | %8.4f %8.4f %8.4f | %s\n",
               segments[s].name, zavg, rmse[0], rmse[1], rmse[2], winner[best]);
    }

    /* ── Grand summary ───────────────────────────────────────────────── */
    printf("\n");
    printf("  ════════════════════════════════════════════════════════════════════\n");
    printf("  RESULTS — Crisis 2 RMSE (scored segments only)\n");
    printf("  ────────────────────────────────────────────────────────────────────\n");
    printf("  %-25s %10s %10s %10s\n", "Mode", "Crisis2", "Total", "last μ");
    printf("  ─────────────────────── ────────── ────────── ──────────\n");

    int best_scored = 0;
    for (int m = 0; m < 3; m++) {
        printf("  %-25s %10.4f %10.4f %10.3f\n",
               mode_name(modes[m]),
               results[m].rmse_scored, results[m].rmse_total,
               results[m].last_mu_pushed);
        if (results[m].rmse_scored < results[best_scored].rmse_scored)
            best_scored = m;
    }

    printf("\n");
    printf("  Best Crisis 2 RMSE: %s\n", mode_name(modes[best_scored]));
    printf("  Segment wins:       BIDIR=%d  ONEWAY=%d  PHASE1=%d\n",
           wins[0], wins[1], wins[2]);

    double bidir_s  = results[0].rmse_scored;
    double oneway_s = results[1].rmse_scored;
    double phase1_s = results[2].rmse_scored;

    printf("\n  Relative Crisis 2 RMSE:\n");
    printf("    BIDIR  vs ONEWAY:  %+.1f%%\n", 100.0 * (bidir_s / oneway_s - 1.0));
    printf("    BIDIR  vs PHASE1:  %+.1f%%\n", 100.0 * (bidir_s / phase1_s - 1.0));
    printf("    ONEWAY vs PHASE1:  %+.1f%%\n", 100.0 * (oneway_s / phase1_s - 1.0));

    /* ── Diagnosis ───────────────────────────────────────────────────── */
    printf("\n  ────────────────────────────────────────────────────────────────────\n");
    printf("  DIAGNOSIS\n");
    printf("  ────────────────────────────────────────────────────────────────────\n");

    if (bidir_s < oneway_s * 0.97) {
        printf("  ✓ BIDIR beats ONEWAY by >3%%. Memory preservation works.\n");
        printf("    Saved ceilings survived the calm period intact.\n");
    } else if (bidir_s < oneway_s) {
        printf("  ~ BIDIR slightly better than ONEWAY (%.1f%%).\n",
               100.0 * (1.0 - bidir_s / oneway_s));
        printf("    Effect is small — calm period may not have been long enough.\n");
    } else {
        printf("  ✗ BIDIR not better than ONEWAY. Either:\n");
        printf("    - Cloud didn't degenerate during calm (ESS stayed healthy)\n");
        printf("    - Saved values were wrong (bad learning in Crisis 1)\n");
        printf("    - z-range trigger didn't fire correctly\n");
    }

    printf("\n");
    if (bidir_s < phase1_s * 0.97) {
        printf("  ✓ BIDIR beats PHASE1 by >3%%. Learned ceilings beat prior defaults.\n");
    } else if (bidir_s < phase1_s) {
        printf("  ~ BIDIR slightly better than PHASE1 (%.1f%%).\n",
               100.0 * (1.0 - bidir_s / phase1_s));
    } else {
        printf("  ✗ BIDIR not better than PHASE1. Valve not helping.\n");
    }

    printf("\n");
    if (oneway_s > phase1_s) {
        printf("  ! ONEWAY worse than PHASE1 — cloud degeneracy during calm\n");
        printf("    actively hurt performance. The one-way ratchet is harmful.\n");
    }

    printf("  ════════════════════════════════════════════════════════════════════\n\n");

    return 0;
}
