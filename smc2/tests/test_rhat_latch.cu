/*═══════════════════════════════════════════════════════════════════════════════
 * @file test_rhat_latch.cu
 * @brief Validate GATE_RHAT_LATCH for curve params during Phase 2 transition
 *
 * Flow:
 *   Phase 1 (windows 1-8):
 *     - Params 0-3 free (GATE_KALMAN_MIN), 4-7 fixed in SMC² + GATE_LOCKED
 *     - Stressed DGP so SMC² can learn dynamics
 *
 *   Phase 2 transition (after window 8):
 *     - Unfix params 4-5 (μ_scale, μ_rate) in SMC²
 *     - Call set_free_mask to free them → GATE_RHAT_LATCH
 *
 *   Phase 2 (windows 9-20+):
 *     - Stressed data continues
 *     - R̂ for params 4-5 should eventually drop below threshold → latch fires
 *
 *   Calm period (after latch):
 *     - R̂ would spike (no stressed data) but latch holds
 *
 * Checks:
 *   1. Params 4-5 stay LOCKED during Phase 1
 *   2. After transition, params 4-5 become RHAT_LATCH (converged=0)
 *   3. R̂ eventually drops → latch fires (converged=1)
 *   4. Latch holds through calm period
 *   5. Gated values match Kalman after latch
 *   6. Params 6-7 stay LOCKED throughout (Phase 3 not triggered)
 *
 * Build:
 *   smc2_add_test(test_rhat_latch NEEDS_BPF)
 *═══════════════════════════════════════════════════════════════════════════════*/

#include "gpu_bpf_full.cuh"
#include "smc2_param_tracker.cuh"
#include "smc2_convergence_diag.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

/* ── PRNG ────────────────────────────────────────────────────────────────── */

struct PRNG { unsigned int state; };

static inline PRNG prng_create(unsigned int seed) {
    PRNG p; p.state = seed ? seed : 1; return p;
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
    float rho, sigma_z;
    float mu_base, mu_scale, mu_rate;
    float sigma_h_base, sigma_h_scale, sigma_h_rate;
    float theta_base, theta_scale, theta_rate;
};

static inline float sat_exp(float base, float scale, float rate, float z) {
    return base + scale * (1.0f - expf(-rate * z));
}

static TrueDGP default_dgp() {
    TrueDGP d;
    d.rho = 0.98f;  d.sigma_z = 0.15f;
    d.mu_base = -4.5f;  d.mu_scale = 3.0f;  d.mu_rate = 0.5f;
    d.sigma_h_base = 0.10f;  d.sigma_h_scale = 0.50f;  d.sigma_h_rate = 0.30f;
    d.theta_base = 0.02f;  d.theta_scale = 0.08f;  d.theta_rate = 1.5f;
    return d;
}

/* ── Data generation ─────────────────────────────────────────────────────── */

struct Segment { const char* name; int ticks; float z_bias; };

static void generate_segment(const TrueDGP& dgp, const Segment& seg,
                              float& z_tilde, float& h, PRNG* rng,
                              std::vector<float>& returns,
                              std::vector<float>& log_returns_sq,
                              std::vector<float>& true_h,
                              std::vector<float>& true_z)
{
    for (int t = 0; t < seg.ticks; t++) {
        z_tilde = dgp.rho * z_tilde + dgp.sigma_z * prng_randn(rng)
                  + seg.z_bias * (1.0f - dgp.rho);
        float z = 1.5f * (1.0f + tanhf(z_tilde));
        float mu_z    = sat_exp(dgp.mu_base, dgp.mu_scale, dgp.mu_rate, z);
        float sigma_h = sat_exp(dgp.sigma_h_base, dgp.sigma_h_scale, dgp.sigma_h_rate, z);
        float theta_z = sat_exp(dgp.theta_base, dgp.theta_scale, dgp.theta_rate, z);
        float phi = 1.0f - theta_z;
        h = phi * h + theta_z * mu_z + sigma_h * prng_randn(rng);
        float y = expf(h * 0.5f) * prng_randn(rng);
        returns.push_back(y);
        log_returns_sq.push_back(logf(y * y + 1e-20f));
        true_h.push_back(h);
        true_z.push_back(z);
    }
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(int argc, char** argv) {
    int n_bpf       = (argc > 1) ? atoi(argv[1]) : 30000;
    int n_theta     = 1024;
    int n_inner     = 512;
    int window_size = 3000;
    int stride      = 1500;

    TrueDGP dgp = default_dgp();
    PRNG rng = prng_create(98765);

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║   GATE_RHAT_LATCH Test: Phase 2 Curve Param Convergence             ║\n");
    printf("║   Phase 1 → Phase 2 transition → latch → calm hold                  ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n\n");

    printf("  Config: BPF=%dK | SMC²=%d×%d | W=%d stride=%d\n\n",
           n_bpf/1000, n_theta, n_inner, window_size, stride);

    /* ── Create ParamTracker ─────────────────────────────────────────── */
    ParamTracker* tracker = param_tracker_create(window_size, stride,
                                                  n_theta, n_inner);

    SMC2StateCUDA* smc2 = param_tracker_get_smc2(tracker);

    /* Priors centered on truth (tight for curve params — helps 8-param estimation) */
    smc2->prior.rho_mean         = dgp.rho;          smc2->prior.rho_std         = 0.05f;
    smc2->prior.sigma_total_mean = dgp.sigma_z;      smc2->prior.sigma_total_std = 0.1f;
    smc2->prior.r_split_mean     = 0.5f;              smc2->prior.r_split_std     = 0.2f;
    smc2->prior.mu_base_mean     = dgp.mu_base;      smc2->prior.mu_base_std     = 1.0f;
    smc2->prior.mu_scale_mean    = dgp.mu_scale;     smc2->prior.mu_scale_std    = 1.5f;
    smc2->prior.mu_rate_mean     = dgp.mu_rate;      smc2->prior.mu_rate_std     = 0.3f;
    smc2->prior.sigma_scale_mean = dgp.sigma_h_scale;
    smc2->prior.sigma_scale_std  = 0.3f;
    smc2->prior.sigma_rate_mean  = dgp.sigma_h_rate;
    smc2->prior.sigma_rate_std   = 0.2f;

    smc2->theta_curve.base  = dgp.theta_base;
    smc2->theta_curve.scale = dgp.theta_scale;
    smc2->theta_curve.rate  = dgp.theta_rate;

    /* Phase 1: fix curve params 4-7 in SMC² */
    {
        uint8_t mask[N_PARAMS] = {0};
        float   vals[N_PARAMS] = {0};
        mask[4] = 1; vals[4] = dgp.mu_scale;
        mask[5] = 1; vals[5] = dgp.mu_rate;
        mask[6] = 1; vals[6] = dgp.sigma_h_scale;
        mask[7] = 1; vals[7] = dgp.sigma_h_rate;
        smc2_cuda_set_fixed_params(smc2, mask, vals);
    }

    param_tracker_set_theta_curve(tracker, dgp.theta_base,
                                  dgp.theta_scale, dgp.theta_rate);

    /* Phase 1 gating: 0-3 KALMAN_MIN, 4-7 LOCKED */
    int phase1_mask[N_PARAMS] = {1, 1, 1, 1, 0, 0, 0, 0};
    param_tracker_set_free_mask(tracker, phase1_mask);
    param_tracker_set_min_windows(tracker, 2);

    float prior_def[N_PARAMS] = {
        0.85f, 0.15f, 0.50f, -10.0f,
        dgp.mu_scale, dgp.mu_rate, dgp.sigma_h_scale, dgp.sigma_h_rate
    };
    param_tracker_set_prior_defaults(tracker, prior_def);

    /* Create BPF */
    GpuBpfState* bpf = gpu_bpf_create(n_bpf, prior_def[0], dgp.sigma_z,
                                        prior_def[3], 0.0f, 0.0f, 42);
    gpu_bpf_disable_mu_learning(bpf);
    gpu_bpf_enable_rho_learning(bpf, 0);

    /* ── Generate data ───────────────────────────────────────────────── */
    /* Phase 1: 12K stressed (8 windows) */
    /* Phase 2: 18K stressed (12 windows — curve params converge) */
    /* Calm:    15K calm (10 windows — latch hold test) */

    std::vector<float> returns, log_rsq, true_h, true_z;
    float z_tilde = 0.0f, h = dgp.mu_base;

    struct Phase {
        const char* label;
        std::vector<Segment> segs;
        int trigger_window;       /* phase transition happens AFTER this window */
    };

    /* Phase 1 segments: alternating stress */
    std::vector<Segment> p1_segs = {
        {"L: Low 1",  2000, -2.0f}, {"L: High 1", 2000, 2.0f},
        {"L: Low 2",  2000, -2.0f}, {"L: High 2", 2000, 2.0f},
        {"L: Low 3",  2000, -2.0f}, {"L: High 3", 2000, 2.0f},
    };
    /* Phase 2 segments: more stress for curve param identification */
    std::vector<Segment> p2_segs = {
        {"P2: Low 1",  3000, -1.5f}, {"P2: High 1", 3000, 2.0f},
        {"P2: Low 2",  3000, -1.5f}, {"P2: High 2", 3000, 2.0f},
        {"P2: Low 3",  3000, -1.0f}, {"P2: High 3", 3000, 2.5f},
    };
    /* Calm segments: latch hold test */
    std::vector<Segment> calm_segs = {
        {"Calm 1", 5000, -3.0f}, {"Calm 2", 5000, -3.0f}, {"Calm 3", 5000, -3.0f},
    };

    printf("  Generating data...\n");
    for (auto& s : p1_segs) generate_segment(dgp, s, z_tilde, h, &rng, returns, log_rsq, true_h, true_z);
    int p1_end = (int)returns.size();
    for (auto& s : p2_segs) generate_segment(dgp, s, z_tilde, h, &rng, returns, log_rsq, true_h, true_z);
    int p2_end = (int)returns.size();
    for (auto& s : calm_segs) generate_segment(dgp, s, z_tilde, h, &rng, returns, log_rsq, true_h, true_z);
    int N = (int)returns.size();

    printf("  Phase 1: %d ticks (stressed) → Phase 2: %d ticks (stressed) → Calm: %d ticks\n",
           p1_end, p2_end - p1_end, N - p2_end);
    printf("  Total: %d ticks\n\n", N);

    /* ── Print header ────────────────────────────────────────────────── */
    static const char* PNAMES[] = {
        "rho", "sig_t", "r_spl", "mu_b", "mu_s", "mu_r", "sig_s", "sig_r"
    };

    printf("  Win  Tick Ph|");
    for (int i = 0; i < N_PARAMS; i++) printf(" %5s", PNAMES[i]);
    printf(" | d²   ESS   | Conv  | Notes\n");

    printf("  ---- ---- --|");
    for (int i = 0; i < N_PARAMS; i++) printf(" -----");
    printf(" |-----------|-------|------\n");

    /* ── Main loop ───────────────────────────────────────────────────── */
    int n_windows = 0;
    int phase = 1;
    int phase2_transition_window = -1;

    /* Tracking for checks */
    int p45_locked_during_p1 = 1;       /* Check 1 */
    int p45_rhat_latch_after_trans = 0; /* Check 2 */
    int p45_latched_window = -1;        /* Check 3 */
    int p45_latch_held = 1;             /* Check 4 */
    int p67_locked_throughout = 1;      /* Check 6 */

    /* Snapshot values at latch point */
    float gated_at_latch[N_PARAMS] = {0};
    float kalman_at_latch[N_PARAMS] = {0};

    for (int t = 0; t < N; t++) {
        param_tracker_feed(tracker, log_rsq[t]);
        BpfResult br = gpu_bpf_step(bpf, returns[t]);

        if (param_tracker_window_ready(tracker)) {
            param_tracker_run_window(tracker);
            n_windows++;

            int conv[N_PARAMS];
            param_tracker_get_converged(tracker, conv);

            ParamSnapshot snap;
            param_tracker_get_snapshot(tracker, &snap);

            ConvergenceReport rpt;
            param_tracker_get_conv_report(tracker, &rpt);

            float kalman_x[N_PARAMS];
            param_tracker_get_kalman_x(tracker, kalman_x);

            /* Push gated params to BPF */
            gpu_bpf_set_mu(bpf, snap.theta[3]);
            gpu_bpf_set_rho(bpf, snap.theta[0]);

            /* ── Phase transition at tick p1_end ─────────────────── */
            if (phase == 1 && t >= p1_end && phase2_transition_window < 0) {
                phase = 2;
                phase2_transition_window = n_windows;

                /* Unfix params 4-5 in SMC² (keep 6-7 fixed) */
                {
                    uint8_t mask[N_PARAMS] = {0};
                    float   vals[N_PARAMS] = {0};
                    mask[6] = 1; vals[6] = dgp.sigma_h_scale;
                    mask[7] = 1; vals[7] = dgp.sigma_h_rate;
                    smc2_cuda_set_fixed_params(smc2, mask, vals);
                }

                /* Free params 4-5 in gating → GATE_RHAT_LATCH */
                int phase2_mask[N_PARAMS] = {1, 1, 1, 1, 1, 1, 0, 0};
                param_tracker_set_free_mask(tracker, phase2_mask);

                printf("  ──── PHASE 2 TRANSITION at window %d (tick %d) ──────────────────\n",
                       n_windows, t);
                printf("        params 4-5 (μ_scale, μ_rate) unfixed → GATE_RHAT_LATCH\n");
            }

            /* ── Check tracking ──────────────────────────────────── */

            /* Check 1: params 4-5 locked during Phase 1 */
            if (phase == 1) {
                if (conv[4] != -1 || conv[5] != -1)
                    p45_locked_during_p1 = 0;
            }

            /* Check 2: after transition, 4-5 should be 0 (not yet converged)
             * before they latch */
            if (phase2_transition_window > 0 && n_windows == phase2_transition_window + 1) {
                /* Re-read after transition took effect */
                param_tracker_get_converged(tracker, conv);
                if (conv[4] == 0 && conv[5] == 0)
                    p45_rhat_latch_after_trans = 1;
            }

            /* Check 3: detect latch point */
            if (p45_latched_window < 0 && conv[4] == 1 && conv[5] == 1) {
                p45_latched_window = n_windows;
                memcpy(gated_at_latch, snap.theta, sizeof(gated_at_latch));
                memcpy(kalman_at_latch, kalman_x, sizeof(kalman_at_latch));
            }

            /* Check 4: once latched, should never revert */
            if (p45_latched_window > 0) {
                if (conv[4] != 1 || conv[5] != 1)
                    p45_latch_held = 0;
            }

            /* Check 6: params 6-7 always locked */
            if (conv[6] != -1 || conv[7] != -1)
                p67_locked_throughout = 0;

            /* ── Print row ───────────────────────────────────────── */
            int cur_phase = (t < p1_end) ? 1 : (t < p2_end) ? 2 : 0;
            printf("  %3d %5d P%d|", n_windows, t, cur_phase);
            if (!rpt.ready) {
                for (int i = 0; i < N_PARAMS; i++) printf("    — ");
            } else {
                for (int i = 0; i < N_PARAMS; i++) {
                    if (conv[i] == -1)
                        printf("  lock");
                    else if (conv[i] == 1)
                        printf(" \033[92m%5.2f\033[0m", rpt.rhat[i]);  /* green */
                    else
                        printf(" \033[91m%5.2f\033[0m", rpt.rhat[i]);  /* red */
                }
            }
            printf(" | %4.1f %5.0f", rpt.mahal_mean, snap.last_ess);

            /* Conv count for free params */
            int n_conv = 0, n_free = 0;
            for (int i = 0; i < N_PARAMS; i++) {
                if (conv[i] >= 0) { n_free++; if (conv[i] == 1) n_conv++; }
            }
            printf(" | %d/%d", n_conv, n_free);

            /* Notes */
            if (n_windows == phase2_transition_window)
                printf("   ← Phase 2");
            if (n_windows == p45_latched_window)
                printf("   ← LATCH");
            if (cur_phase == 0 && p45_latched_window > 0)
                printf("   calm");

            printf("\n");
        }
    }

    /* ── Results ─────────────────────────────────────────────────────── */
    printf("\n  ════════════════════════════════════════════════════════════════════\n");
    printf("  RESULTS\n");
    printf("  ════════════════════════════════════════════════════════════════════\n\n");

    int pass = 1;

    /* Check 1 */
    if (p45_locked_during_p1) {
        printf("  ✓ Check 1: params 4-5 stayed LOCKED during Phase 1\n");
    } else {
        printf("  ✗ Check 1 FAIL: params 4-5 were not locked during Phase 1\n");
        pass = 0;
    }

    /* Check 2 */
    if (p45_rhat_latch_after_trans) {
        printf("  ✓ Check 2: params 4-5 transitioned to RHAT_LATCH (converged=0) after Phase 2\n");
    } else {
        printf("  ✗ Check 2 FAIL: params 4-5 did not enter RHAT_LATCH state\n");
        pass = 0;
    }

    /* Check 3 */
    if (p45_latched_window > 0) {
        int windows_to_latch = p45_latched_window - phase2_transition_window;
        printf("  ✓ Check 3: params 4-5 latched at window %d (%d windows after Phase 2 transition)\n",
               p45_latched_window, windows_to_latch);
        printf("      μ_scale: gated=%.4f  kalman=%.4f  truth=%.4f\n",
               gated_at_latch[4], kalman_at_latch[4], dgp.mu_scale);
        printf("      μ_rate:  gated=%.4f  kalman=%.4f  truth=%.4f\n",
               gated_at_latch[5], kalman_at_latch[5], dgp.mu_rate);
    } else {
        printf("  ✗ Check 3 FAIL: params 4-5 never latched (R̂ never dropped below threshold)\n");
        /* This could happen if 12 windows isn't enough. Not necessarily a code bug. */
        pass = 0;
    }

    /* Check 4 */
    if (p45_latched_window > 0 && p45_latch_held) {
        printf("  ✓ Check 4: latch held through calm period (never reverted)\n");
    } else if (p45_latched_window > 0) {
        printf("  ✗ Check 4 FAIL: latch reverted during calm period!\n");
        pass = 0;
    } else {
        printf("  ~ Check 4: skipped (never latched)\n");
    }

    /* Check 5: gated == kalman at latch point */
    if (p45_latched_window > 0) {
        float gap4 = fabsf(gated_at_latch[4] - kalman_at_latch[4]);
        float gap5 = fabsf(gated_at_latch[5] - kalman_at_latch[5]);
        if (gap4 < 0.01f && gap5 < 0.01f) {
            printf("  ✓ Check 5: gated values match Kalman at latch point\n");
        } else {
            printf("  ✗ Check 5 FAIL: gated != kalman at latch (gap4=%.4f gap5=%.4f)\n",
                   gap4, gap5);
            pass = 0;
        }
    } else {
        printf("  ~ Check 5: skipped (never latched)\n");
    }

    /* Check 6 */
    if (p67_locked_throughout) {
        printf("  ✓ Check 6: params 6-7 stayed LOCKED throughout (Phase 3 not triggered)\n");
    } else {
        printf("  ✗ Check 6 FAIL: params 6-7 were modified\n");
        pass = 0;
    }

    /* Final state */
    printf("\n  ── Final Param State ──\n");
    int conv_final[N_PARAMS];
    param_tracker_get_converged(tracker, conv_final);
    ParamSnapshot snap_f;
    param_tracker_get_snapshot(tracker, &snap_f);
    float kalman_f[N_PARAMS];
    param_tracker_get_kalman_x(tracker, kalman_f);
    float truth[N_PARAMS] = {
        dgp.rho, dgp.sigma_z, 0.5f, dgp.mu_base,
        dgp.mu_scale, dgp.mu_rate, dgp.sigma_h_scale, dgp.sigma_h_rate
    };

    printf("  %-8s  %-6s  %10s  %10s  %10s\n",
           "Param", "Conv", "Gated", "Kalman", "Truth");
    printf("  ──────── ────── ────────── ────────── ──────────\n");
    for (int i = 0; i < N_PARAMS; i++) {
        const char* st = (conv_final[i] == 1) ? "CONV" :
                         (conv_final[i] == -1) ? "LOCK" : "—";
        printf("  %-8s  %-6s  %10.4f  %10.4f  %10.4f\n",
               PNAMES[i], st, snap_f.theta[i], kalman_f[i], truth[i]);
    }

    printf("\n  ════════════════════════════════════════════════════════════════════\n");
    if (pass)
        printf("  ALL CHECKS PASSED ✓\n");
    else
        printf("  SOME CHECKS FAILED ✗\n");
    printf("  ════════════════════════════════════════════════════════════════════\n\n");

    param_tracker_destroy(tracker);
    gpu_bpf_destroy(bpf);

    return pass ? 0 : 1;
}
