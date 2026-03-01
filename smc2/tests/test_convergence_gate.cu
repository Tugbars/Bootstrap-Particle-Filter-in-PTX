/*═══════════════════════════════════════════════════════════════════════════════
 * @file test_convergence_gate.cu
 * @brief Validate convergence-gated ParamTracker → BPF 1D pipeline
 *
 * Runs the full production path:
 *   param_tracker_feed() → window_ready() → run_window() → get_snapshot()
 *   → push gated params to BPF 1D → step BPF → measure RMSE
 *
 * Same multi-regime DGP as test_kalman_fair:
 *   12K learning (alternating stress) → 30K calm → 12K scored crisis
 *
 * What this validates:
 *   1. Gated snapshot holds prior defaults until R̂ converges
 *   2. Phase 1 params (ρ, σ_total, r, μ_base) converge within ~5 windows
 *   3. BPF RMSE is not degraded by gating (prior defaults are reasonable)
 *   4. After convergence, pushed values track Kalman estimates
 *   5. Per-param convergence table shows R̂ trajectory
 *
 * Build:
 *   Part of smc2/tests/ — add to CMakeLists.txt:
 *     smc2_add_test(test_convergence_gate)
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

/* ── Segments ────────────────────────────────────────────────────────────── */

struct Segment {
    const char* name;
    int ticks;
    float z_bias;
    int score;
};

struct GeneratedData {
    std::vector<float> returns;
    std::vector<float> log_returns_sq;
    std::vector<float> true_h, true_z;
    std::vector<int> segment_starts;
    std::vector<const char*> segment_names;
    std::vector<int> segment_score;
    int N, score_start;
};

static GeneratedData generate_data(const TrueDGP& dgp,
                                    const std::vector<Segment>& segments,
                                    PRNG* rng) {
    GeneratedData gd;
    gd.score_start = -1;
    float z_tilde = 0.0f, h = dgp.mu_base;

    for (size_t s = 0; s < segments.size(); s++) {
        const Segment& seg = segments[s];
        gd.segment_starts.push_back((int)gd.returns.size());
        gd.segment_names.push_back(seg.name);
        gd.segment_score.push_back(seg.score);
        if (seg.score && gd.score_start < 0)
            gd.score_start = (int)gd.returns.size();

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
            gd.returns.push_back(y);
            gd.log_returns_sq.push_back(logf(y * y + 1e-20f));
            gd.true_h.push_back(h);
            gd.true_z.push_back(z);
        }
    }
    gd.N = (int)gd.returns.size();
    return gd;
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(int argc, char** argv) {
    int n_bpf       = (argc > 1) ? atoi(argv[1]) : 30000;
    int n_theta     = 1024;
    int n_inner     = 512;
    int window_size = 3000;
    int stride      = 1500;

    TrueDGP dgp = default_dgp();

    /* ── Multi-regime DGP (identical to kalman_fair) ─────────────────── */
    std::vector<Segment> segments = {
        {"L: Low 1",   2000, -2.0f, 0},
        {"L: High 1",  2000,  2.0f, 0},
        {"L: Low 2",   2000, -2.0f, 0},
        {"L: High 2",  2000,  2.0f, 0},
        {"L: Low 3",   2000, -2.0f, 0},
        {"L: High 3",  2000,  2.0f, 0},
        {"Calm 1",    10000, -3.0f, 0},
        {"Calm 2",    10000, -3.0f, 0},
        {"Calm 3",    10000, -3.0f, 0},
        {"T: Low 1",   2000, -1.5f, 1},
        {"T: High 1",  2000,  1.5f, 1},
        {"T: Low 2",   2000, -1.5f, 1},
        {"T: High 2",  2000,  1.5f, 1},
        {"T: Low 3",   2000, -1.0f, 1},
        {"T: High 3",  2000,  2.0f, 1},
    };

    PRNG dgp_rng = prng_create(98765);
    GeneratedData gd = generate_data(dgp, segments, &dgp_rng);
    int N = gd.N;

    /* ── Header ──────────────────────────────────────────────────────── */
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║   Convergence-Gated Pipeline Test (ParamTracker → BPF 1D)           ║\n");
    printf("║   Amnesia + Kalman + R̂ gating on multi-regime DGP                  ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n\n");

    printf("  Config: %d ticks | W=%d | stride=%d | BPF=%dK | SMC²=%d×%d\n",
           N, window_size, stride, n_bpf/1000, n_theta, n_inner);
    printf("  Score starts at tick %d\n", gd.score_start);
    printf("  Phase 1: ρ, σ_total, r, μ_base free (GATE_KALMAN_MIN, 2 windows)\n");
    printf("  Phase 2/3: curve params locked (GATE_LOCKED)\n");
    printf("  Gating: fast params push Kalman after 2 windows, curve params R̂-latched\n\n");

    /* Segment summary */
    int n_seg = (int)segments.size();
    printf("  %-16s %6s %5s %6s %6s\n", "Segment", "Ticks", "Score", "z_min", "z_max");
    printf("  ──────────────── ────── ───── ────── ──────\n");
    for (int s = 0; s < n_seg; s++) {
        int start = gd.segment_starts[s];
        int end = (s+1 < n_seg) ? gd.segment_starts[s+1] : N;
        float zmin = 1e6f, zmax = -1e6f;
        for (int t = start; t < end; t++) {
            if (gd.true_z[t] < zmin) zmin = gd.true_z[t];
            if (gd.true_z[t] > zmax) zmax = gd.true_z[t];
        }
        printf("  %-16s %6d %5s %6.2f %6.2f\n",
               segments[s].name, end - start,
               segments[s].score ? "YES" : "---", zmin, zmax);
    }
    printf("\n");

    /* ── Create ParamTracker (owns SMC² + Kalman) ────────────────────── */
    ParamTracker* tracker = param_tracker_create(window_size, stride,
                                                  n_theta, n_inner);

    /* Configure SMC² priors via internal access */
    SMC2StateCUDA* smc2 = param_tracker_get_smc2(tracker);

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

    /* Fix curve shape params in SMC² (Phase 1: only 4 free) */
    {
        uint8_t mask[N_PARAMS] = {0};
        float   vals[N_PARAMS] = {0};
        mask[4] = 1; vals[4] = dgp.mu_scale;
        mask[5] = 1; vals[5] = dgp.mu_rate;
        mask[6] = 1; vals[6] = dgp.sigma_h_scale;
        mask[7] = 1; vals[7] = dgp.sigma_h_rate;
        smc2_cuda_set_fixed_params(smc2, mask, vals);
    }

    /* Set θ(z) curve on tracker */
    param_tracker_set_theta_curve(tracker, dgp.theta_base,
                                  dgp.theta_scale, dgp.theta_rate);

    /* ── Configure convergence gating ────────────────────────────────── */

    /* Phase 1 free mask: params 0-3 free, 4-7 locked.
     * set_free_mask sets locked params to GATE_LOCKED.
     * Default gate_mode for 0-3 is already GATE_KALMAN_MIN (from create). */
    int free_mask[N_PARAMS] = {1, 1, 1, 1, 0, 0, 0, 0};
    param_tracker_set_free_mask(tracker, free_mask);

    /* min_windows = 2: push Kalman estimates starting from window 2.
     * Window 1 initializes Kalman, window 2 updates — then we trust it. */
    param_tracker_set_min_windows(tracker, 2);

    /* Prior defaults: what BPF sees BEFORE min_windows.
     * μ_base = -10.0 (truth = -4.5) — large gap, tests that gate releases quickly.
     * Params 4-7: DGP truth (fixed in SMC², GATE_LOCKED in gating). */
    float prior_def[N_PARAMS] = {
        0.85f,               /* ρ — prior mean (truth=0.98) */
        0.15f,               /* σ_total — prior mean (truth=0.15) */
        0.50f,               /* r_split — prior mean */
       -10.0f,               /* μ_base — prior mean (truth=-4.5, BIG gap) */
        dgp.mu_scale,        /* μ_scale — locked at truth */
        dgp.mu_rate,         /* μ_rate — locked at truth */
        dgp.sigma_h_scale,   /* σ_scale — locked at truth */
        dgp.sigma_h_rate     /* σ_rate — locked at truth */
    };
    param_tracker_set_prior_defaults(tracker, prior_def);

    /* ── Create BPF 1D ───────────────────────────────────────────────── */
    /* Init with prior defaults — same values the gating will push
     * before any param converges */
    GpuBpfState* bpf = gpu_bpf_create(n_bpf, prior_def[0],
                                        prior_def[1] * prior_def[2],  /* σ_z = r·σ_total */
                                        prior_def[3],                  /* μ_base */
                                        0.0f, 0.0f, 42);
    gpu_bpf_disable_mu_learning(bpf);
    gpu_bpf_enable_rho_learning(bpf, 0);

    /* ── Print convergence table header ──────────────────────────────── */
    static const char* PNAMES[] = {
        "rho", "sig_t", "r_spl", "mu_b", "mu_s", "mu_r", "sig_s", "sig_r"
    };
    static const char* GMODE[] = {"KAL", "R̂L", "LCK"};

    printf("  ────────────────────────────────────────────────────────────────────\n");
    printf("  GATE MODES\n");
    printf("  ────────────────────────────────────────────────────────────────────\n");
    {
        for (int i = 0; i < N_PARAMS; i++) {
            int mode = (i < 4) ? GATE_KALMAN_MIN : GATE_LOCKED;
            printf("    %-6s: %s", PNAMES[i], GMODE[mode]);
            if (mode == GATE_KALMAN_MIN) printf(" (push after %d windows)", 2);
            printf("\n");
        }
    }

    printf("\n  ────────────────────────────────────────────────────────────────────\n");
    printf("  CONVERGENCE TABLE (R̂ per window, gated push values)\n");
    printf("  ────────────────────────────────────────────────────────────────────\n\n");

    printf("  Win  Tick |");
    for (int i = 0; i < N_PARAMS; i++) printf(" %5s", PNAMES[i]);
    printf(" | d²   Ptr    ESS   | mu_push  rho_push | Conv\n");

    printf("  ---- -----|");
    for (int i = 0; i < N_PARAMS; i++) printf(" -----");
    printf(" |-------------------|--------------------|---------\n");

    /* ── Main loop ───────────────────────────────────────────────────── */
    std::vector<float> est_h(N, 0.0f);
    int n_windows = 0;
    int first_all_conv = -1;
    int mu_reverted = 0;  /* Track if μ_base ever reverted to prior after converging */

    /* Track when each param first converges */
    int first_conv_window[N_PARAMS];
    for (int i = 0; i < N_PARAMS; i++) first_conv_window[i] = -1;

    for (int t = 0; t < N; t++) {
        /* Feed log(y²) to tracker (OCSN transform) */
        param_tracker_feed(tracker, gd.log_returns_sq[t]);

        /* BPF step on raw return */
        BpfResult br = gpu_bpf_step(bpf, gd.returns[t]);
        est_h[t] = br.h_mean;

        /* Window boundary */
        if (param_tracker_window_ready(tracker)) {
            param_tracker_run_window(tracker);
            n_windows++;

            /* Get gated snapshot */
            ParamSnapshot snap;
            param_tracker_get_snapshot(tracker, &snap);

            /* Get convergence report */
            ConvergenceReport rpt;
            param_tracker_get_conv_report(tracker, &rpt);

            /* Get raw Kalman state for comparison */
            float kalman_x[N_PARAMS];
            param_tracker_get_kalman_x(tracker, kalman_x);

            /* Push gated params to BPF */
            gpu_bpf_set_mu(bpf, snap.theta[3]);    /* gated μ_base */
            gpu_bpf_set_rho(bpf, snap.theta[0]);   /* gated ρ */

            /* Track first convergence per param */
            int conv[N_PARAMS];
            param_tracker_get_converged(tracker, conv);
            for (int i = 0; i < N_PARAMS; i++) {
                if (conv[i] == 1 && first_conv_window[i] < 0)
                    first_conv_window[i] = n_windows;
            }

            /* Check if all free params are converged */
            if (first_all_conv < 0) {
                int all = 1;
                for (int i = 0; i < N_PARAMS; i++) {
                    if (free_mask[i] && conv[i] != 1) { all = 0; break; }
                }
                if (all) first_all_conv = n_windows;
            }

            /* Check μ_base never reverts (latch test) */
            if (first_conv_window[3] >= 0 && conv[3] != 1)
                mu_reverted = 1;

            /* Print row */
            printf("  %3d %5d |", n_windows, t);
            if (!rpt.ready) {
                for (int i = 0; i < N_PARAMS; i++) printf("    — ");
            } else {
                for (int i = 0; i < N_PARAMS; i++) {
                    if (rpt.converged[i] == -1)
                        printf("  lock");
                    else if (rpt.converged[i] == 1)
                        printf(" %5.2f", rpt.rhat[i]);
                    else
                        printf(" \033[91m%5.2f\033[0m", rpt.rhat[i]);
                }
            }
            printf(" | %4.1f %5.3f %5.0f",
                   rpt.mahal_mean, rpt.p_trace_current, snap.last_ess);
            printf(" | %7.3f  %6.4f",
                   snap.theta[3], snap.theta[0]);

            /* Show if value is gated or learned */
            int is_mu_gated  = (conv[3] != 1);
            int is_rho_gated = (conv[0] != 1);
            printf(" %s%s",
                   is_mu_gated  ? " μ=prior" : "",
                   is_rho_gated ? " ρ=prior" : "");

            /* Count actual gate-converged free params */
            int n_gate_conv = 0, n_free = 0;
            for (int i = 0; i < N_PARAMS; i++) {
                if (free_mask[i]) { n_free++; if (conv[i] == 1) n_gate_conv++; }
            }
            int all_free_conv = (n_gate_conv == n_free && n_free > 0);

            printf(" | %d/%d", n_gate_conv, n_free);
            if (all_free_conv) printf(" ✓");
            printf("\n");

            /* Compare gated vs Kalman for the first few windows */
            if (n_windows <= 3 || n_windows == first_all_conv) {
                printf("       gated: [");
                for (int i = 0; i < 4; i++) printf(" %.3f", snap.theta[i]);
                printf(" ]\n");
                printf("       kalman:[");
                for (int i = 0; i < 4; i++) printf(" %.3f", kalman_x[i]);
                printf(" ]\n");
                printf("       truth: [");
                float truth4[4] = {dgp.rho, dgp.sigma_z, 0.5f, dgp.mu_base};
                /* Note: truth for σ_total ≈ dgp.sigma_z since DGP doesn't
                 * use (σ_total, r) directly — σ_z is the relevant param.
                 * The r_split truth depends on σ_base. */
                for (int i = 0; i < 4; i++) printf(" %.3f", truth4[i]);
                printf(" ]\n");
            }
        }
    }

    /* ── Convergence summary ─────────────────────────────────────────── */
    printf("\n  ────────────────────────────────────────────────────────────────────\n");
    printf("  CONVERGENCE SUMMARY\n");
    printf("  ────────────────────────────────────────────────────────────────────\n\n");

    printf("  %-12s  %-5s  %-8s  %-15s  %-15s  %-12s\n",
           "Param", "Gate", "Status", "First Conv", "Prior Default", "Final Value");
    printf("  ──────────── ───── ──────── ─────────────── ─────────────── ────────────\n");
    int conv_final[N_PARAMS];
    param_tracker_get_converged(tracker, conv_final);
    ParamSnapshot snap_final;
    param_tracker_get_snapshot(tracker, &snap_final);
    float kalman_final[N_PARAMS];
    param_tracker_get_kalman_x(tracker, kalman_final);

    static const char* GMODE_FULL[] = {"KAL", "R̂L", "LCK"};
    for (int i = 0; i < N_PARAMS; i++) {
        const char* status = (conv_final[i] == 1) ? "CONV" :
                             (conv_final[i] == -1) ? "LOCKED" : "—";
        int gmode = (i < 4) ? GATE_KALMAN_MIN : GATE_LOCKED;
        printf("  %-12s  %-5s  %-8s", PNAMES[i], GMODE_FULL[gmode], status);
        if (first_conv_window[i] >= 0)
            printf("  win %-10d", first_conv_window[i]);
        else
            printf("  %-15s", "never");
        printf("  %12.4f  %12.4f", prior_def[i], snap_final.theta[i]);
        if (conv_final[i] == 1)
            printf("  (kalman=%.4f)", kalman_final[i]);
        printf("\n");
    }

    if (first_all_conv >= 0)
        printf("\n  All free params converged at window %d (tick %d)\n",
               first_all_conv, first_all_conv * stride);
    else
        printf("\n  ⚠ Not all free params converged within %d windows\n", n_windows);

    /* ── RMSE ────────────────────────────────────────────────────────── */
    printf("\n  ────────────────────────────────────────────────────────────────────\n");
    printf("  RMSE\n");
    printf("  ────────────────────────────────────────────────────────────────────\n\n");

    /* Total RMSE */
    {
        double sum_sq = 0; int count = 0;
        for (int t = 0; t < N; t++) {
            if (!std::isnan(est_h[t]) && !std::isinf(est_h[t])) {
                double err = (double)est_h[t] - (double)gd.true_h[t];
                sum_sq += err * err; count++;
            }
        }
        printf("  Total RMSE:      %.4f  (%d ticks)\n",
               (count > 0) ? sqrt(sum_sq / count) : 999.0, count);
    }

    /* Scored RMSE (crisis only) */
    double scored_rmse = 999.0;
    {
        double sum_sq = 0; int count = 0;
        int start = (gd.score_start >= 0) ? gd.score_start : 0;
        for (int t = start; t < N; t++) {
            if (!std::isnan(est_h[t]) && !std::isinf(est_h[t])) {
                double err = (double)est_h[t] - (double)gd.true_h[t];
                sum_sq += err * err; count++;
            }
        }
        scored_rmse = (count > 0) ? sqrt(sum_sq / count) : 999.0;
        printf("  Scored RMSE:     %.4f  (%d ticks, crisis only)\n",
               scored_rmse, count);
    }

    /* Pre-convergence RMSE (before all free params converge) */
    if (first_all_conv > 0) {
        int conv_tick = first_all_conv * stride;
        double sum_sq = 0; int count = 0;
        for (int t = 0; t < conv_tick && t < N; t++) {
            if (!std::isnan(est_h[t]) && !std::isinf(est_h[t])) {
                double err = (double)est_h[t] - (double)gd.true_h[t];
                sum_sq += err * err; count++;
            }
        }
        printf("  Pre-conv RMSE:   %.4f  (%d ticks, before all converged)\n",
               (count > 0) ? sqrt(sum_sq / count) : 999.0, count);
    }

    /* Post-convergence RMSE */
    if (first_all_conv > 0) {
        int conv_tick = first_all_conv * stride;
        double sum_sq = 0; int count = 0;
        for (int t = conv_tick; t < N; t++) {
            if (!std::isnan(est_h[t]) && !std::isinf(est_h[t])) {
                double err = (double)est_h[t] - (double)gd.true_h[t];
                sum_sq += err * err; count++;
            }
        }
        printf("  Post-conv RMSE:  %.4f  (%d ticks, after all converged)\n",
               (count > 0) ? sqrt(sum_sq / count) : 999.0, count);
    }

    /* ── Verdict ─────────────────────────────────────────────────────── */
    printf("\n  ────────────────────────────────────────────────────────────────────\n");
    printf("  VERDICT\n");
    printf("  ────────────────────────────────────────────────────────────────────\n\n");

    int pass = 1;

    /* Check 1: Phase 1 params should converge at window 2 (GATE_KALMAN_MIN) */
    int phase1_ok = 1;
    for (int i = 0; i < 4; i++) {
        if (first_conv_window[i] != 2) { phase1_ok = 0; break; }
    }
    if (phase1_ok) {
        printf("  ✓ Phase 1 params (ρ, σ_total, r, μ_base) converged at window 2\n");
    } else {
        printf("  ✗ FAIL: Phase 1 params did not converge at window 2\n");
        for (int i = 0; i < 4; i++)
            printf("      %s: window %d\n", PNAMES[i], first_conv_window[i]);
        pass = 0;
    }

    /* Check 2: μ_base was gated for exactly 1 window (prior held, then Kalman) */
    if (first_conv_window[3] == 2) {
        printf("  ✓ μ_base gated for 1 window (prior default held), then pushed Kalman\n");
    } else {
        printf("  ~ μ_base converged at window %d (expected 2)\n", first_conv_window[3]);
    }

    /* Check 3: μ_base NEVER reverted to prior after converging (latch test) */
    if (!mu_reverted) {
        printf("  ✓ μ_base never reverted to prior after convergence (latch works)\n");
    } else {
        printf("  ✗ FAIL: μ_base reverted to prior after convergence (latch broken)\n");
        pass = 0;
    }

    /* Check 4: locked params should show -1 convergence and prior values */
    int locked_ok = 1;
    for (int i = 4; i < N_PARAMS; i++) {
        if (conv_final[i] != -1) { locked_ok = 0; break; }
        if (fabsf(snap_final.theta[i] - prior_def[i]) > 1e-4f) { locked_ok = 0; break; }
    }
    if (locked_ok) {
        printf("  ✓ Locked params (4-7) held at prior defaults throughout\n");
    } else {
        printf("  ✗ FAIL: locked params were modified\n");
        pass = 0;
    }

    /* Check 5: scored RMSE sanity */
    if (scored_rmse < 5.0) {
        printf("  ✓ Scored RMSE = %.4f (reasonable)\n", scored_rmse);
    } else {
        printf("  ✗ FAIL: scored RMSE = %.4f (too high, gating may be broken)\n",
               scored_rmse);
        pass = 0;
    }

    /* Check 6: after convergence, gated values should track Kalman (not prior) */
    float mu_gap = fabsf(snap_final.theta[3] - kalman_final[3]);
    if (mu_gap < 0.01f) {
        printf("  ✓ Final μ_base: gated=%.4f matches kalman=%.4f\n",
               snap_final.theta[3], kalman_final[3]);
    } else {
        printf("  ✗ FAIL: μ_base gated=%.4f != kalman=%.4f (gap=%.4f)\n",
               snap_final.theta[3], kalman_final[3], mu_gap);
        pass = 0;
    }

    printf("\n  ════════════════════════════════════════════════════════════════════\n");
    if (pass)
        printf("  ALL CHECKS PASSED ✓\n");
    else
        printf("  SOME CHECKS FAILED ✗\n");
    printf("  ════════════════════════════════════════════════════════════════════\n\n");

    /* ── Cleanup ─────────────────────────────────────────────────────── */
    param_tracker_destroy(tracker);
    gpu_bpf_destroy(bpf);

    return pass ? 0 : 1;
}
