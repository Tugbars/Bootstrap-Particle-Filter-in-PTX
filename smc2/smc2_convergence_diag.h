/**
 * @file smc2_convergence_diag.h
 * @brief Window-based convergence diagnostics for amnesiac SMC² + Kalman
 *
 * Analogous to Gelman-Rubin R̂ for MCMC chains, using amnesiac SMC² windows
 * as independent "chains". Each window reinitializes from prior and produces
 * (θ̂_k, Σ_k) independently — their agreement diagnoses convergence.
 *
 * Diagnostics:
 *
 *   1. R̂ (per-param)    √(B/W) — between-window variance vs mean posterior
 *                        variance. At convergence with overlapping windows
 *                        (50% overlap): R̂ ∈ [0.5, 1.3]. R̂ > 2.0 signals
 *                        non-convergence.
 *
 *   2. Mahalanobis d²    Kalman innovation ν'S⁻¹ν. Under correct model,
 *                        d² ~ χ²(n_free). Mean should ≈ n_free.
 *
 *   3. P-trace           tr(P) — total Kalman uncertainty. Should decrease
 *                        monotonically and plateau at the information floor
 *                        set by Q.
 *
 *   4. CV (per-param)    √P_ii / |x_i| — relative uncertainty. Below 5%
 *                        means the Kalman is confident on that parameter.
 *
 * Usage:
 *   ConvergenceDiag diag;
 *   conv_diag_init(&diag, 8);
 *
 *   // After each param_tracker_run_window():
 *   conv_diag_push(&diag, theta_hat, sigma_diag, mahal_d2, p_trace);
 *
 *   // Query:
 *   ConvergenceReport rpt;
 *   conv_diag_report(&diag, kalman_x, kalman_P_diag, free_mask, 1.5f, &rpt);
 *   if (rpt.all_converged) { ... }
 *
 * Header-only, pure C, no CUDA dependencies.
 */

#ifndef SMC2_CONVERGENCE_DIAG_H
#define SMC2_CONVERGENCE_DIAG_H

#include <math.h>
#include <string.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * Constants
 * ═══════════════════════════════════════════════════════════════════════════ */

#ifndef CONV_DIAG_N_PARAMS
#define CONV_DIAG_N_PARAMS  8
#endif

#define CONV_DIAG_MAX_BUF   16   /* Max rolling window of posteriors */

/* ═══════════════════════════════════════════════════════════════════════════
 * Structures
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    /* Circular buffer of window posteriors */
    float theta_hat[CONV_DIAG_MAX_BUF][CONV_DIAG_N_PARAMS]; /* posterior mean  */
    float sigma_diag[CONV_DIAG_MAX_BUF][CONV_DIAG_N_PARAMS];/* diag(Σ_k)      */
    float mahal_d2[CONV_DIAG_MAX_BUF];                       /* Mahalanobis d²  */
    float p_trace[CONV_DIAG_MAX_BUF];                        /* tr(P) snapshot  */

    int count;   /* Total windows pushed (monotonically increasing) */
    int head;    /* Next write index (circular, mod M)              */
    int M;       /* Rolling buffer size                             */
} ConvergenceDiag;

typedef struct ConvergenceReport {
    /* Per-parameter diagnostics */
    float rhat[CONV_DIAG_N_PARAMS];       /* √(B/W) — window R̂               */
    float B[CONV_DIAG_N_PARAMS];          /* Between-window variance           */
    float W[CONV_DIAG_N_PARAMS];          /* Mean within-window variance       */
    float cv[CONV_DIAG_N_PARAMS];         /* √P_ii / |x_i|                    */
    int   converged[CONV_DIAG_N_PARAMS];  /* 1=yes, 0=no, -1=locked           */

    /* Aggregate */
    float mahal_mean;       /* Mean d² over buffer                             */
    float mahal_expected;   /* Expected d² = n_free (χ² dof)                   */
    float p_trace_current;  /* Latest tr(P)                                    */
    float p_trace_prev;     /* Previous tr(P)                                  */
    int   n_free;           /* Number of free params checked                   */
    int   n_converged;      /* Free params with R̂ < threshold                 */
    int   all_converged;    /* 1 if n_converged == n_free && n_free > 0        */
    int   ready;            /* 1 if count >= M (buffer full)                   */
} ConvergenceReport;

/* ═══════════════════════════════════════════════════════════════════════════
 * Init
 * ═══════════════════════════════════════════════════════════════════════════ */

static inline void conv_diag_init(ConvergenceDiag* d, int M)
{
    memset(d, 0, sizeof(*d));
    d->M = (M > CONV_DIAG_MAX_BUF) ? CONV_DIAG_MAX_BUF : (M < 2 ? 2 : M);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Push one window's results into the rolling buffer
 *
 * theta_hat  — posterior mean θ̂_k [N_PARAMS]
 * sigma_diag — diagonal of posterior covariance Σ_k [N_PARAMS]
 * mahal_d2   — Mahalanobis distance ν'S⁻¹ν (diagonal approx is fine)
 * p_trace    — tr(P) from Kalman after this update
 * ═══════════════════════════════════════════════════════════════════════════ */

static inline void conv_diag_push(ConvergenceDiag* d,
                                   const float theta_hat[],
                                   const float sigma_diag[],
                                   float mahal_d2,
                                   float p_trace)
{
    int idx = d->head;
    memcpy(d->theta_hat[idx],  theta_hat,  CONV_DIAG_N_PARAMS * sizeof(float));
    memcpy(d->sigma_diag[idx], sigma_diag, CONV_DIAG_N_PARAMS * sizeof(float));
    d->mahal_d2[idx] = mahal_d2;
    d->p_trace[idx]  = p_trace;

    d->head = (d->head + 1) % d->M;
    d->count++;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Compute convergence report from the rolling buffer
 *
 * kalman_x      — current Kalman state x [N_PARAMS]  (for CV)
 * kalman_P_diag — current diag(P) [N_PARAMS]          (for CV)
 * free_mask     — 1 = free (check convergence), 0 = locked (skip)
 * rhat_thresh   — R̂ threshold (suggest 1.5)
 * ═══════════════════════════════════════════════════════════════════════════ */

static inline void conv_diag_report(const ConvergenceDiag* d,
                                     const float kalman_x[],
                                     const float kalman_P_diag[],
                                     const int free_mask[],
                                     float rhat_thresh,
                                     ConvergenceReport* r)
{
    memset(r, 0, sizeof(*r));

    int n = (d->count < d->M) ? d->count : d->M;
    r->ready = (n >= d->M);

    if (n < 2) return;  /* Need ≥2 windows for variance */

    /* ── Per-parameter: B, W, R̂ ────────────────────────────────────── */
    for (int i = 0; i < CONV_DIAG_N_PARAMS; i++) {
        float sum = 0.0f, sum2 = 0.0f, sumW = 0.0f;
        for (int k = 0; k < n; k++) {
            float v = d->theta_hat[k][i];
            sum  += v;
            sum2 += v * v;
            sumW += d->sigma_diag[k][i];
        }

        float mean = sum / (float)n;
        float B = (sum2 / (float)n) - mean * mean;   /* var of window means */
        float W = sumW / (float)n;                     /* mean of window vars */

        /* Bessel correction for small n */
        if (n > 1) B *= (float)n / (float)(n - 1);

        r->B[i] = B;
        r->W[i] = W;

        if (W > 1e-12f) {
            r->rhat[i] = sqrtf(B / W);
        } else {
            r->rhat[i] = (B > 1e-12f) ? 99.0f : 0.0f;
        }

        /* CV from Kalman state */
        if (fabsf(kalman_x[i]) > 1e-8f) {
            r->cv[i] = sqrtf(kalman_P_diag[i]) / fabsf(kalman_x[i]);
        } else {
            r->cv[i] = (kalman_P_diag[i] > 1e-12f) ? 99.0f : 0.0f;
        }

        /* Per-param convergence (only free params) */
        if (free_mask[i]) {
            r->n_free++;
            r->converged[i] = (r->rhat[i] < rhat_thresh) ? 1 : 0;
            if (r->converged[i]) r->n_converged++;
        } else {
            r->converged[i] = -1;  /* locked — not checked */
        }
    }

    r->all_converged = (r->n_free > 0 && r->n_converged == r->n_free);

    /* ── Mahalanobis mean ──────────────────────────────────────────── */
    float msum = 0.0f;
    for (int k = 0; k < n; k++) msum += d->mahal_d2[k];
    r->mahal_mean     = msum / (float)n;
    r->mahal_expected  = (float)r->n_free;

    /* ── P-trace ───────────────────────────────────────────────────── */
    int last = (d->head - 1 + d->M) % d->M;
    r->p_trace_current = d->p_trace[last];
    if (n >= 2) {
        int prev = (d->head - 2 + d->M) % d->M;
        r->p_trace_prev = d->p_trace[prev];
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Compute diagonal Mahalanobis d² from Kalman innovation
 *
 * Helper for callers that don't want to invert the full S matrix.
 * Uses diagonal approximation: d² ≈ Σ_i ν_i² / S_ii
 *
 * nu     — innovation θ̂_k − x_predicted [N_PARAMS]
 * S_diag — diagonal of innovation cov P̄ + Σ_k [N_PARAMS]
 * ═══════════════════════════════════════════════════════════════════════════ */

static inline float conv_diag_mahal_diag(const float nu[],
                                          const float S_diag[],
                                          int n_params)
{
    float d2 = 0.0f;
    for (int i = 0; i < n_params; i++) {
        if (S_diag[i] > 1e-12f)
            d2 += nu[i] * nu[i] / S_diag[i];
    }
    return d2;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Print a one-line summary (for diagnostic output during runs)
 * ═══════════════════════════════════════════════════════════════════════════ */

static inline void conv_diag_print_line(const ConvergenceReport* r, int win_idx)
{
    printf(" %3d |", win_idx);

    if (!r->ready) {
        for (int i = 0; i < CONV_DIAG_N_PARAMS; i++) printf("    — ");
        printf("| d²=%5.1f | Ptr=%7.4f | —\n", r->mahal_mean, r->p_trace_current);
        return;
    }

    for (int i = 0; i < CONV_DIAG_N_PARAMS; i++) {
        if (r->converged[i] == -1)
            printf("  lock");
        else if (r->converged[i])
            printf(" %5.2f", r->rhat[i]);
        else
            printf(" %s%.2f%s", "\033[91m", r->rhat[i], "\033[0m");
    }

    printf(" | d²=%5.1f", r->mahal_mean);
    printf(" | Ptr=%7.4f", r->p_trace_current);
    printf(" | %d/%d", r->n_converged, r->n_free);
    if (r->all_converged) printf("  ✓");
    printf("\n");
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Print column header (matches conv_diag_print_line layout)
 * ═══════════════════════════════════════════════════════════════════════════ */

static inline void conv_diag_print_header(const char* param_names[])
{
    printf(" Win |");
    for (int i = 0; i < CONV_DIAG_N_PARAMS; i++)
        printf(" %5s", param_names[i]);
    printf(" | Mahal  | P-trace | Conv\n");

    printf("-----|");
    for (int i = 0; i < CONV_DIAG_N_PARAMS; i++) printf("------");
    printf("-|--------|---------|------\n");
}

#ifdef __cplusplus
}
#endif

#endif /* SMC2_CONVERGENCE_DIAG_H */