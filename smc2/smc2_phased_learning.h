/*═══════════════════════════════════════════════════════════════════════════════
 * @file smc2_phased_learning.h
 * @brief Sequential identification controller for sat_exp curve parameters
 *
 * Manages which SMC² parameters are active vs fixed, advancing through
 * learning phases as the observed z-range expands:
 *
 *   Phase 1 (calm):  Learn floors only (3 params: ρ, σ_total, r_split + μ_floor)
 *   Phase 2 (stress): + ceilings (6 params: + μ_ceiling, σ_ceiling)
 *   Phase 3 (full):   + rates (8 params: + μ_rate, σ_rate)
 *
 * Floor/ceiling reparameterization:
 *   floor   = base                    (value at z = 0)
 *   ceiling = base + scale            (value at z → ∞)
 *   scale   = ceiling - floor         (derived, not proposed directly)
 *
 * This eliminates the base/scale ridge: floor and ceiling are pinned at
 * opposite ends of z-space, making them orthogonal in the likelihood.
 *
 * The controller sits on top of the existing SMC² fixed_mask mechanism.
 * No kernel changes required.
 *
 * Usage:
 *   PhasedLearner* pl = phased_create(smc2, config);
 *   // ... per window:
 *   phased_observe_z(pl, z_mean, z_min, z_max);
 *   phased_update(pl);  // may advance phase and update masks
 *═══════════════════════════════════════════════════════════════════════════════*/

#ifndef SMC2_PHASED_LEARNING_H
#define SMC2_PHASED_LEARNING_H

#include "smc2_rbpf_batch.cuh"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>

/* ── Parameter indices (must match ThetaParticlesSoA order) ─────────────── */

enum ParamIdx {
    P_RHO          = 0,
    P_SIGMA_TOTAL  = 1,
    P_R_SPLIT      = 2,
    P_MU_BASE      = 3,   /* = μ_floor */
    P_MU_SCALE     = 4,   /* = μ_ceiling - μ_floor */
    P_MU_RATE      = 5,
    P_SIGMA_SCALE  = 6,   /* = σ_ceiling - σ_floor (σ_floor is derived) */
    P_SIGMA_RATE   = 7
};

/* ── Learning phases ────────────────────────────────────────────────────── */

enum LearningPhase {
    PHASE_1_FLOORS   = 1,  /* ρ, σ_total, r_split, μ_base (floors only)      */
    PHASE_2_CEILINGS = 2,  /* + μ_scale, σ_scale (floor+ceiling, rates fixed) */
    PHASE_3_RATES    = 3   /* + μ_rate, σ_rate (all 8 free)                   */
};

/* ── Z-range tracker ────────────────────────────────────────────────────── */

typedef struct {
    float z_min_seen;         /**< Minimum z observed across all windows     */
    float z_max_seen;         /**< Maximum z observed across all windows     */
    float z_ema;              /**< Exponential moving average of z           */
    int   high_z_count;       /**< Consecutive windows with z_max > threshold */
    int   full_cycle_count;   /**< Windows where both low and high z present  */
    int   n_windows;          /**< Total windows observed                     */
} ZRangeTracker;

/* ── Configuration ──────────────────────────────────────────────────────── */

typedef struct {
    /* Phase 1 → 2 trigger */
    float ceiling_z_threshold;    /**< z_max must exceed this (default: 2.5)      */
    int   ceiling_z_sustained;    /**< For this many consecutive windows (def: 3) */

    /* Phase 2 → 3 trigger */
    float rate_z_low;             /**< Low end of transition region (def: 0.5)    */
    float rate_z_high;            /**< High end (def: 2.5)                        */
    int   rate_cycles_required;   /**< Full low→high cycles needed (def: 2)       */

    /* Identification check */
    int   enable_ident_check;     /**< Require likelihood sensitivity? (def: 1)   */
    float ident_ll_delta;         /**< Min log-lik change for param to be identified */

    /* Fixed values for locked parameters */
    float fixed_mu_scale;         /**< μ_scale when locked in Phase 1              */
    float fixed_mu_rate;          /**< μ_rate when locked in Phase 1-2             */
    float fixed_sigma_scale;      /**< σ_scale when locked in Phase 1              */
    float fixed_sigma_rate;       /**< σ_rate when locked in Phase 1-2             */

    /* EMA smoothing for z tracker */
    float z_ema_alpha;            /**< EMA decay (default: 0.1)                    */
} PhasedConfig;

static inline PhasedConfig phased_default_config(void) {
    PhasedConfig c;
    c.ceiling_z_threshold  = 2.5f;
    c.ceiling_z_sustained  = 3;
    c.rate_z_low           = 0.5f;
    c.rate_z_high          = 2.5f;
    c.rate_cycles_required = 2;
    c.enable_ident_check   = 1;
    c.ident_ll_delta       = 2.0f;   /* ~2 nats = meaningful */
    c.fixed_mu_scale       = 2.0f;   /* Prior: moderate ceiling-floor gap */
    c.fixed_mu_rate        = 0.3f;   /* Prior: gradual transition */
    c.fixed_sigma_scale    = 0.5f;
    c.fixed_sigma_rate     = 0.3f;
    c.z_ema_alpha          = 0.1f;
    return c;
}

/* ── Phased learner state ───────────────────────────────────────────────── */

typedef struct {
    SMC2StateCUDA*   smc2;          /**< The SMC² system we control          */
    PhasedConfig     config;
    LearningPhase    phase;
    ZRangeTracker    z_tracker;

    /* Identification scores (log-lik sensitivity per param) */
    float            ident_score[N_PARAMS];

    /* Phase transition history */
    int              phase2_entered_at;  /**< Window index when Phase 2 entered */
    int              phase3_entered_at;  /**< Window index when Phase 3 entered */
} PhasedLearner;

/* Forward declarations */
static inline void phased_apply_mask(PhasedLearner* pl);

/* ── Create / destroy ───────────────────────────────────────────────────── */

static inline PhasedLearner* phased_create(
    SMC2StateCUDA* smc2,
    PhasedConfig   config
) {
    PhasedLearner* pl = (PhasedLearner*)calloc(1, sizeof(PhasedLearner));
    pl->smc2   = smc2;
    pl->config = config;
    pl->phase  = PHASE_1_FLOORS;

    pl->z_tracker.z_min_seen = 1e6f;
    pl->z_tracker.z_max_seen = -1e6f;
    pl->z_tracker.z_ema      = 0.0f;

    pl->phase2_entered_at = -1;
    pl->phase3_entered_at = -1;

    /* Apply Phase 1 mask immediately */
    phased_apply_mask(pl);

    return pl;
}

static inline void phased_destroy(PhasedLearner* pl) {
    free(pl);
}

/* ── Build and apply parameter mask for current phase ───────────────────── */

static inline void phased_apply_mask(PhasedLearner* pl) {
    uint8_t mask[N_PARAMS];
    float   values[N_PARAMS];

    memset(mask, 0, sizeof(mask));
    memset(values, 0, sizeof(values));

    switch (pl->phase) {
    case PHASE_1_FLOORS:
        /* Free: ρ, σ_total, r_split, μ_base (4 params)
         * Fixed: μ_scale, μ_rate, σ_scale, σ_rate              */
        mask[P_MU_SCALE]    = 1;  values[P_MU_SCALE]    = pl->config.fixed_mu_scale;
        mask[P_MU_RATE]     = 1;  values[P_MU_RATE]     = pl->config.fixed_mu_rate;
        mask[P_SIGMA_SCALE] = 1;  values[P_SIGMA_SCALE] = pl->config.fixed_sigma_scale;
        mask[P_SIGMA_RATE]  = 1;  values[P_SIGMA_RATE]  = pl->config.fixed_sigma_rate;
        break;

    case PHASE_2_CEILINGS:
        /* Free: ρ, σ_total, r_split, μ_base, μ_scale, σ_scale (6 params)
         * Fixed: μ_rate, σ_rate                                  */
        mask[P_MU_RATE]     = 1;  values[P_MU_RATE]     = pl->config.fixed_mu_rate;
        mask[P_SIGMA_RATE]  = 1;  values[P_SIGMA_RATE]  = pl->config.fixed_sigma_rate;
        break;

    case PHASE_3_RATES:
        /* All 8 free */
        break;
    }

    smc2_cuda_set_fixed_params(pl->smc2, mask, values);
}

/* ── Feed z observations from a completed window ────────────────────────── */

static inline void phased_observe_z(
    PhasedLearner* pl,
    float z_mean,       /**< Weighted mean z from this window        */
    float z_min,        /**< Min z observed in this window           */
    float z_max         /**< Max z observed in this window           */
) {
    ZRangeTracker* zt = &pl->z_tracker;

    /* Update global extremes */
    if (z_min < zt->z_min_seen) zt->z_min_seen = z_min;
    if (z_max > zt->z_max_seen) zt->z_max_seen = z_max;

    /* EMA of z */
    if (zt->n_windows == 0) {
        zt->z_ema = z_mean;
    } else {
        zt->z_ema += pl->config.z_ema_alpha * (z_mean - zt->z_ema);
    }

    /* Count consecutive high-z windows */
    if (z_max > pl->config.ceiling_z_threshold) {
        zt->high_z_count++;
    } else {
        zt->high_z_count = 0;
    }

    /* Count full-cycle windows (both low and high z present) */
    if (z_min < pl->config.rate_z_low && z_max > pl->config.rate_z_high) {
        zt->full_cycle_count++;
    }

    zt->n_windows++;
}

/* ── Feed z range from inner particles (more granular) ──────────────────── */

static inline void phased_observe_z_from_smc2(PhasedLearner* pl) {
    float z_mean, z_min, z_max;
    smc2_cuda_get_z_range(pl->smc2, &z_mean, &z_min, &z_max);
    phased_observe_z(pl, z_mean, z_min, z_max);
}

/* ── Identification check: does perturbing a param change likelihood? ────
 *
 * Compute numerical sensitivity: for each locked param, compare
 * log-likelihood at current value vs current ± δ. If the difference
 * exceeds ident_ll_delta, the param is identified by the data.
 *
 * This is called before advancing phases to prevent premature unlocking.
 * ──────────────────────────────────────────────────────────────────────── */

static inline float phased_ident_score_param(
    PhasedLearner* pl,
    int param_idx,
    float delta
) {
    /*
     * Sensitivity approximation from particle cloud:
     *
     * For a fixed param, all θ-particles have the same value.
     * We can't compute sensitivity directly without extra likelihood
     * evaluations. Instead, use the prior predictive:
     *
     * If the empirical covariance shows the param has near-zero variance
     * (all particles collapsed to the fixed value), and perturbing it
     * would change the adaptive covariance structure, the param is
     * identified.
     *
     * Practical approach: check if z-range covers the region where this
     * param matters. Floors matter at z≈0, ceilings at z→∞, rates
     * in the transition region.
     *
     * TODO: Implement proper finite-difference likelihood sensitivity
     * by running CPMMH with param perturbed ±δ and comparing LL.
     * For now, use z-range as proxy.
     */
    ZRangeTracker* zt = &pl->z_tracker;

    switch (param_idx) {
    case P_MU_SCALE:
    case P_SIGMA_SCALE:
        /* Ceilings: identified when high z observed */
        return zt->z_max_seen;

    case P_MU_RATE:
    case P_SIGMA_RATE:
        /* Rates: identified when transition region observed */
        if (zt->z_min_seen < pl->config.rate_z_low &&
            zt->z_max_seen > pl->config.rate_z_high) {
            return zt->z_max_seen - zt->z_min_seen;  /* z span */
        }
        return 0.0f;

    default:
        /* Floor params — always identified */
        return 10.0f;
    }
}

/* ── Phase transition logic ─────────────────────────────────────────────── */

static inline int phased_update(PhasedLearner* pl) {
    LearningPhase prev_phase = pl->phase;
    ZRangeTracker* zt = &pl->z_tracker;

    switch (pl->phase) {
    case PHASE_1_FLOORS:
        /*
         * Advance to Phase 2 when:
         *   - z_max > ceiling_z_threshold for ceiling_z_sustained consecutive windows
         *   - (optional) identification check passes for ceiling params
         */
        if (zt->high_z_count >= pl->config.ceiling_z_sustained) {
            int identified = 1;
            if (pl->config.enable_ident_check) {
                float score_mu  = phased_ident_score_param(pl, P_MU_SCALE, 0.5f);
                float score_sig = phased_ident_score_param(pl, P_SIGMA_SCALE, 0.3f);
                identified = (score_mu > pl->config.ceiling_z_threshold &&
                              score_sig > pl->config.ceiling_z_threshold);
            }
            if (identified) {
                pl->phase = PHASE_2_CEILINGS;
                pl->phase2_entered_at = zt->n_windows;
                phased_apply_mask(pl);
            }
        }
        break;

    case PHASE_2_CEILINGS:
        /*
         * Advance to Phase 3 when:
         *   - Enough full z-cycles observed (low→high)
         *   - Rate params are identifiable from transition region data
         */
        if (zt->full_cycle_count >= pl->config.rate_cycles_required) {
            int identified = 1;
            if (pl->config.enable_ident_check) {
                float score_mr = phased_ident_score_param(pl, P_MU_RATE, 0.1f);
                float score_sr = phased_ident_score_param(pl, P_SIGMA_RATE, 0.1f);
                identified = (score_mr > pl->config.ident_ll_delta &&
                              score_sr > pl->config.ident_ll_delta);
            }
            if (identified) {
                pl->phase = PHASE_3_RATES;
                pl->phase3_entered_at = zt->n_windows;
                phased_apply_mask(pl);
            }
        }
        break;

    case PHASE_3_RATES:
        /* Terminal phase — never go backward */
        break;
    }

    int advanced = (pl->phase != prev_phase);
    if (advanced) {
        printf("[PHASED] Phase %d → %d at window %d "
               "(z_range: [%.2f, %.2f], high_z_count: %d, cycles: %d)\n",
               prev_phase, pl->phase, zt->n_windows,
               zt->z_min_seen, zt->z_max_seen,
               zt->high_z_count, zt->full_cycle_count);
    }

    return advanced;
}

/* ── Floor/ceiling helpers ──────────────────────────────────────────────── */

/** Convert (floor, ceiling) to internal (base, scale) representation */
static inline void floor_ceiling_to_base_scale(
    float floor, float ceiling,
    float* base, float* scale
) {
    *base  = floor;
    *scale = ceiling - floor;
}

/** Convert internal (base, scale) to (floor, ceiling) */
static inline void base_scale_to_floor_ceiling(
    float base, float scale,
    float* floor, float* ceiling
) {
    *floor   = base;
    *ceiling = base + scale;
}

/* ── Override fixed values mid-run ──────────────────────────────────────── */

/**
 * @brief Update the fixed values for locked parameters.
 *
 * Call this when offline calibration provides better rate estimates,
 * or when SMC² from a previous epoch has learned curve shape.
 */
static inline void phased_set_fixed_rates(
    PhasedLearner* pl,
    float mu_rate,
    float sigma_rate
) {
    pl->config.fixed_mu_rate    = mu_rate;
    pl->config.fixed_sigma_rate = sigma_rate;

    /* Re-apply if rates are currently fixed */
    if (pl->phase < PHASE_3_RATES) {
        phased_apply_mask(pl);
    }
}

static inline void phased_set_fixed_ceilings(
    PhasedLearner* pl,
    float mu_scale,
    float sigma_scale
) {
    pl->config.fixed_mu_scale    = mu_scale;
    pl->config.fixed_sigma_scale = sigma_scale;

    /* Re-apply if ceilings are currently fixed */
    if (pl->phase < PHASE_2_CEILINGS) {
        phased_apply_mask(pl);
    }
}

/* ── Diagnostics ────────────────────────────────────────────────────────── */

static inline void phased_print_status(const PhasedLearner* pl) {
    const char* phase_names[] = {"???", "FLOORS (4p)", "CEILINGS (6p)", "RATES (8p)"};
    const ZRangeTracker* zt = &pl->z_tracker;

    printf("Phased Learning Status:\n");
    printf("  Phase:          %d — %s\n", pl->phase, phase_names[pl->phase]);
    printf("  Windows:        %d\n", zt->n_windows);
    printf("  z_range:        [%.3f, %.3f]\n", zt->z_min_seen, zt->z_max_seen);
    printf("  z_ema:          %.3f\n", zt->z_ema);
    printf("  high_z_streak:  %d / %d needed\n",
           zt->high_z_count, pl->config.ceiling_z_sustained);
    printf("  full_cycles:    %d / %d needed\n",
           zt->full_cycle_count, pl->config.rate_cycles_required);

    if (pl->phase2_entered_at >= 0)
        printf("  Phase 2 at:     window %d\n", pl->phase2_entered_at);
    if (pl->phase3_entered_at >= 0)
        printf("  Phase 3 at:     window %d\n", pl->phase3_entered_at);

    printf("  Active params:  ");
    uint8_t mask[N_PARAMS];
    memcpy(mask, pl->smc2->fixed_mask, N_PARAMS);
    const char* names[] = {"ρ", "σ_tot", "r", "μ_base", "μ_scl", "μ_rate", "σ_scl", "σ_rate"};
    for (int i = 0; i < N_PARAMS; i++) {
        if (!mask[i]) printf("%s ", names[i]);
    }
    printf("\n  Fixed params:   ");
    for (int i = 0; i < N_PARAMS; i++) {
        if (mask[i]) printf("%s=%.3f ", names[i], pl->smc2->fixed_values[i]);
    }
    printf("\n");
}

/* ── Integration with pipeline ──────────────────────────────────────────── */

/**
 * @brief Call after each SMC² window completes.
 *
 * Typical usage inside the pipeline:
 *
 *   smc2_cuda_process_window(smc2, obs, W, stream);
 *   cudaStreamSynchronize(stream);
 *
 *   // Feed z stats to phased controller
 *   float z_mean = smc2_cuda_get_z_mean(smc2);
 *   phased_observe_z(pl, z_mean, z_min, z_max);
 *
 *   // Check for phase advancement
 *   if (phased_update(pl)) {
 *       printf("Advanced to phase %d!\n", pl->phase);
 *   }
 *
 *   // Push params to dBPF (pipeline handles this)
 *   pipeline_push_params(pipeline);
 */

#endif /* SMC2_PHASED_LEARNING_H */
