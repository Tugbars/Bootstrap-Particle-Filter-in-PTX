/*═══════════════════════════════════════════════════════════════════════════════
 * @file smc2_phased_learning.h
 * @brief Bidirectional identification controller for sat_exp curve parameters
 *
 * Manages which SMC² parameters are active vs fixed. The "valve" opens and
 * closes based on the z-range observed within each SMC² window:
 *
 *   Phase 1 (calm):   Learn floors only (4 params: ρ, σ_total, r_split, μ_base)
 *   Phase 2 (stress): + ceilings (6 params: + μ_scale, σ_scale)
 *   Phase 3 (full):   + rates (8 params: + μ_rate, σ_rate)
 *
 * Unlike v1 (one-way ratchet), phases can go BACKWARD:
 *
 *   Phase 3 → 2:  z-range narrows → lock rates to LEARNED values
 *   Phase 2 → 1:  z stays calm    → lock ceilings to LEARNED values
 *   Phase 1 → 2:  z rises         → unlock ceilings (starting from saved values)
 *   Phase 2 → 3:  z spans range   → unlock rates (starting from saved values)
 *
 * Why bidirectional?
 *   During calm periods, ceiling/rate params become unidentifiable. The particle
 *   cloud wastes diversity exploring the ridge manifold, ESS bleeds, and good
 *   estimates from previous crises get corrupted. Locking back:
 *     1. Preserves learned values (external memory for the cloud)
 *     2. Removes degenerate dimensions (healthy ESS)
 *     3. Reduces attack surface for cascading BPF↔SMC² failures
 *
 * The controller sits on top of the existing SMC² fixed_mask mechanism.
 * No kernel changes required.
 *
 * Usage:
 *   PhasedLearner* pl = phased_create(smc2, config);
 *   // ... per window:
 *   phased_observe_z_from_smc2(pl);
 *   phased_update(pl);  // may advance OR retreat phase and update masks
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
    P_SIGMA_SCALE  = 6,   /* = σ_ceiling - σ_floor */
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
    /* Per-window observations */
    float z_min_window;       /**< Min z in current window                    */
    float z_max_window;       /**< Max z in current window                    */
    float z_mean_window;      /**< Mean z in current window                   */

    /* Global statistics */
    float z_min_seen;         /**< Minimum z observed across all windows      */
    float z_max_seen;         /**< Maximum z observed across all windows      */
    float z_ema;              /**< Exponential moving average of z            */
    int   n_windows;          /**< Total windows observed                     */

    /* ── Forward triggers (unlock) ────────────────────────────────────── */
    int   high_z_streak;      /**< Consecutive windows with z_max > threshold */
    int   wide_range_streak;  /**< Consecutive windows with z_range > thresh  */

    /* ── Backward triggers (lock) ─────────────────────────────────────── */
    int   calm_streak;        /**< Consecutive windows with z_max < threshold */
    int   narrow_range_streak;/**< Consecutive windows with z_range < thresh  */
} ZRangeTracker;

/* ── Configuration ──────────────────────────────────────────────────────── */

typedef struct {
    /* ── Phase 1 ↔ 2 triggers (symmetric) ─────────────────────────────── */
    float ceiling_z_threshold;    /**< z_max threshold for unlock/lock (def: 2.0) */
    int   ceiling_z_sustained;    /**< Consecutive windows needed (def: 3)        */

    /* ── Phase 2 ↔ 3 triggers (symmetric) ─────────────────────────────── */
    float rate_z_range_threshold; /**< z_max - z_min threshold (def: 1.5)         */
    int   rate_range_sustained;   /**< Consecutive windows needed (def: 3)        */

    /* ── Fixed values for locked parameters ───────────────────────────── */
    /* These start as prior defaults, then get overwritten with learned    */
    /* values when the valve locks back.                                   */
    float fixed_mu_scale;         /**< μ_scale when locked                        */
    float fixed_mu_rate;          /**< μ_rate when locked                         */
    float fixed_sigma_scale;      /**< σ_scale when locked                        */
    float fixed_sigma_rate;       /**< σ_rate when locked                         */

    /* ── Flags ────────────────────────────────────────────────────────── */
    int   learned_ceilings;       /**< 1 if ceilings have been learned at least once */
    int   learned_rates;          /**< 1 if rates have been learned at least once    */

    /* EMA smoothing for z tracker */
    float z_ema_alpha;            /**< EMA decay (default: 0.1)                   */

    /* ── Backward transition control ──────────────────────────────── */
    int   enable_backward;        /**< 1=valve (bidir), 0=ratchet (one-way)       */
} PhasedConfig;

static inline PhasedConfig phased_default_config(void) {
    PhasedConfig c;

    /* Symmetric thresholds */
    c.ceiling_z_threshold   = 2.0f;
    c.ceiling_z_sustained   = 3;
    c.rate_z_range_threshold = 1.5f;
    c.rate_range_sustained  = 3;

    /* Prior defaults (used until first learning) */
    c.fixed_mu_scale       = 2.0f;   /* Moderate ceiling-floor gap */
    c.fixed_mu_rate        = 0.3f;   /* Gradual transition */
    c.fixed_sigma_scale    = 0.5f;
    c.fixed_sigma_rate     = 0.3f;

    c.learned_ceilings     = 0;
    c.learned_rates        = 0;

    c.z_ema_alpha          = 0.1f;
    c.enable_backward      = 1;      /* Bidirectional by default */
    return c;
}

/* ── Phase transition history entry ─────────────────────────────────────── */

typedef struct {
    int            window;
    LearningPhase  from;
    LearningPhase  to;
    float          z_min;
    float          z_max;
    float          z_range;
} PhaseTransition;

#define MAX_TRANSITIONS 64

/* ── Phased learner state ───────────────────────────────────────────────── */

typedef struct {
    SMC2StateCUDA*   smc2;          /**< The SMC² system we control          */
    PhasedConfig     config;
    LearningPhase    phase;
    ZRangeTracker    z_tracker;

    /* Phase transition history */
    PhaseTransition  history[MAX_TRANSITIONS];
    int              n_transitions;

    /* Convenience: first forward entries */
    int              phase2_entered_at;  /**< Window index when Phase 2 first entered */
    int              phase3_entered_at;  /**< Window index when Phase 3 first entered */
} PhasedLearner;

/* Forward declarations */
static inline void phased_apply_mask(PhasedLearner* pl);
static inline void phased_save_and_lock_ceilings(PhasedLearner* pl);
static inline void phased_save_and_lock_rates(PhasedLearner* pl);
static inline void phased_record_transition(PhasedLearner* pl,
                                             LearningPhase from, LearningPhase to);

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

    pl->n_transitions     = 0;
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

/* ── Save learned values from cloud before locking ──────────────────────── */

static inline void phased_save_and_lock_ceilings(PhasedLearner* pl) {
    float theta_mean[N_PARAMS];
    smc2_cuda_get_theta_mean(pl->smc2, theta_mean);

    pl->config.fixed_mu_scale    = theta_mean[P_MU_SCALE];
    pl->config.fixed_sigma_scale = theta_mean[P_SIGMA_SCALE];
    pl->config.learned_ceilings  = 1;

    printf("[PHASED] Saving ceilings: μ_scale=%.4f  σ_scale=%.4f\n",
           pl->config.fixed_mu_scale, pl->config.fixed_sigma_scale);
}

static inline void phased_save_and_lock_rates(PhasedLearner* pl) {
    float theta_mean[N_PARAMS];
    smc2_cuda_get_theta_mean(pl->smc2, theta_mean);

    pl->config.fixed_mu_rate    = theta_mean[P_MU_RATE];
    pl->config.fixed_sigma_rate = theta_mean[P_SIGMA_RATE];
    pl->config.learned_rates    = 1;

    printf("[PHASED] Saving rates: μ_rate=%.4f  σ_rate=%.4f\n",
           pl->config.fixed_mu_rate, pl->config.fixed_sigma_rate);
}

/* ── Record a phase transition ──────────────────────────────────────────── */

static inline void phased_record_transition(PhasedLearner* pl,
                                             LearningPhase from,
                                             LearningPhase to) {
    ZRangeTracker* zt = &pl->z_tracker;

    if (pl->n_transitions < MAX_TRANSITIONS) {
        PhaseTransition* t = &pl->history[pl->n_transitions++];
        t->window  = zt->n_windows;
        t->from    = from;
        t->to      = to;
        t->z_min   = zt->z_min_window;
        t->z_max   = zt->z_max_window;
        t->z_range = zt->z_max_window - zt->z_min_window;
    }

    const char* dir = (to > from) ? "▲ UNLOCK" : "▼ LOCK  ";
    printf("[PHASED] %s Phase %d → %d at window %d  "
           "(z_win: [%.2f, %.2f]  range: %.2f)\n",
           dir, from, to, zt->n_windows,
           zt->z_min_window, zt->z_max_window,
           zt->z_max_window - zt->z_min_window);
}

/* ── Feed z observations from a completed window ────────────────────────── */

static inline void phased_observe_z(
    PhasedLearner* pl,
    float z_mean,       /**< Weighted mean z from this window        */
    float z_min,        /**< Min z observed in this window           */
    float z_max         /**< Max z observed in this window           */
) {
    ZRangeTracker* zt = &pl->z_tracker;

    /* Store current window stats */
    zt->z_min_window  = z_min;
    zt->z_max_window  = z_max;
    zt->z_mean_window = z_mean;

    /* Update global extremes */
    if (z_min < zt->z_min_seen) zt->z_min_seen = z_min;
    if (z_max > zt->z_max_seen) zt->z_max_seen = z_max;

    /* EMA of z */
    if (zt->n_windows == 0) {
        zt->z_ema = z_mean;
    } else {
        zt->z_ema += pl->config.z_ema_alpha * (z_mean - zt->z_ema);
    }

    float z_range = z_max - z_min;

    /* ── Forward streaks (unlock triggers) ────────────────────────────── */

    /* High-z streak: z_max above ceiling threshold */
    if (z_max > pl->config.ceiling_z_threshold) {
        zt->high_z_streak++;
        zt->calm_streak = 0;          /* Reset backward counter */
    } else {
        zt->high_z_streak = 0;
        zt->calm_streak++;             /* Build backward counter */
    }

    /* Wide-range streak: z_range above rate threshold */
    if (z_range > pl->config.rate_z_range_threshold) {
        zt->wide_range_streak++;
        zt->narrow_range_streak = 0;   /* Reset backward counter */
    } else {
        zt->wide_range_streak = 0;
        zt->narrow_range_streak++;     /* Build backward counter */
    }

    zt->n_windows++;
}

/* ── Feed z range from inner particles ──────────────────────────────────── */

/* ── Feed z range from inner particles (ROBUST version) ────────────────────── */

static inline void phased_observe_z_from_smc2(PhasedLearner* pl) {
    float z_mean, z_min, z_max;
    /* Use robust version: per-θ means, then min/max of those.
     * Raw smc2_cuda_get_z_range() takes min/max across ALL N_theta × N_inner
     * inner particles — a single outlier at z>2.0 prevents calm_streak from
     * ever accumulating, so the backward valve (Phase 3→2→1) never fires.
     * The robust version computes z̄ per θ-particle first, then min/max of
     * those 1024 means. One outlier inner particle is diluted by N_inner-1. */
    smc2_cuda_get_z_range_robust(pl->smc2, &z_mean, &z_min, &z_max);
    phased_observe_z(pl, z_mean, z_min, z_max);
}

/* ── Phase transition logic (bidirectional) ─────────────────────────────── */

static inline int phased_update(PhasedLearner* pl) {
    LearningPhase prev_phase = pl->phase;
    ZRangeTracker* zt = &pl->z_tracker;

    switch (pl->phase) {

    case PHASE_1_FLOORS:
        /* ── Forward: Phase 1 → 2 ──────────────────────────────────── */
        /* z_max > threshold for N consecutive windows                   */
        if (zt->high_z_streak >= pl->config.ceiling_z_sustained) {
            pl->phase = PHASE_2_CEILINGS;
            if (pl->phase2_entered_at < 0)
                pl->phase2_entered_at = zt->n_windows;
            phased_record_transition(pl, prev_phase, pl->phase);
            phased_apply_mask(pl);
        }
        /* (No backward from Phase 1 — it's the floor) */
        break;

    case PHASE_2_CEILINGS:
        /* ── Forward: Phase 2 → 3 ──────────────────────────────────── */
        /* z-range within window > threshold for N consecutive windows   */
        if (zt->wide_range_streak >= pl->config.rate_range_sustained) {
            pl->phase = PHASE_3_RATES;
            if (pl->phase3_entered_at < 0)
                pl->phase3_entered_at = zt->n_windows;
            phased_record_transition(pl, prev_phase, pl->phase);
            phased_apply_mask(pl);
        }
        /* ── Backward: Phase 2 → 1 ────────────────────────────────── */
        /* z_max < threshold for N consecutive windows (calm returned)   */
        /* Save learned ceilings before locking                          */
        else if (pl->config.enable_backward &&
                 zt->calm_streak >= pl->config.ceiling_z_sustained) {
            phased_save_and_lock_ceilings(pl);
            pl->phase = PHASE_1_FLOORS;
            phased_record_transition(pl, prev_phase, pl->phase);
            phased_apply_mask(pl);
        }
        break;

    case PHASE_3_RATES:
        /* ── Backward: Phase 3 → 2 ────────────────────────────────── */
        /* z-range narrows for N consecutive windows                     */
        /* Save learned rates before locking                             */
        if (pl->config.enable_backward &&
            zt->narrow_range_streak >= pl->config.rate_range_sustained) {
            phased_save_and_lock_rates(pl);
            pl->phase = PHASE_2_CEILINGS;
            phased_record_transition(pl, prev_phase, pl->phase);
            phased_apply_mask(pl);

            /* Check if we should also retreat to Phase 1 immediately.   */
            /* This can happen if both range AND z_max dropped together. */
            if (zt->calm_streak >= pl->config.ceiling_z_sustained) {
                LearningPhase mid = pl->phase;
                phased_save_and_lock_ceilings(pl);
                pl->phase = PHASE_1_FLOORS;
                phased_record_transition(pl, mid, pl->phase);
                phased_apply_mask(pl);
            }
        }
        break;
    }

    return (pl->phase != prev_phase);
}

/* ── Override fixed values mid-run ──────────────────────────────────────── */

/**
 * @brief Update the fixed values for locked parameters.
 *
 * Call this when offline calibration provides better estimates,
 * or when warm-starting from a previous session.
 */
static inline void phased_set_fixed_rates(
    PhasedLearner* pl,
    float mu_rate,
    float sigma_rate
) {
    pl->config.fixed_mu_rate    = mu_rate;
    pl->config.fixed_sigma_rate = sigma_rate;
    pl->config.learned_rates    = 1;

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
    pl->config.learned_ceilings  = 1;

    /* Re-apply if ceilings are currently fixed */
    if (pl->phase < PHASE_2_CEILINGS) {
        phased_apply_mask(pl);
    }
}

/* ── Query ──────────────────────────────────────────────────────────────── */

/** @brief Get current phase */
static inline LearningPhase phased_get_phase(const PhasedLearner* pl) {
    return pl->phase;
}

/** @brief Check if a param group has ever been learned */
static inline int phased_has_learned_ceilings(const PhasedLearner* pl) {
    return pl->config.learned_ceilings;
}

static inline int phased_has_learned_rates(const PhasedLearner* pl) {
    return pl->config.learned_rates;
}

/* ── Diagnostics ────────────────────────────────────────────────────────── */

static inline void phased_print_status(const PhasedLearner* pl) {
    const char* phase_names[] = {"???", "FLOORS (4p)", "CEILINGS (6p)", "RATES (8p)"};
    const ZRangeTracker* zt = &pl->z_tracker;

    printf("Phased Learning Status (bidirectional):\n");
    printf("  Phase:            %d — %s\n", pl->phase, phase_names[pl->phase]);
    printf("  Windows:          %d\n", zt->n_windows);
    printf("  z_window:         [%.3f, %.3f]  range=%.3f\n",
           zt->z_min_window, zt->z_max_window,
           zt->z_max_window - zt->z_min_window);
    printf("  z_global:         [%.3f, %.3f]\n", zt->z_min_seen, zt->z_max_seen);
    printf("  z_ema:            %.3f\n", zt->z_ema);

    printf("  ── Forward (unlock) ──\n");
    printf("    high_z_streak:    %d / %d\n",
           zt->high_z_streak, pl->config.ceiling_z_sustained);
    printf("    wide_range_streak:%d / %d\n",
           zt->wide_range_streak, pl->config.rate_range_sustained);

    printf("  ── Backward (lock) ──\n");
    printf("    calm_streak:      %d / %d\n",
           zt->calm_streak, pl->config.ceiling_z_sustained);
    printf("    narrow_streak:    %d / %d\n",
           zt->narrow_range_streak, pl->config.rate_range_sustained);

    printf("  ── Saved values ──\n");
    printf("    μ_scale:  %.4f  %s\n", pl->config.fixed_mu_scale,
           pl->config.learned_ceilings ? "(learned)" : "(prior)");
    printf("    σ_scale:  %.4f  %s\n", pl->config.fixed_sigma_scale,
           pl->config.learned_ceilings ? "(learned)" : "(prior)");
    printf("    μ_rate:   %.4f  %s\n", pl->config.fixed_mu_rate,
           pl->config.learned_rates ? "(learned)" : "(prior)");
    printf("    σ_rate:   %.4f  %s\n", pl->config.fixed_sigma_rate,
           pl->config.learned_rates ? "(learned)" : "(prior)");

    if (pl->n_transitions > 0) {
        printf("  ── Transition history (%d) ──\n", pl->n_transitions);
        for (int i = 0; i < pl->n_transitions; i++) {
            const PhaseTransition* t = &pl->history[i];
            const char* dir = (t->to > t->from) ? "▲" : "▼";
            printf("    %s win=%d  P%d→P%d  z=[%.2f,%.2f] range=%.2f\n",
                   dir, t->window, t->from, t->to,
                   t->z_min, t->z_max, t->z_range);
        }
    }

    printf("  ── Active params ──\n    Free:  ");
    uint8_t mask[N_PARAMS];
    memcpy(mask, pl->smc2->fixed_mask, N_PARAMS);
    const char* names[] = {"ρ", "σ_tot", "r", "μ_base", "μ_scl", "μ_rate", "σ_scl", "σ_rate"};
    for (int i = 0; i < N_PARAMS; i++) {
        if (!mask[i]) printf("%s ", names[i]);
    }
    printf("\n    Fixed: ");
    for (int i = 0; i < N_PARAMS; i++) {
        if (mask[i]) printf("%s=%.3f ", names[i], pl->smc2->fixed_values[i]);
    }
    printf("\n");
}

#endif /* SMC2_PHASED_LEARNING_H */