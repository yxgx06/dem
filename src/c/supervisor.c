#include "supervisor.h"
#include <math.h>
#include <string.h>

/* Forward internal declarations */
static void log_event(egga_state_t *state, float t, int32_t code, float value);
static void edge(egga_state_t *state, float t, int32_t flags, int32_t bit, int32_t code, float value);

void egga_supervisor_init(egga_state_t *state) {
    memset(state, 0, sizeof(egga_state_t));
    state->mode = EGGA_MODE_NOMINAL;
    state->last_t = NAN;
    state->speed_cmd = NAN;
    state->last_hop_t = -INFINITY;
}

static inline float understeer_coeff(const egga_config_t *cfg, float mass_scale) {
    float mass = cfg->mass_nominal * mass_scale;
    return (mass / cfg->wheelbase) * (cfg->lr / cfg->cr - cfg->lf / cfg->cf);
}

static inline float steer_angle_limit(const egga_config_t *cfg, float speed, float ay_max, float mass_hi) {
    float v = fmaxf(speed, cfg->min_speed_for_limit);
    float angle = (cfg->wheelbase / (v * v) + understeer_coeff(cfg, mass_hi)) * ay_max;
    return fminf(cfg->steer_hw_max, angle);
}

static inline bool lateral_margin_violated(
    const egga_config_t *cfg, float e_abs, float e_rate_abs, float ay_max, float tau_bar
) {
    float deviation = e_rate_abs * tau_bar + 0.5f * ay_max * tau_bar * tau_bar;
    return (e_abs + deviation) > cfg->lane_margin;
}

static void log_event(egga_state_t *state, float t, int32_t code, float value) {
    int32_t i = state->log_n % EGGA_LOG_CAPACITY;
    state->log_t[i] = t;
    state->log_code[i] = code;
    state->log_val[i] = value;
    state->log_n++;
}

static void edge(egga_state_t *state, float t, int32_t flags, int32_t bit, int32_t code, float value) {
    if ((flags & bit) && !(state->prev_flags & bit)) {
        log_event(state, t, code, value);
    }
}

static inline float accumulate(float accumulated, bool bad, float dt) {
    return bad ? (accumulated + dt) : 0.0f;
}

static inline float accumulate_leaky(float accumulated, bool bad, float dt, float decay, float cap) {
    if (bad) {
        return fminf(cap, accumulated + dt);
    }
    return fmaxf(0.0f, accumulated - decay * dt);
}

static void push_command(egga_state_t *state, float command) {
    state->hist[state->hist_n % EGGA_HISTORY] = command;
    state->hist_n++;
}

static void command_range(const egga_state_t *state, int32_t window, float *low, float *high) {
    int32_t count = window;
    if (state->hist_n < count) count = state->hist_n;
    if (count > EGGA_HISTORY) count = EGGA_HISTORY;
    if (count <= 0) {
        *low = 0.0f;
        *high = 0.0f;
        return;
    }
    float mn = state->hist[(state->hist_n - 1) % EGGA_HISTORY];
    float mx = mn;
    for (int32_t i = 1; i < count; ++i) {
        float val = state->hist[(state->hist_n - 1 - i + EGGA_HISTORY * 2) % EGGA_HISTORY];
        if (val < mn) mn = val;
        if (val > mx) mx = val;
    }
    *low = mn;
    *high = mx;
}

static bool yaw_residual_bad(
    egga_state_t *state, const egga_config_t *cfg, float speed, float yaw_rate,
    float steer_meas, float mass_hi, float dt
) {
    float denom = cfg->wheelbase + understeer_coeff(cfg, mass_hi) * speed * speed;
    float predicted = (speed * steer_meas) / denom;
    float alpha = dt / (cfg->yaw_lpf_tau + dt);
    state->yaw_pred += alpha * (predicted - state->yaw_pred);
    float residual = fabsf(yaw_rate - state->yaw_pred);
    return (speed >= cfg->yaw_min_speed) && (residual > (cfg->yaw_abs + cfg->yaw_rel * fabsf(state->yaw_pred)));
}

static bool actuator_residual_bad(
    const egga_state_t *state, const egga_config_t *cfg, float steer_meas, float tau_bar, float dt
) {
    int32_t window = (int32_t)roundf(tau_bar / dt) + 1;
    float low = 0.0f, high = 0.0f;
    command_range(state, window, &low, &high);
    float r1 = low - steer_meas;
    float r2 = steer_meas - high;
    float residual = fmaxf(0.0f, fmaxf(r1, r2));
    return residual > cfg->actuator_abs;
}

static inline bool saturation_bad(const egga_config_t *cfg, float steer_cmd, float steer_limit) {
    return fabsf(steer_cmd) >= (cfg->saturation_level * steer_limit);
}

static int32_t rl_check(
    egga_state_t *state, const egga_config_t *cfg, const float gain[4], float dt
) {
    for (int i = 0; i < 4; ++i) {
        if (!isfinite(gain[i])) {
            state->last_rl_valid = false;
            return EGGA_REASON_RL_REJECTED_RANGE;
        }
    }
    if (gain[0] < egga_kp_axis[0] || gain[0] > egga_kp_axis[EGGA_N_KP - 1] ||
        gain[1] < egga_ki_axis[0] || gain[1] > egga_ki_axis[EGGA_N_KI - 1] ||
        gain[2] < egga_kd_axis[0] || gain[2] > egga_kd_axis[EGGA_N_KD - 1] ||
        gain[3] < egga_khead_axis[0] || gain[3] > egga_khead_axis[EGGA_N_KHEAD - 1]) {
        state->last_rl_valid = false;
        return EGGA_REASON_RL_REJECTED_RANGE;
    }
    if (state->last_rl_valid) {
        for (int i = 0; i < 4; ++i) {
            if (fabsf(gain[i] - state->last_rl[i]) / dt > cfg->rl_rate[i]) {
                return EGGA_REASON_RL_REJECTED_RATE;
            }
        }
    }
    for (int i = 0; i < 4; ++i) {
        state->last_rl[i] = gain[i];
    }
    state->last_rl_valid = true;
    return EGGA_REASON_NONE;
}

static int32_t desired_mode(const egga_config_t *cfg, int32_t current, float tau, float mu_hi, float quality) {
    float tau_th = (current == EGGA_MODE_DEGRADED_ACTUATOR) ? cfg->tau_degraded_exit : cfg->tau_degraded_enter;
    if (tau > tau_th) return EGGA_MODE_DEGRADED_ACTUATOR;

    float mu_th = (current == EGGA_MODE_LOW_MU) ? cfg->mu_low_exit : cfg->mu_low_enter;
    if (mu_hi < mu_th) return EGGA_MODE_LOW_MU;

    float q_th = (current == EGGA_MODE_CAUTIOUS) ? cfg->quality_exit : cfg->quality_enter;
    if (quality < q_th) return EGGA_MODE_CAUTIOUS;

    return EGGA_MODE_NOMINAL;
}

static void next_mode(
    const egga_state_t *state, const egga_config_t *cfg, float t, bool no_set, bool hard_fault,
    float tau, float mu_hi, float quality, bool reset, int32_t *out_mode, int32_t *out_reason
) {
    int32_t current = state->mode;
    if (current == EGGA_MODE_MINIMAL_RISK) {
        if (reset && !no_set && !hard_fault) {
            *out_mode = EGGA_MODE_FALLBACK;
            *out_reason = EGGA_REASON_RESET;
            return;
        }
        *out_mode = current;
        *out_reason = EGGA_REASON_NONE;
        return;
    }
    if (no_set) {
        *out_mode = EGGA_MODE_MINIMAL_RISK;
        *out_reason = EGGA_REASON_NO_VERIFIED_SET;
        return;
    }
    if (hard_fault) {
        *out_mode = EGGA_MODE_FALLBACK;
        *out_reason = (current != EGGA_MODE_FALLBACK) ? EGGA_REASON_MODE_CHANGE : EGGA_REASON_NONE;
        return;
    }
    if (current == EGGA_MODE_FALLBACK) {
        if (state->healthy_for >= cfg->recover - 1e-4f) {
            *out_mode = EGGA_MODE_CAUTIOUS;
            *out_reason = EGGA_REASON_RECOVERY;
            return;
        }
        *out_mode = current;
        *out_reason = EGGA_REASON_NONE;
        return;
    }
    int32_t des = desired_mode(cfg, current, tau, mu_hi, quality);
    if (des != current && (t - state->mode_entry_t >= cfg->min_dwell - 1e-4f)) {
        *out_mode = des;
        *out_reason = EGGA_REASON_MODE_CHANGE;
        return;
    }
    *out_mode = current;
    *out_reason = EGGA_REASON_NONE;
}

/* ------------------------------------------------------------------ Envelope Geometry */

static inline float ay_max_calc(float mu_eff) {
    return EGGA_AY_FRACTION_K * fmaxf(mu_eff, 0.0f) * EGGA_GRAVITY;
}

static float speed_cap_calc(float kappa_ahead, float mu_eff) {
    if (!isfinite(kappa_ahead) || !isfinite(mu_eff)) return -1.0f;
    float kappa = fabsf(kappa_ahead);
    float v_max = egga_speed_axis[EGGA_N_SPEED - 1];
    float cap = (kappa < 1e-9f) ? v_max : fminf(v_max, sqrtf(ay_max_calc(mu_eff) / kappa));
    return (cap >= EGGA_MIN_FEASIBLE_SPEED) ? cap : -1.0f;
}

static bool bracket_axis(
    const float *axis, int32_t n, float val, bool clamp_lo, bool clamp_hi,
    int32_t *b0, int32_t *b1
) {
    if (!isfinite(val)) return false;
    if (val < axis[0]) {
        if (!clamp_lo) return false;
        *b0 = 0; *b1 = 0; return true;
    }
    if (val > axis[n - 1]) {
        if (!clamp_hi) return false;
        *b0 = n - 1; *b1 = n - 1; return true;
    }
    for (int32_t i = 0; i < n; ++i) {
        if (axis[i] >= val) {
            if (axis[i] == val) {
                *b0 = i; *b1 = i;
            } else {
                *b0 = i - 1; *b1 = i;
            }
            return true;
        }
    }
    *b0 = n - 1; *b1 = n - 1;
    return true;
}

static bool get_cell_mask(
    float speed, float mu_eff, float tau_q, float mass_q,
    uint64_t mask[EGGA_MASK_WORDS]
) {
    int32_t bs0, bs1, bm0, bm1, bt0, bt1, bmass0, bmass1;
    if (!bracket_axis(egga_speed_axis, EGGA_N_SPEED, speed, false, false, &bs0, &bs1)) return false;
    if (!bracket_axis(egga_mu_axis, EGGA_N_MU, mu_eff, false, true, &bm0, &bm1)) return false;
    if (!bracket_axis(egga_tau_axis, EGGA_N_TAU, tau_q, true, false, &bt0, &bt1)) return false;
    if (!bracket_axis(egga_mass_axis, EGGA_N_MASS, mass_q, true, false, &bmass0, &bmass1)) return false;

    for (int w = 0; w < EGGA_MASK_WORDS; ++w) mask[w] = ~0ULL;
    mask[EGGA_MASK_WORDS - 1] &= (1ULL << 24) - 1ULL; // 1176 % 64 = 24 bits

    int32_t s_pts[2] = {bs0, bs1};
    int32_t m_pts[2] = {bm0, bm1};
    int32_t t_pts[2] = {bt0, bt1};
    int32_t mass_pts[2] = {bmass0, bmass1};

    int32_t ns = (bs0 == bs1) ? 1 : 2;
    int32_t nm = (bm0 == bm1) ? 1 : 2;
    int32_t nt = (bt0 == bt1) ? 1 : 2;
    int32_t nmass = (bmass0 == bmass1) ? 1 : 2;

    for (int is = 0; is < ns; ++is) {
        int32_t s = s_pts[is];
        for (int im = 0; im < nm; ++im) {
            int32_t m = m_pts[im];
            for (int it = 0; it < nt; ++it) {
                int32_t t = t_pts[it];
                for (int imass = 0; imass < nmass; ++imass) {
                    int32_t mass = mass_pts[imass];
                    int32_t cell = ((s * EGGA_N_MU + m) * EGGA_N_TAU + t) * EGGA_N_MASS + mass;
                    for (int w = 0; w < EGGA_MASK_WORDS; ++w) {
                        mask[w] &= egga_envelope_masks[cell][w];
                    }
                }
            }
        }
    }

    for (int w = 0; w < EGGA_MASK_WORDS; ++w) {
        if (mask[w] != 0ULL) return true;
    }
    return false;
}

static float max_verified_speed_calc(float mu_eff, float tau_q, float mass_q) {
    uint64_t tmp_mask[EGGA_MASK_WORDS];
    for (int s = EGGA_N_SPEED - 1; s >= 0; --s) {
        if (get_cell_mask(egga_speed_axis[s], mu_eff, tau_q, mass_q, tmp_mask)) {
            return egga_speed_axis[s];
        }
    }
    return -1.0f;
}

static float rate_limit_calc(float speed, float mass) {
    int32_t bs0, bs1, bm0, bm1;
    if (!bracket_axis(egga_speed_axis, EGGA_N_SPEED, speed, false, false, &bs0, &bs1)) return -1.0f;
    if (!bracket_axis(egga_mass_axis, EGGA_N_MASS, mass, true, false, &bm0, &bm1)) return -1.0f;

    float r00 = egga_rate_limit[bs0][bm0];
    float r01 = egga_rate_limit[bs0][bm1];
    float r10 = egga_rate_limit[bs1][bm0];
    float r11 = egga_rate_limit[bs1][bm1];
    return fminf(fminf(r00, r01), fminf(r10, r11));
}

static inline float gain_dist_sq(int32_t i0, int32_t i1, int32_t i2, int32_t i3, const float target[4]) {
    float s0 = (egga_kp_axis[EGGA_N_KP - 1] - egga_kp_axis[0]);
    float s1 = (egga_ki_axis[EGGA_N_KI - 1] - egga_ki_axis[0]);
    float s2 = (egga_kd_axis[EGGA_N_KD - 1] - egga_kd_axis[0]);
    float s3 = (egga_khead_axis[EGGA_N_KHEAD - 1] - egga_khead_axis[0]);

    float d0 = (egga_kp_axis[i0] - target[0]) / (s0 > 0.0f ? s0 : 1.0f);
    float d1 = (egga_ki_axis[i1] - target[1]) / (s1 > 0.0f ? s1 : 1.0f);
    float d2 = (egga_kd_axis[i2] - target[2]) / (s2 > 0.0f ? s2 : 1.0f);
    float d3 = (egga_khead_axis[i3] - target[3]) / (s3 > 0.0f ? s3 : 1.0f);

    return d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
}

static bool project_index(
    const uint64_t mask[EGGA_MASK_WORDS], const float target[4], int32_t out_idx[4]
) {
    float best_dist = INFINITY;
    bool found = false;

    for (int32_t i0 = 0; i0 < EGGA_N_KP; ++i0) {
        for (int32_t i1 = 0; i1 < EGGA_N_KI; ++i1) {
            for (int32_t i2 = 0; i2 < EGGA_N_KD; ++i2) {
                for (int32_t i3 = 0; i3 < EGGA_N_KHEAD; ++i3) {
                    int32_t g_idx = ((i0 * EGGA_N_KI + i1) * EGGA_N_KD + i2) * EGGA_N_KHEAD + i3;
                    int32_t word = g_idx / 64;
                    int32_t bit = g_idx % 64;
                    if ((mask[word] >> bit) & 1ULL) {
                        float dist = gain_dist_sq(i0, i1, i2, i3, target);
                        if (dist < best_dist) {
                            best_dist = dist;
                            out_idx[0] = i0; out_idx[1] = i1;
                            out_idx[2] = i2; out_idx[3] = i3;
                            found = true;
                        }
                    }
                }
            }
        }
    }
    return found;
}

static inline bool mask_contains_idx(const uint64_t mask[EGGA_MASK_WORDS], const int32_t idx[4]) {
    if (idx[0] < 0 || idx[0] >= EGGA_N_KP ||
        idx[1] < 0 || idx[1] >= EGGA_N_KI ||
        idx[2] < 0 || idx[2] >= EGGA_N_KD ||
        idx[3] < 0 || idx[3] >= EGGA_N_KHEAD) return false;
    int32_t g_idx = ((idx[0] * EGGA_N_KI + idx[1]) * EGGA_N_KD + idx[2]) * EGGA_N_KHEAD + idx[3];
    return ((mask[g_idx / 64] >> (g_idx % 64)) & 1ULL) != 0ULL;
}

static void next_gain_index(
    const uint64_t mask[EGGA_MASK_WORDS], bool have_current, const int32_t current[4],
    const int32_t target[4], bool may_hop, int32_t out_idx[4], bool *out_forced
) {
    if (!have_current || !mask_contains_idx(mask, current)) {
        out_idx[0] = target[0]; out_idx[1] = target[1];
        out_idx[2] = target[2]; out_idx[3] = target[3];
        *out_forced = true;
        return;
    }
    if (!may_hop) {
        out_idx[0] = current[0]; out_idx[1] = current[1];
        out_idx[2] = current[2]; out_idx[3] = current[3];
        *out_forced = false;
        return;
    }

    int32_t best[4] = {current[0], current[1], current[2], current[3]};
    int32_t best_dist = (current[0] - target[0]) * (current[0] - target[0]) +
                        (current[1] - target[1]) * (current[1] - target[1]) +
                        (current[2] - target[2]) * (current[2] - target[2]) +
                        (current[3] - target[3]) * (current[3] - target[3]);

    for (int d0 = -1; d0 <= 1; ++d0) {
        for (int d1 = -1; d1 <= 1; ++d1) {
            for (int d2 = -1; d2 <= 1; ++d2) {
                for (int d3 = -1; d3 <= 1; ++d3) {
                    int32_t cand[4] = {current[0] + d0, current[1] + d1, current[2] + d2, current[3] + d3};
                    if (mask_contains_idx(mask, cand)) {
                        int32_t dist = (cand[0] - target[0]) * (cand[0] - target[0]) +
                                       (cand[1] - target[1]) * (cand[1] - target[1]) +
                                       (cand[2] - target[2]) * (cand[2] - target[2]) +
                                       (cand[3] - target[3]) * (cand[3] - target[3]);
                        if (dist < best_dist) {
                            best_dist = dist;
                            best[0] = cand[0]; best[1] = cand[1];
                            best[2] = cand[2]; best[3] = cand[3];
                        }
                    }
                }
            }
        }
    }
    out_idx[0] = best[0]; out_idx[1] = best[1];
    out_idx[2] = best[2]; out_idx[3] = best[3];
    *out_forced = false;
}

static bool mask_contains_gain(const uint64_t mask[EGGA_MASK_WORDS], const float gain[4]) {
    int32_t idx[4] = {-1, -1, -1, -1};
    for (int i = 0; i < EGGA_N_KP; ++i) { if (fabsf(egga_kp_axis[i] - gain[0]) < 1e-5f) { idx[0] = i; break; } }
    for (int i = 0; i < EGGA_N_KI; ++i) { if (fabsf(egga_ki_axis[i] - gain[1]) < 1e-5f) { idx[1] = i; break; } }
    for (int i = 0; i < EGGA_N_KD; ++i) { if (fabsf(egga_kd_axis[i] - gain[2]) < 1e-5f) { idx[2] = i; break; } }
    for (int i = 0; i < EGGA_N_KHEAD; ++i) { if (fabsf(egga_khead_axis[i] - gain[3]) < 1e-5f) { idx[3] = i; break; } }

    if (idx[0] < 0 || idx[1] < 0 || idx[2] < 0 || idx[3] < 0) return false;
    return mask_contains_idx(mask, idx);
}

static bool finite_inputs(const egga_inputs_t *inp) {
    if (!isfinite(inp->speed) || inp->speed <= 0.0f) return false;
    if (!isfinite(inp->speed_request)) return false;
    if (!isfinite(inp->curvature_ahead)) return false;
    if (!isfinite(inp->mu_lo) || !isfinite(inp->mu_hi)) return false;
    if (!isfinite(inp->tau_bar) || !isfinite(inp->mass_hi)) return false;
    if (!isfinite(inp->quality)) return false;
    if (!isfinite(inp->yaw_rate) || !isfinite(inp->steer_meas) || !isfinite(inp->steer_cmd)) return false;
    if (!isfinite(inp->e_abs) || !isfinite(inp->e_rate_abs)) return false;
    return true;
}

/* ------------------------------------------------------------------ Core Supervisor Step */

void egga_supervisor_step(
    egga_state_t *state,
    const egga_config_t *cfg,
    const egga_inputs_t *in,
    egga_outputs_t *out
) {
    bool time_ok = isfinite(in->t) && (isnan(state->last_t) || in->t > state->last_t);
    bool invalid = !(time_ok && finite_inputs(in));

    float dt = cfg->dt;
    float t = 0.0f;
    if (time_ok) {
        dt = isnan(state->last_t) ? cfg->dt : (in->t - state->last_t);
        t = in->t;
        state->last_t = t;
    } else {
        dt = cfg->dt;
        t = isnan(state->last_t) ? 0.0f : state->last_t;
    }

    float safe = EGGA_MIN_FEASIBLE_SPEED;
    float speed, request, curvature;
    float mu_lo, mu_hi, tau, mass_hi, quality;
    int32_t est_status;
    float yaw_rate, steer_meas, steer_cmd, e_abs, e_rate;

    if (invalid) {
        speed = safe; request = safe; curvature = 0.0f;
        mu_lo = cfg->mu_floor; mu_hi = 1.0f;
        tau = cfg->tau_prior; mass_hi = cfg->mass_prior_max; quality = 0.0f;
        est_status = EGGA_EST_INVALID;
        yaw_rate = steer_meas = steer_cmd = e_abs = e_rate = 0.0f;
    } else {
        speed = in->speed; request = in->speed_request; curvature = in->curvature_ahead;
        mu_lo = in->mu_lo; mu_hi = in->mu_hi;
        tau = in->tau_bar; mass_hi = in->mass_hi; quality = in->quality;
        est_status = in->est_status;
        yaw_rate = in->yaw_rate; steer_meas = in->steer_meas; steer_cmd = in->steer_cmd;
        e_abs = in->e_abs; e_rate = in->e_rate_abs;
    }

    float mu_top = egga_mu_axis[EGGA_N_MU - 1];
    float mu_eff = fminf(fmaxf(mu_lo, cfg->mu_floor), mu_top);
    bool conservative = (state->mode == EGGA_MODE_FALLBACK || state->mode == EGGA_MODE_MINIMAL_RISK);
    float tau_q = conservative ? fmaxf(tau, cfg->tau_fallback) : tau;
    float mass_q = conservative ? fmaxf(mass_hi, cfg->mass_fallback) : mass_hi;
    mass_q = fmaxf(mass_q, 1.0f);

    float ay_max = ay_max_calc(mu_eff);
    float steer_limit = steer_angle_limit(cfg, speed, ay_max, mass_q);

    if (!invalid) {
        push_command(state, steer_cmd);
        bool yaw_now = yaw_residual_bad(state, cfg, speed, yaw_rate, steer_meas, mass_q, dt);
        state->yaw_bad = accumulate(state->yaw_bad, yaw_now, dt);
        bool act_now = actuator_residual_bad(state, cfg, steer_meas, tau_q, dt);
        state->act_bad = accumulate(state->act_bad, act_now, dt);
        bool sat_now = saturation_bad(cfg, steer_cmd, steer_limit);
        state->sat_bad = accumulate(state->sat_bad, sat_now, dt);
        bool stale_now = (est_status >= EGGA_EST_STALE);
        state->est_bad = accumulate_leaky(state->est_bad, stale_now, dt, 3.0f, 0.6f);
    }

    bool yaw_flag = (state->yaw_bad >= cfg->yaw_persist);
    bool act_flag = (state->act_bad >= cfg->actuator_persist);
    bool sat_flag = (state->sat_bad >= cfg->saturation_persist);
    bool est_flag = (state->est_bad >= cfg->stale_grace);
    bool hard_fault = invalid || yaw_flag || act_flag || sat_flag || est_flag;

    float cap = speed_cap_calc(curvature, mu_eff);
    float v_ver = max_verified_speed_calc(mu_eff, tau_q, mass_q);
    bool domain_violation = (!invalid) && (mu_hi < cfg->mu_floor);
    bool no_set = domain_violation || (cap < 0.0f) || (v_ver < 0.0f);

    state->healthy_for = (hard_fault || no_set) ? 0.0f : (state->healthy_for + dt);

    int32_t previous_mode = state->mode;
    bool reset_ok = in->request_reset && !invalid && (state->healthy_for >= cfg->stale_grace - 1e-4f);
    int32_t mode = 0, mode_reason = 0;
    next_mode(state, cfg, t, no_set, hard_fault, tau, mu_hi, quality, reset_ok, &mode, &mode_reason);

    if (mode != previous_mode) {
        state->mode = mode;
        state->mode_entry_t = t;
        state->mode_changes++;
        log_event(state, t, mode_reason, (float)mode);
    }

    bool rl_ok = false;
    if (in->rl_valid && !invalid) {
        int32_t rl_reason = rl_check(state, cfg, in->rl_gain, dt);
        if (rl_reason != EGGA_REASON_NONE) {
            state->rl_rejects++;
            if (state->rl_rejects == 1 || state->rl_rejects == cfg->rl_persist_rejects) {
                log_event(state, t, rl_reason, (float)state->rl_rejects);
            }
        } else {
            state->rl_rejects = 0;
        }
        rl_ok = (rl_reason == EGGA_REASON_NONE) && !conservative;
    }

    bool margin_flag = lateral_margin_violated(cfg, e_abs, e_rate, ay_max, tau);
    float scale = cfg->speed_scale[state->mode] * (margin_flag ? cfg->margin_speed_scale : 1.0f);
    float target;
    if (cap < 0.0f || v_ver < 0.0f || state->mode == EGGA_MODE_MINIMAL_RISK) {
        target = safe;
    } else {
        target = fmaxf(safe, fminf(request * scale, fminf(cap, v_ver)));
    }

    if (isnan(state->speed_cmd)) {
        state->speed_cmd = speed;
    }
    state->speed_cmd += fmaxf(-cfg->decel * dt, fminf(cfg->accel * dt, target - state->speed_cmd));
    state->speed_cmd = fmaxf(safe, state->speed_cmd);

    uint64_t mask[EGGA_MASK_WORDS];
    bool have_mask = get_cell_mask(fmaxf(state->speed_cmd, speed), mu_eff, tau_q, mass_q, mask);
    if (!have_mask && v_ver >= 0.0f) {
        have_mask = get_cell_mask(v_ver, mu_eff, tau_q, mass_q, mask);
    }

    bool forced = false;
    if (have_mask) {
        const float *proposal = rl_ok ? in->rl_gain : in->reference_gain;
        int32_t target_idx[4] = {0, 0, 0, 0};
        project_index(mask, proposal, target_idx);

        bool may_hop = (t - state->last_hop_t >= cfg->gain_hop_period - 1e-4f);
        int32_t new_idx[4] = {0, 0, 0, 0};
        next_gain_index(mask, state->have_gain, state->gain_idx, target_idx, may_hop, new_idx, &forced);

        if (!state->have_gain || new_idx[0] != state->gain_idx[0] || new_idx[1] != state->gain_idx[1] ||
            new_idx[2] != state->gain_idx[2] || new_idx[3] != state->gain_idx[3]) {
            state->last_hop_t = t;
        }
        for (int i = 0; i < 4; ++i) state->gain_idx[i] = new_idx[i];
        if (forced && state->have_gain) {
            log_event(state, t, EGGA_REASON_GAIN_FORCED, 1.0f);
        }
        state->have_gain = true;
    }

    float applied[4];
    if (state->have_gain) {
        applied[0] = egga_kp_axis[state->gain_idx[0]];
        applied[1] = egga_ki_axis[state->gain_idx[1]];
        applied[2] = egga_kd_axis[state->gain_idx[2]];
        applied[3] = egga_khead_axis[state->gain_idx[3]];
    } else {
        for (int i = 0; i < 4; ++i) applied[i] = in->reference_gain[i];
    }

    uint64_t actual_mask[EGGA_MASK_WORDS];
    bool have_actual_mask = get_cell_mask(speed, mu_eff, tau_q, mass_q, actual_mask);
    bool verified = have_actual_mask && state->have_gain && mask_contains_gain(actual_mask, applied);

    float rate_lim = rate_limit_calc(fmaxf(speed, state->speed_cmd), mass_q);
    if (rate_lim < 0.0f) rate_lim = cfg->steer_rate_hw_max;

    int32_t flags = 0;
    if (yaw_flag) flags |= EGGA_FLAG_YAW;
    if (act_flag) flags |= EGGA_FLAG_ACTUATOR;
    if (sat_flag) flags |= EGGA_FLAG_SATURATION;
    if (est_flag) flags |= EGGA_FLAG_ESTIMATOR;
    if (margin_flag) flags |= EGGA_FLAG_MARGIN;
    if (invalid) flags |= EGGA_FLAG_INVALID;
    if (no_set) flags |= EGGA_FLAG_NO_SET;
    if (domain_violation) flags |= EGGA_FLAG_DOMAIN;
    if (target < request - 1e-5f) flags |= EGGA_FLAG_CAPPED;

    edge(state, t, flags, EGGA_FLAG_YAW, EGGA_REASON_YAW_RESIDUAL, state->yaw_bad);
    edge(state, t, flags, EGGA_FLAG_ACTUATOR, EGGA_REASON_ACTUATOR_RESIDUAL, state->act_bad);
    edge(state, t, flags, EGGA_FLAG_SATURATION, EGGA_REASON_SATURATION, state->sat_bad);
    edge(state, t, flags, EGGA_FLAG_ESTIMATOR, EGGA_REASON_ESTIMATOR_FAULT, state->est_bad);
    edge(state, t, flags, EGGA_FLAG_MARGIN, EGGA_REASON_LATERAL_MARGIN, e_abs);
    edge(state, t, flags, EGGA_FLAG_INVALID, EGGA_REASON_INVALID_INPUT, 0.0f);
    edge(state, t, flags, EGGA_FLAG_DOMAIN, EGGA_REASON_DOMAIN_VIOLATION, mu_hi);
    edge(state, t, flags, EGGA_FLAG_CAPPED, EGGA_REASON_SPEED_CAPPED, target);

    state->prev_flags = flags;

    for (int i = 0; i < 4; ++i) out->gains[i] = applied[i];
    out->speed_cmd = state->speed_cmd;
    out->steer_limit = steer_limit;
    out->rate_limit = rate_lim;
    out->mode = state->mode;
    out->rl_applied = rl_ok && verified;
    out->forced_gain_jump = forced;
    out->verified = verified;
    out->flags = flags;
}
