#include "jackknife_guard.h"
#include <math.h>

void egga_jackknife_guard_step(
    const egga_jackknife_config_t* cfg,
    const egga_jackknife_inputs_t* inp,
    egga_jackknife_outputs_t* out
) {
    if (!cfg || !inp || !out) {
        return;
    }

    float theta_a = isfinite(inp->theta_a) ? inp->theta_a : 0.0f;
    float theta_a_dot = isfinite(inp->theta_a_dot) ? inp->theta_a_dot : 0.0f;
    float ltr = isfinite(inp->ltr) ? inp->ltr : 0.0f;
    if (ltr < 0.0f) ltr = 0.0f;
    if (ltr > 1.0f) ltr = 1.0f;

    float steer_cmd = isfinite(inp->steer_cmd_req) ? inp->steer_cmd_req : 0.0f;
    float steer_rate = isfinite(inp->steer_rate_req) ? inp->steer_rate_req : 0.0f;

    float v_eff = (isfinite(inp->vx) && inp->vx > 1.0f) ? inp->vx : 1.0f;
    float mu_eff = (isfinite(inp->mu) && inp->mu > 0.05f) ? inp->mu : 0.05f;

    float arg = (mu_eff * cfg->gravity * cfg->l2) / (v_eff * v_eff);
    if (arg < -1.0f) arg = -1.0f;
    if (arg > 1.0f) arg = 1.0f;

    float theta_raw = asinf(arg);
    float theta_crit = fabsf(theta_raw);
    if (theta_crit < cfg->min_theta_crit) theta_crit = cfg->min_theta_crit;
    if (theta_crit > cfg->max_theta_crit) theta_crit = cfg->max_theta_crit;

    float predicted_theta = fabsf(theta_a) + cfg->tau_air * fabsf(theta_a_dot);
    float h_jackknife = theta_crit - predicted_theta;

    int32_t flags = EGGA_FLAG_JACKKNIFE_OK;
    float trailer_drag = 0.0f;
    int32_t is_jk_crit = 0;

    if (h_jackknife <= 0.0f) {
        is_jk_crit = 1;
        flags |= EGGA_FLAG_JACKKNIFE_CRITICAL;
        trailer_drag = 1.0f;
    } else if (h_jackknife < 0.08f) {
        flags |= EGGA_FLAG_JACKKNIFE_MARGIN;
        trailer_drag = (0.08f - h_jackknife) / 0.08f;
        if (trailer_drag > 1.0f) trailer_drag = 1.0f;
    }

    int32_t is_ro_crit = 0;
    float rate_scale = 1.0f;

    if (ltr >= cfg->ltr_critical) {
        is_ro_crit = 1;
        flags |= EGGA_FLAG_ROLLOVER_WARNING;
        rate_scale = 0.0f;
    } else if (ltr >= cfg->ltr_warning) {
        flags |= EGGA_FLAG_ROLLOVER_WARNING;
        rate_scale = (cfg->ltr_critical - ltr) / (cfg->ltr_critical - cfg->ltr_warning);
        if (rate_scale < 0.0f) rate_scale = 0.0f;
    }

    if (is_jk_crit) {
        if (rate_scale > 0.2f) rate_scale = 0.2f;
    }

    float steer_rate_safe = steer_rate * rate_scale;
    if (steer_rate_safe < -cfg->steer_rate_max) steer_rate_safe = -cfg->steer_rate_max;
    if (steer_rate_safe > cfg->steer_rate_max) steer_rate_safe = cfg->steer_rate_max;

    float steer_cmd_safe = steer_cmd;
    if (is_ro_crit) {
        steer_cmd_safe = steer_cmd * 0.8f;
    } else if (is_jk_crit) {
        if ((theta_a > 0.0f && steer_cmd > 0.0f) || (theta_a < 0.0f && steer_cmd < 0.0f)) {
            steer_cmd_safe = steer_cmd * 0.5f;
        } else {
            steer_cmd_safe = steer_cmd;
        }
    }

    out->theta_crit = theta_crit;
    out->h_jackknife = h_jackknife;
    out->is_jackknife_critical = is_jk_crit;
    out->is_rollover_critical = is_ro_crit;
    out->steer_rate_safe = steer_rate_safe;
    out->steer_cmd_safe = steer_cmd_safe;
    out->trailer_brake_pressure = trailer_drag;
    out->status_flags = flags;
}
