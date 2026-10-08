#include "friction_circle.h"
#include <math.h>

void egga_allocate_friction_circle(
    const egga_friction_demand_t* demand,
    float speed,
    float wheelbase,
    float steer_hw_max,
    egga_friction_allocated_t* out
) {
    if (!demand || !out) {
        return;
    }

    float g = (demand->gravity > 0.0f) ? demand->gravity : 9.80665f;
    float mu = demand->mu;
    if (!isfinite(mu) || mu < 1e-4f) {
        mu = 1e-4f;
    }

    float gamma = demand->safety_factor;
    if (gamma < 0.1f) gamma = 0.1f;
    if (gamma > 1.0f) gamma = 1.0f;

    float a_max_total = gamma * mu * g;

    float ax_raw = isfinite(demand->ax_req) ? demand->ax_req : 0.0f;
    float ay_raw = isfinite(demand->ay_req) ? demand->ay_req : 0.0f;

    float ax_abs = fabsf(ax_raw);
    float ay_abs = fabsf(ay_raw);

    float rho = sqrtf(ax_abs * ax_abs + ay_abs * ay_abs) / (mu * g);

    float ax_safe = 0.0f;
    float ay_safe = 0.0f;
    int32_t is_clamped = 0;
    int32_t clamped_axis = EGGA_FC_CLAMP_NONE;

    if (rho <= gamma) {
        ax_safe = ax_raw;
        ay_safe = ay_raw;
        is_clamped = 0;
        clamped_axis = EGGA_FC_CLAMP_NONE;
    } else {
        is_clamped = 1;
        if (demand->policy == EGGA_FC_STEERING_PRIORITY) {
            float ay_clamped = (ay_abs < a_max_total) ? ay_abs : a_max_total;
            float rem_sq = a_max_total * a_max_total - ay_clamped * ay_clamped;
            if (rem_sq < 0.0f) rem_sq = 0.0f;
            float ax_clamped = sqrtf(rem_sq);
            if (ax_abs < ax_clamped) ax_clamped = ax_abs;

            clamped_axis = (ay_abs <= a_max_total) ? EGGA_FC_CLAMP_LONGITUDINAL : EGGA_FC_CLAMP_BOTH;
            ax_safe = copysignf(ax_clamped, ax_raw);
            ay_safe = copysignf(ay_clamped, ay_raw);
        } else if (demand->policy == EGGA_FC_BRAKING_PRIORITY) {
            float ax_clamped = (ax_abs < a_max_total) ? ax_abs : a_max_total;
            float rem_sq = a_max_total * a_max_total - ax_clamped * ax_clamped;
            if (rem_sq < 0.0f) rem_sq = 0.0f;
            float ay_clamped = sqrtf(rem_sq);
            if (ay_abs < ay_clamped) ay_clamped = ay_abs;

            clamped_axis = (ax_abs <= a_max_total) ? EGGA_FC_CLAMP_LATERAL : EGGA_FC_CLAMP_BOTH;
            ax_safe = copysignf(ax_clamped, ax_raw);
            ay_safe = copysignf(ay_clamped, ay_raw);
        } else { /* EGGA_FC_BALANCED */
            float scale = gamma / rho;
            ax_safe = ax_raw * scale;
            ay_safe = ay_raw * scale;
            clamped_axis = EGGA_FC_CLAMP_BOTH;
        }
    }

    float v = (isfinite(speed) && speed > 0.5f) ? speed : 0.5f;
    float wb = (isfinite(wheelbase) && wheelbase > 0.5f) ? wheelbase : 2.8f;
    float steer_kinematic = atan2f(fabsf(ay_safe) * wb, v * v);
    float delta_max_coupled = (steer_kinematic < steer_hw_max) ? steer_kinematic : steer_hw_max;

    out->ax_safe = ax_safe;
    out->ay_safe = ay_safe;
    out->utilisation = rho;
    out->is_clamped = is_clamped;
    out->clamped_axis = clamped_axis;
    out->delta_max_coupled = delta_max_coupled;
}
