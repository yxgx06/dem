#ifndef EGGA_FRICTION_CIRCLE_H
#define EGGA_FRICTION_CIRCLE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    EGGA_FC_STEERING_PRIORITY = 0,
    EGGA_FC_BALANCED = 1,
    EGGA_FC_BRAKING_PRIORITY = 2
} egga_fc_policy_t;

typedef enum {
    EGGA_FC_CLAMP_NONE = 0,
    EGGA_FC_CLAMP_LONGITUDINAL = 1,
    EGGA_FC_CLAMP_LATERAL = 2,
    EGGA_FC_CLAMP_BOTH = 3
} egga_fc_clamp_axis_t;

typedef struct {
    float ax_req;
    float ay_req;
    float mu;
    float gravity;
    float safety_factor;
    int32_t policy;
} egga_friction_demand_t;

typedef struct {
    float ax_safe;
    float ay_safe;
    float utilisation;
    int32_t is_clamped;
    int32_t clamped_axis;
    float delta_max_coupled;
} egga_friction_allocated_t;

void egga_allocate_friction_circle(
    const egga_friction_demand_t* demand,
    float speed,
    float wheelbase,
    float steer_hw_max,
    egga_friction_allocated_t* out
);

#ifdef __cplusplus
}
#endif

#endif /* EGGA_FRICTION_CIRCLE_H */
