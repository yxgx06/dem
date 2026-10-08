#ifndef EGGA_JACKKNIFE_GUARD_H
#define EGGA_JACKKNIFE_GUARD_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define EGGA_FLAG_JACKKNIFE_OK 0
#define EGGA_FLAG_JACKKNIFE_MARGIN 1
#define EGGA_FLAG_JACKKNIFE_CRITICAL 2
#define EGGA_FLAG_ROLLOVER_WARNING 4

typedef struct {
    float l2;
    float tau_air;
    float gravity;
    float ltr_warning;
    float ltr_critical;
    float steer_rate_max;
    float min_theta_crit;
    float max_theta_crit;
} egga_jackknife_config_t;

typedef struct {
    float theta_a;
    float theta_a_dot;
    float vx;
    float mu;
    float ltr;
    float steer_cmd_req;
    float steer_rate_req;
} egga_jackknife_inputs_t;

typedef struct {
    float theta_crit;
    float h_jackknife;
    int32_t is_jackknife_critical;
    int32_t is_rollover_critical;
    float steer_rate_safe;
    float steer_cmd_safe;
    float trailer_brake_pressure;
    int32_t status_flags;
} egga_jackknife_outputs_t;

void egga_jackknife_guard_step(
    const egga_jackknife_config_t* cfg,
    const egga_jackknife_inputs_t* inp,
    egga_jackknife_outputs_t* out
);

#ifdef __cplusplus
}
#endif

#endif /* EGGA_JACKKNIFE_GUARD_H */
