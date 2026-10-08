#ifndef EGGA_TYPES_H
#define EGGA_TYPES_H

#include <stdint.h>
#include <stdbool.h>

#define EGGA_LOG_CAPACITY 256
#define EGGA_HISTORY 32

typedef enum {
    EGGA_MODE_NOMINAL = 0,
    EGGA_MODE_CAUTIOUS = 1,
    EGGA_MODE_LOW_MU = 2,
    EGGA_MODE_DEGRADED_ACTUATOR = 3,
    EGGA_MODE_FALLBACK = 4,
    EGGA_MODE_MINIMAL_RISK = 5
} egga_mode_t;

typedef enum {
    EGGA_EST_OK = 0,
    EGGA_EST_LOW_EXCITATION = 1,
    EGGA_EST_STALE = 2,
    EGGA_EST_INVALID = 3
} egga_est_status_t;

typedef enum {
    EGGA_REASON_NONE = 0,
    EGGA_REASON_MODE_CHANGE = 1,
    EGGA_REASON_GAIN_FORCED = 2,
    EGGA_REASON_RL_REJECTED_RANGE = 3,
    EGGA_REASON_RL_REJECTED_RATE = 4,
    EGGA_REASON_SPEED_CAPPED = 5,
    EGGA_REASON_NO_VERIFIED_SET = 6,
    EGGA_REASON_ESTIMATOR_FAULT = 7,
    EGGA_REASON_YAW_RESIDUAL = 8,
    EGGA_REASON_ACTUATOR_RESIDUAL = 9,
    EGGA_REASON_SATURATION = 10,
    EGGA_REASON_LATERAL_MARGIN = 11,
    EGGA_REASON_INVALID_INPUT = 12,
    EGGA_REASON_RECOVERY = 13,
    EGGA_REASON_DOMAIN_VIOLATION = 14,
    EGGA_REASON_RESET = 15
} egga_reason_t;

#define EGGA_FLAG_YAW         1
#define EGGA_FLAG_ACTUATOR    2
#define EGGA_FLAG_SATURATION  4
#define EGGA_FLAG_ESTIMATOR   8
#define EGGA_FLAG_MARGIN      16
#define EGGA_FLAG_INVALID     32
#define EGGA_FLAG_NO_SET      64
#define EGGA_FLAG_DOMAIN      128
#define EGGA_FLAG_CAPPED      256

typedef struct {
    float dt;
    float gain_hop_period;
    float min_dwell;
    float recover;
    float stale_grace;
    float mu_floor;
    float tau_prior;
    float mass_prior_max;
    float tau_fallback;
    float mass_fallback;
    float tau_degraded_enter;
    float tau_degraded_exit;
    float mu_low_enter;
    float mu_low_exit;
    float quality_enter;
    float quality_exit;
    float speed_scale[6];
    float yaw_lpf_tau;
    float yaw_abs;
    float yaw_rel;
    float yaw_persist;
    float yaw_min_speed;
    float actuator_abs;
    float actuator_persist;
    float saturation_level;
    float saturation_persist;
    float rl_rate[4];
    int32_t rl_persist_rejects;
    float lane_margin;
    float margin_speed_scale;
    float min_speed_for_limit;
    float decel;
    float accel;
    float wheelbase;
    float lf;
    float lr;
    float cf;
    float cr;
    float gravity;
    float mass_nominal;
    float steer_hw_max;
    float steer_rate_hw_max;
} egga_config_t;

typedef struct {
    float t;
    float speed;
    float speed_request;
    float curvature_ahead;
    float mu_lo;
    float mu_hi;
    float tau_bar;
    float mass_hi;
    float quality;
    int32_t est_status;
    float yaw_rate;
    float steer_meas;
    float steer_cmd;
    float e_abs;
    float e_rate_abs;
    bool rl_valid;
    float rl_gain[4];
    float reference_gain[4];
    bool request_reset;
} egga_inputs_t;

typedef struct {
    float gains[4];
    float speed_cmd;
    float steer_limit;
    float rate_limit;
    int32_t mode;
    bool rl_applied;
    bool forced_gain_jump;
    bool verified;
    int32_t flags;
} egga_outputs_t;

typedef struct {
    int32_t mode;
    float mode_entry_t;
    float last_t;
    float healthy_for;
    float yaw_pred;
    float yaw_bad;
    float act_bad;
    float sat_bad;
    float est_bad;
    int32_t rl_rejects;
    bool last_rl_valid;
    float speed_cmd;
    int32_t gain_idx[4];
    bool have_gain;
    float last_hop_t;
    float last_rl[4];
    float hist[EGGA_HISTORY];
    int32_t hist_n;
    int32_t prev_flags;
    float log_t[EGGA_LOG_CAPACITY];
    int32_t log_code[EGGA_LOG_CAPACITY];
    float log_val[EGGA_LOG_CAPACITY];
    int32_t log_n;
    int32_t mode_changes;
} egga_state_t;

#endif /* EGGA_TYPES_H */
