/*
 * egga_supervisor_sfun.c
 *
 * MATLAB/Simulink Level-2 C MEX S-Function Wrapper for EGGA Verified Runtime Supervisor.
 * Compatible with Simulink Vehicle Dynamics Blockset, Powertrain Blockset, and CarSim.
 *
 * Standards: ANSI C99 / MISRA C:2012 Aligned. Zero Dynamic Heap Allocation.
 */

#define S_FUNCTION_NAME egga_supervisor_sfun
#define S_FUNCTION_LEVEL 2

#include "simstruc.h"
#include <math.h>

#include "egga_types.h"
#include "supervisor.h"
#include "friction_circle.h"

#define NUM_INPUT_PORTS 1
#define NUM_OUTPUT_PORTS 1

/* Input port width: 23 scalar signals */
#define IN_WIDTH 23

/* Output port width: 11 scalar signals */
#define OUT_WIDTH 11

static void mdlInitializeSizes(SimStruct *S) {
    ssSetNumSFcnParams(S, 0); /* No run-time mask parameters required */
    if (ssGetNumSFcnParams(S) != ssGetSFcnParamsCount(S)) {
        return;
    }

    /* Configure continuous / discrete states */
    ssSetNumContStates(S, 0);
    ssSetNumDiscStates(S, 0);

    /* Configure input port */
    if (!ssSetNumInputPorts(S, NUM_INPUT_PORTS)) return;
    ssSetInputPortWidth(S, 0, IN_WIDTH);
    ssSetInputPortDirectFeedThrough(S, 0, 1);
    ssSetInputPortRequiredContiguous(S, 0, 1);

    /* Configure output port */
    if (!ssSetNumOutputPorts(S, NUM_OUTPUT_PORTS)) return;
    ssSetOutputPortWidth(S, 0, OUT_WIDTH);

    /* Configure sample time */
    ssSetNumSampleTimes(S, 1);

    /* Allocate work vectors for static supervisor state (sizeof(egga_state_t) = 3304 B) */
    ssSetNumPWork(S, 2); /* PWork[0] = egga_state_t*, PWork[1] = egga_config_t* */
    ssSetSimStateCompliance(S, USE_DEFAULT_SIM_STATE);
    ssSetOptions(S, SS_OPTION_EXCEPTION_FREE_CODE);
}

static void mdlInitializeSampleTimes(SimStruct *S) {
    /* Inherited or 100 Hz discrete execution (dt = 0.01 s) */
    ssSetSampleTime(S, 0, 0.01);
    ssSetOffsetTime(S, 0, 0.0);
}

#define MDL_START
#if defined(MDL_START)
static void mdlStart(SimStruct *S) {
    /* Fixed static state structures allocated in persistent PWork memory */
    static egga_state_t s_state;
    static egga_config_t s_cfg;

    egga_supervisor_init(&s_state);

    /* Default automotive configuration */
    s_cfg.dt = 0.01f;
    s_cfg.gain_hop_period = 0.20f;
    s_cfg.min_dwell = 0.50f;
    s_cfg.recover = 2.00f;
    s_cfg.stale_grace = 0.30f;
    s_cfg.mu_floor = 0.25f;
    s_cfg.tau_prior = 0.03f;
    s_cfg.mass_prior_max = 1.60f;
    s_cfg.tau_fallback = 0.07f;
    s_cfg.mass_fallback = 1.40f;
    s_cfg.tau_degraded_enter = 0.05f;
    s_cfg.tau_degraded_exit = 0.04f;
    s_cfg.mu_low_enter = 0.40f;
    s_cfg.mu_low_exit = 0.50f;
    s_cfg.quality_enter = 0.30f;
    s_cfg.quality_exit = 0.50f;
    s_cfg.speed_scale[0] = 1.0f;
    s_cfg.speed_scale[1] = 0.85f;
    s_cfg.speed_scale[2] = 0.70f;
    s_cfg.speed_scale[3] = 0.60f;
    s_cfg.speed_scale[4] = 0.50f;
    s_cfg.speed_scale[5] = 0.00f;
    s_cfg.yaw_lpf_tau = 0.05f;
    s_cfg.yaw_abs = 0.15f;
    s_cfg.yaw_rel = 0.30f;
    s_cfg.yaw_persist = 0.10f;
    s_cfg.yaw_min_speed = 3.0f;
    s_cfg.actuator_abs = 0.10f;
    s_cfg.actuator_persist = 0.10f;
    s_cfg.saturation_level = 0.95f;
    s_cfg.saturation_persist = 0.20f;
    s_cfg.rl_rate[0] = 3.0f;
    s_cfg.rl_rate[1] = 0.2f;
    s_cfg.rl_rate[2] = 0.8f;
    s_cfg.rl_rate[3] = 3.0f;
    s_cfg.rl_persist_rejects = 5;
    s_cfg.lane_margin = 0.50f;
    s_cfg.margin_speed_scale = 0.75f;
    s_cfg.min_speed_for_limit = 2.0f;
    s_cfg.decel = 3.0f;
    s_cfg.accel = 2.0f;
    s_cfg.wheelbase = 2.80f;
    s_cfg.lf = 1.20f;
    s_cfg.lr = 1.60f;
    s_cfg.cf = 80000.0f;
    s_cfg.cr = 80000.0f;
    s_cfg.gravity = 9.80665f;
    s_cfg.mass_nominal = 1600.0f;
    s_cfg.steer_hw_max = 0.60f;
    s_cfg.steer_rate_hw_max = 0.80f;

    ssGetPWork(S)[0] = (void*)&s_state;
    ssGetPWork(S)[1] = (void*)&s_cfg;
}
#endif

static void mdlOutputs(SimStruct *S, int_T tid) {
    const real_T *u = (const real_T*) ssGetInputPortSignal(S, 0);
    real_T *y = (real_T*) ssGetOutputPortSignal(S, 0);

    egga_state_t *state = (egga_state_t*) ssGetPWork(S)[0];
    const egga_config_t *cfg = (const egga_config_t*) ssGetPWork(S)[1];

    if (!state || !cfg) return;

    egga_inputs_t in;
    in.t = (float)u[0];
    in.speed = (float)u[1];
    in.speed_request = (float)u[2];
    in.curvature_ahead = (float)u[3];
    in.mu_lo = (float)u[4];
    in.mu_hi = (float)u[5];
    in.tau_bar = (float)u[6];
    in.mass_hi = (float)u[7];
    in.quality = (float)u[8];
    in.est_status = (int32_t)u[9];
    in.yaw_rate = (float)u[10];
    in.steer_meas = (float)u[11];
    in.steer_cmd = (float)u[12];
    in.e_abs = (float)u[13];
    in.e_rate_abs = (float)u[14];
    in.rl_valid = (int32_t)(u[15] > 0.5);
    in.rl_gain[0] = (float)u[16];
    in.rl_gain[1] = (float)u[17];
    in.rl_gain[2] = (float)u[18];
    in.rl_gain[3] = (float)u[19];
    in.reference_gain[0] = (float)u[20];
    in.reference_gain[1] = (float)u[21];
    in.reference_gain[2] = (float)u[22];
    in.reference_gain[3] = (float)u[23];
    in.request_reset = (int32_t)(u[24] > 0.5);

    egga_outputs_t out;
    egga_supervisor_step(state, cfg, &in, &out);

    /* Write to output bus */
    y[0]  = (real_T)out.gains[0];
    y[1]  = (real_T)out.gains[1];
    y[2]  = (real_T)out.gains[2];
    y[3]  = (real_T)out.gains[3];
    y[4]  = (real_T)out.speed_cmd;
    y[5]  = (real_T)out.steer_limit;
    y[6]  = (real_T)out.rate_limit;
    y[7]  = (real_T)out.mode;
    y[8]  = (real_T)out.rl_applied;
    y[9]  = (real_T)out.verified;
    y[10] = (real_T)out.flags;
}

static void mdlTerminate(SimStruct *S) {
    /* Pure stack execution - zero dynamic memory cleanup required */
    (void)S;
}

#ifdef MATLAB_MEX_FILE
#include "simulink.c"
#else
#include "cg_sfun.h"
#endif
