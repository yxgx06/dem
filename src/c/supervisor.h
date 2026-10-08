#ifndef EGGA_SUPERVISOR_H
#define EGGA_SUPERVISOR_H

#include "egga_types.h"
#include "supervisor_envelope_data.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize supervisor state.
 * @param state State struct to initialize
 */
void egga_supervisor_init(egga_state_t *state);

/**
 * Perform one tick of the verified runtime supervisor.
 * @param state Supervisor state struct (mutated in place)
 * @param cfg Flat supervisor and vehicle configuration
 * @param in Input measurements, estimates and proposed gains
 * @param out Output commanded speed, gains, demand limits, mode, and flags
 */
void egga_supervisor_step(
    egga_state_t *state,
    const egga_config_t *cfg,
    const egga_inputs_t *in,
    egga_outputs_t *out
);

#ifdef __cplusplus
}
#endif

#endif /* EGGA_SUPERVISOR_H */
