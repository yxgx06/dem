#ifndef EGGA_ACTOR_H
#define EGGA_ACTOR_H

#include "actor_weights.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Predict gain adjustments from 6D observation vector.
 * @param obs Input observation vector (length 6)
 * @param delta_K Output gain adjustments (length 4)
 */
void egga_actor_predict(const float obs[EGGA_OBS_DIM], float delta_K[EGGA_ACT_DIM]);

/**
 * Predict absolute RL gain proposal (reference_gains + delta_K).
 * @param obs Input observation vector (length 6)
 * @param proposal Output proposed gains [Kp, Ki, Kd, Khead]
 */
void egga_actor_proposal(const float obs[EGGA_OBS_DIM], float proposal[EGGA_ACT_DIM]);

#ifdef __cplusplus
}
#endif

#endif /* EGGA_ACTOR_H */
