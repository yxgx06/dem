#include "actor.h"
#include <math.h>

void egga_actor_predict(const float obs[EGGA_OBS_DIM], float delta_K[EGGA_ACT_DIM]) {
    float h[EGGA_ACTOR_HIDDEN_DIM];

    /* Hidden layer: h = tanh(W1 * obs + b1) */
    for (int j = 0; j < EGGA_ACTOR_HIDDEN_DIM; ++j) {
        float sum = EGGA_ACTOR_B1[j];
        for (int i = 0; i < EGGA_OBS_DIM; ++i) {
            sum += EGGA_ACTOR_W1[j][i] * obs[i];
        }
        h[j] = tanhf(sum);
    }

    /* Output layer: y = tanh(W2 * h + b2) */
    for (int k = 0; k < EGGA_ACT_DIM; ++k) {
        float sum = EGGA_ACTOR_B2[k];
        for (int j = 0; j < EGGA_ACTOR_HIDDEN_DIM; ++j) {
            sum += EGGA_ACTOR_W2[k][j] * h[j];
        }
        float act = tanhf(sum);
        if (act > 1.0f) act = 1.0f;
        if (act < -1.0f) act = -1.0f;
        delta_K[k] = act * EGGA_DELTA_K_BOUNDS[k];
    }
}

void egga_actor_proposal(const float obs[EGGA_OBS_DIM], float proposal[EGGA_ACT_DIM]) {
    float delta_K[EGGA_ACT_DIM];
    egga_actor_predict(obs, delta_K);
    for (int k = 0; k < EGGA_ACT_DIM; ++k) {
        proposal[k] = EGGA_REFERENCE_GAINS[k] + delta_K[k];
    }
}
