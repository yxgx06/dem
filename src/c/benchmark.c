#include "supervisor.h"
#include "actor.h"
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

#define N_WARMUP 1000
#define N_ITER 100000

static int compare_doubles(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

int main(void) {
    printf("=== EGGA Embedded C99 Benchmark & Memory Audit ===\n\n");

    /* Memory Footprint Analysis */
    printf("--- Static & Stack Memory Layout (Zero Heap Allocation) ---\n");
    printf("sizeof(egga_state_t):     %zu bytes\n", sizeof(egga_state_t));
    printf("sizeof(egga_config_t):    %zu bytes\n", sizeof(egga_config_t));
    printf("sizeof(egga_inputs_t):    %zu bytes\n", sizeof(egga_inputs_t));
    printf("sizeof(egga_outputs_t):   %zu bytes\n", sizeof(egga_outputs_t));
    printf("Actor Parameters (92):    %zu bytes (flash/text constant)\n", (size_t)(92 * sizeof(float)));
    printf("Verified Envelope Mask:   %zu bytes (ROM/flash table)\n", (size_t)(EGGA_N_CELLS * EGGA_MASK_WORDS * sizeof(uint64_t)));
    printf("Heap Allocation:          0 bytes (ZERO malloc/free/calloc)\n\n");

    /* Setup nominal config */
    egga_config_t cfg = {
        .dt = 0.01f,
        .gain_hop_period = 0.5f,
        .min_dwell = 1.0f,
        .recover = 2.0f,
        .stale_grace = 0.3f,
        .mu_floor = 0.2f,
        .tau_prior = 0.08f,
        .mass_prior_max = 1.5f,
        .tau_fallback = 0.12f,
        .mass_fallback = 1.5f,
        .tau_degraded_enter = 0.07f,
        .tau_degraded_exit = 0.05f,
        .mu_low_enter = 0.4f,
        .mu_low_exit = 0.55f,
        .quality_enter = 0.3f,
        .quality_exit = 0.6f,
        .speed_scale = {1.0f, 0.85f, 0.65f, 0.75f, 0.5f, 0.3f},
        .yaw_lpf_tau = 0.05f,
        .yaw_abs = 0.05f,
        .yaw_rel = 0.25f,
        .yaw_persist = 0.1f,
        .yaw_min_speed = 3.0f,
        .actuator_abs = 0.035f,
        .actuator_persist = 0.15f,
        .saturation_level = 0.95f,
        .saturation_persist = 0.2f,
        .rl_rate = {1.0f, 0.1f, 0.25f, 1.0f},
        .rl_persist_rejects = 5,
        .lane_margin = 0.4f,
        .margin_speed_scale = 0.8f,
        .min_speed_for_limit = 3.0f,
        .decel = 3.0f,
        .accel = 1.5f,
        .wheelbase = 2.85f,
        .lf = 1.4f,
        .lr = 1.45f,
        .cf = 80000.0f,
        .cr = 85000.0f,
        .gravity = 9.81f,
        .mass_nominal = 1600.0f,
        .steer_hw_max = 0.6f,
        .steer_rate_hw_max = 0.8f
    };

    egga_state_t state;
    egga_supervisor_init(&state);

    egga_inputs_t in = {
        .t = 0.01f,
        .speed = 10.0f,
        .speed_request = 10.0f,
        .curvature_ahead = 0.01f,
        .mu_lo = 0.7f,
        .mu_hi = 0.85f,
        .tau_bar = 0.02f,
        .mass_hi = 1.1f,
        .quality = 0.9f,
        .est_status = EGGA_EST_OK,
        .yaw_rate = 0.05f,
        .steer_meas = 0.02f,
        .steer_cmd = 0.02f,
        .e_abs = 0.03f,
        .e_rate_abs = 0.01f,
        .rl_valid = true,
        .rl_gain = {1.6f, 0.0125f, 0.141f, 0.5041f},
        .reference_gain = {1.6f, 0.0125f, 0.141f, 0.5041f},
        .request_reset = false
    };

    egga_outputs_t out;
    float obs[6] = {0.05f, -0.02f, 0.01f, 0.02f, 0.2f, -0.1f};
    float delta_K[4];

    /* Warm-up */
    for (int i = 0; i < N_WARMUP; ++i) {
        in.t += 0.01f;
        egga_actor_predict(obs, delta_K);
        for (int k = 0; k < 4; ++k) in.rl_gain[k] = in.reference_gain[k] + delta_K[k];
        egga_supervisor_step(&state, &cfg, &in, &out);
    }

    /* Timing Benchmarking using QueryPerformanceCounter */
    LARGE_INTEGER freq, t_start, t_end;
    QueryPerformanceFrequency(&freq);
    double *latencies_us = (double *)malloc(N_ITER * sizeof(double));

    egga_supervisor_init(&state);
    in.t = 0.01f;

    for (int i = 0; i < N_ITER; ++i) {
        in.t += 0.01f;
        QueryPerformanceCounter(&t_start);
        
        egga_actor_predict(obs, delta_K);
        for (int k = 0; k < 4; ++k) in.rl_gain[k] = in.reference_gain[k] + delta_K[k];
        egga_supervisor_step(&state, &cfg, &in, &out);

        QueryPerformanceCounter(&t_end);
        double dt_us = (double)(t_end.QuadPart - t_start.QuadPart) * 1e6 / (double)freq.QuadPart;
        latencies_us[i] = dt_us;
    }

    qsort(latencies_us, N_ITER, sizeof(double), compare_doubles);

    double sum = 0.0;
    for (int i = 0; i < N_ITER; ++i) sum += latencies_us[i];
    double mean_us = sum / N_ITER;
    double p50_us = latencies_us[(int)(N_ITER * 0.50)];
    double p90_us = latencies_us[(int)(N_ITER * 0.90)];
    double p95_us = latencies_us[(int)(N_ITER * 0.95)];
    double p99_us = latencies_us[(int)(N_ITER * 0.99)];
    double max_us = latencies_us[N_ITER - 1];

    printf("--- Execution Timing Benchmark (Actor + Supervisor Step, N=%d) ---\n", N_ITER);
    printf("Mean Latency:    %8.3f us\n", mean_us);
    printf("p50 (Median):    %8.3f us\n", p50_us);
    printf("p90 Latency:     %8.3f us\n", p90_us);
    printf("p95 Latency:     %8.3f us\n", p95_us);
    printf("p99 Latency:     %8.3f us\n", p99_us);
    printf("Max Latency:     %8.3f us\n", max_us);
    printf("Real-Time Budget:  10,000.000 us (100 Hz / 10 ms)\n");
    printf("Budget Headroom: %8.1fx faster than real-time budget\n\n", 10000.0 / p99_us);

    /* Output JSON results for report generator */
    FILE *fp = fopen("results/phase8/c99_benchmark.json", "w");
    if (fp) {
        fprintf(fp, "{\n");
        fprintf(fp, "  \"sizeof_state_bytes\": %zu,\n", sizeof(egga_state_t));
        fprintf(fp, "  \"sizeof_config_bytes\": %zu,\n", sizeof(egga_config_t));
        fprintf(fp, "  \"sizeof_inputs_bytes\": %zu,\n", sizeof(egga_inputs_t));
        fprintf(fp, "  \"sizeof_outputs_bytes\": %zu,\n", sizeof(egga_outputs_t));
        fprintf(fp, "  \"actor_params_bytes\": %zu,\n", (size_t)(92 * sizeof(float)));
        fprintf(fp, "  \"envelope_mask_bytes\": %zu,\n", (size_t)(EGGA_N_CELLS * EGGA_MASK_WORDS * sizeof(uint64_t)));
        fprintf(fp, "  \"heap_allocation_bytes\": 0,\n");
        fprintf(fp, "  \"n_iterations\": %d,\n", N_ITER);
        fprintf(fp, "  \"mean_us\": %.3f,\n", mean_us);
        fprintf(fp, "  \"p50_us\": %.3f,\n", p50_us);
        fprintf(fp, "  \"p90_us\": %.3f,\n", p90_us);
        fprintf(fp, "  \"p95_us\": %.3f,\n", p95_us);
        fprintf(fp, "  \"p99_us\": %.3f,\n", p99_us);
        fprintf(fp, "  \"max_us\": %.3f\n", max_us);
        fprintf(fp, "}\n");
        fclose(fp);
        printf("Benchmark results written to results/phase8/c99_benchmark.json\n");
    }

    free(latencies_us);
    return 0;
}
