function studio_matlab_bridge(varargin)
%% STUDIO_MATLAB_BRIDGE - Connects MATLAB Simulation Environment to EGGA Studio Server
%
% Streams live vehicle dynamics and controller telemetry from MATLAB into 
% http://127.0.0.1:8088/api/matlab/stream in real-time.
%
% USAGE:
%   studio_matlab_bridge()                                 % Default Proving Ground RL Adaptive
%   studio_matlab_bridge('mission', 'proving_ground', 'controller', 'rl')
%   studio_matlab_bridge('mission', 'proving_ground', 'controller', 'mpc')
%   studio_matlab_bridge('mission', 'proving_ground', 'controller', 'pid')
%   studio_matlab_bridge('mission', 'bicycle_master', 'controller', 'mpc', 'trajectory', 'SineWave')
%
% Can be executed from MATLAB command window, Simulink, or launched by server.py!

p = inputParser;
addParameter(p, 'mission', 'proving_ground');       % 'proving_ground' or 'bicycle_master'
addParameter(p, 'controller', 'rl');                % 'rl', 'mpc', 'pid', 'stanley', 'lqr'
addParameter(p, 'trajectory', 'SineWave');          % For bicycle_master: 'SineWave', 'DoubleLaneChange', 'Circle'
addParameter(p, 'server_url', 'http://127.0.0.1:8088');
addParameter(p, 'speed_multiplier', 1.0);           % 1.0 = real-time, 2.0 = 2x speed
addParameter(p, 'vx', 20.0);                        % Forward speed (m/s) [72 km/h highway speed default]
addParameter(p, 'max_time', 75.0);                  % Simulation duration (s)
parse(p, varargin{:});

mission_name     = char(p.Results.mission);
ctrl_name        = char(p.Results.controller);
traj_name        = char(p.Results.trajectory);
server_url       = char(p.Results.server_url);
speed_mult       = double(p.Results.speed_multiplier);
vx_target        = double(p.Results.vx);
max_time         = double(p.Results.max_time);

stream_url = [server_url '/api/matlab/stream'];
options = weboptions('MediaType', 'application/json', 'Timeout', 5);

fprintf('=================================================================\n');
fprintf('  EGGA STUDIO <-> MATLAB BACKEND BRIDGE CONNECTED                \n');
fprintf('  Mission:       %s\n', mission_name);
fprintf('  Controller:    %s\n', ctrl_name);
fprintf('  Speed:         %.1f m/s (%.1f km/h)\n', vx_target, vx_target * 3.6);
fprintf('  Pace Mult:     %.2fx\n', speed_mult);
fprintf('  Stream URL:    %s\n', stream_url);
fprintf('=================================================================\n\n');

% Set up legacy path
cur_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(cur_dir);
legacy_dir = fullfile(repo_root, 'legacy');
addpath(cur_dir);
addpath(legacy_dir);

if strcmpi(mission_name, 'proving_ground')
    run_proving_ground_mission(stream_url, ctrl_name, max_time, speed_mult, options, vx_target);
elseif strcmpi(mission_name, 'simulink')
    run_simulink_simulation(stream_url, legacy_dir, speed_mult, options, vx_target);
elseif strcmpi(mission_name, 'rl_env')
    run_rl_env_simulation(stream_url, speed_mult, options, vx_target);
else
    run_bicycle_master_simulation(stream_url, ctrl_name, traj_name, max_time, speed_mult, options, vx_target);
end

fprintf('\n>>> MATLAB Simulation Bridge completed.\n');
end


%% ========================================================================
% SUBFUNCTION: Proving Ground Multi-Stage Mission (from mission_proving_ground_rl.m)
%% ========================================================================
function run_proving_ground_mission(stream_url, ctrl_req, max_time, speed_mult, options, Vx_input)
dt = 0.01;
t = (0 : dt : max_time)';
N = length(t);

% Vehicle Constants (2-DOF Dynamic Bicycle Model matching legacy MATLAB mission)
m       = 1500;       % Mass (kg)
Iz      = 3000;       % Yaw inertia (kg*m^2)
lf      = 1.2;        % CG to front axle (m)
lr      = 1.6;        % CG to rear axle (m)
L       = lf + lr;    % Wheelbase (2.8 m)
h_cg    = 0.5;        % CG height (m)
Cf      = 80000;      % Front cornering stiffness (N/rad)
Cr      = 80000;      % Rear cornering stiffness (N/rad)
Vx      = max(double(Vx_input), 5.0); % Forward speed (m/s) [dynamic highway test speed]
g       = 9.81;       % Gravity
delta_max = 0.5;      % Max steering limit (rad)

% Environmental Profiles (Slopes & Friction)
slope_deg = zeros(N, 1);
mu_road   = zeros(N, 1);
stage_id  = zeros(N, 1);

for k = 1:N
    time_k = t(k);
    if time_k < 15
        stage_id(k)  = 1;
        slope_deg(k) = 0;
        mu_road(k)   = 0.85;
    elseif time_k < 30
        stage_id(k)  = 2;
        slope_deg(k) = 10 * (1 - exp(-(time_k-15)/1.5));
        mu_road(k)   = 0.85;
    elseif time_k < 45
        stage_id(k)  = 3;
        slope_deg(k) = 10;
        mu_road(k)   = 0.85 - 0.35 * (1 - exp(-(time_k-30)/1.5));
    elseif time_k < 60
        stage_id(k)  = 4;
        s_trans = (time_k - 45) / 3.0;
        if s_trans <= 1
            slope_deg(k) = 10 - 20 * s_trans;
        else
            slope_deg(k) = -10;
        end
        mu_road(k) = 0.50 - 0.25 * (1 - exp(-(time_k-45)/1.5));
    else
        stage_id(k)  = 5;
        slope_deg(k) = -10 + 10 * min(1, (time_k-60)/2.0);
        mu_road(k)   = 0.25 + 0.35 * min(1, (time_k-60)/2.0);
    end
end

% Generate Continuous Reference Trajectory
X_ref = zeros(N, 1);
Y_ref = zeros(N, 1);
psi_ref = zeros(N, 1);
r_ref = zeros(N, 1);
delta_ref = zeros(N, 1);

for k = 1:N
    time_k = t(k);
    if time_k <= 60
        delta_ref(k) = 0.05 * sin((2*pi/30) * time_k);
    else
        t_lc = time_k - 60;
        if t_lc < 2.5
            delta_ref(k) = 0;
        elseif t_lc < 6.5
            delta_ref(k) = 0.08 * sin((2*pi/4) * (t_lc - 2.5));
        elseif t_lc < 10.5
            delta_ref(k) = -0.08 * sin((2*pi/4) * (t_lc - 6.5));
        else
            delta_ref(k) = 0;
        end
    end
end

Vx_track = max(double(Vx), 32.0);
for k = 2:N
    dt_k = t(k) - t(k-1);
    r_ref(k-1) = (Vx_track / L) * tan(delta_ref(k-1));
    psi_ref(k) = psi_ref(k-1) + r_ref(k-1) * dt_k;
    X_ref(k) = X_ref(k-1) + Vx_track * cos(psi_ref(k-1)) * dt_k;
    Y_ref(k) = Y_ref(k-1) + Vx_track * sin(psi_ref(k-1)) * dt_k;
end
r_ref(N) = (Vx_track / L) * tan(delta_ref(N));

% Select Controller (1=PID, 2=MPC, 3=RL)
if strcmpi(ctrl_req, 'pid')
    c_idx = 1;
    ctrl_label = 'Fixed PID (MATLAB)';
elseif strcmpi(ctrl_req, 'mpc')
    c_idx = 2;
    ctrl_label = 'MPC Preview (MATLAB)';
else
    c_idx = 3;
    ctrl_label = 'RL Adaptive Self-Tuning (MATLAB)';
end

% State variables
X_sim   = X_ref(1);
Y_sim   = Y_ref(1);
psi_sim = psi_ref(1);
Vy_sim  = 0;
r_sim   = 0;
prev_delta = 0;
int_ey = 0;
prev_ey = 0;
rl_state = [];

stage_names = {
    'STAGE 1: FLAT DRY CRUISE (0-15s, \mu=0.85)';
    'STAGE 2: STEEP UPHILL (+10 deg, \mu=0.85)';
    'STAGE 3: UPHILL RAIN (+10 deg, \mu=0.50)';
    'STAGE 4: DOWNHILL BLACK ICE (-10 deg, \mu=0.25)';
    'STAGE 5: EMERGENCY DOUBLE LANE CHANGE (\mu=0.60)'
};

stage_short_names = {
    'Stage 1: Flat Dry';
    'Stage 2: Steep Uphill';
    'Stage 3: Uphill Rain';
    'Stage 4: Downhill Ice';
    'Stage 5: Double Lane Change'
};

% Reference track points downsampled for the studio visualizer
ref_sub = 1:10:N;
ref_pts = [X_ref(ref_sub), Y_ref(ref_sub)];

% Streaming pace: send packet every 4 steps (0.04s = 25 Hz)
steps_per_packet = 4;
packet_dt = dt * steps_per_packet;

tic;
last_send_time = 0;
last_match_idx = 1;

for k = 1:N-1
    time_k = t(k);
    
    % Robust Frenet Orthogonal Path Projection (Velocity-Independent)
    w_min = max(1, last_match_idx - 5);
    w_max = min(N, last_match_idx + 120);
    dists_sq = (X_ref(w_min:w_max) - X_sim).^2 + (Y_ref(w_min:w_max) - Y_sim).^2;
    [~, best_rel] = min(dists_sq);
    j_match = w_min + best_rel - 1;
    last_match_idx = j_match;
    if j_match >= N - 15
        fprintf('>>> Proving Ground Mission Successfully Completed!\n');
        break;
    end
    
    dx_p = X_ref(j_match) - X_sim;
    dy_p = Y_ref(j_match) - Y_sim;
    psi_p = psi_ref(j_match);
    
    % Frenet Orthogonal Cross-Track Error (Positive = Path is left of vehicle -> steer left)
    ey = - dx_p * sin(psi_p) + dy_p * cos(psi_p);
    dpsi = atan2(sin(psi_p - psi_sim), cos(psi_p - psi_sim));
    
    if k > 1
        d_ey = (ey - prev_ey) / dt;
    else
        d_ey = 0;
    end
    prev_ey = ey;
    
    % Tire Normal Loads with Road Slope Load Transfer
    th_rad = deg2rad(slope_deg(k));
    Fzf = (m * g * (lr * cos(th_rad) - h_cg * sin(th_rad))) / L;
    Fzr = (m * g * (lf * cos(th_rad) + h_cg * sin(th_rad))) / L;
    
    Fyf_max = max(mu_road(k) * Fzf, 100);
    Fyr_max = max(mu_road(k) * Fzr, 100);
    
    % Controller Execution
    switch c_idx
        case 1 % Fixed PID
            int_ey = int_ey + ey * dt;
            int_ey = max(min(int_ey, 1.5), -1.5);
            u_pid = 0.80 * ey + 0.05 * int_ey + 0.05 * d_ey;
            delta_cmd = u_pid + 1.00 * dpsi;
            gains = [0.80, 0.05, 0.05, 1.00];
            
        case 2 % MPC / Preview PD+FF
            u_mpc = 0.95 * ey + 0.12 * d_ey + 1.15 * dpsi + (L/Vx)*r_ref(j_match);
            if mu_road(k) < 0.5, u_mpc = u_mpc * 0.85; end
            delta_cmd = u_mpc;
            gains = [0.95, 0.00, 0.12, 1.15];
            
        case 3 % RL Adaptive Controller
            [delta_cmd, gains, rl_state] = rl_adaptive_controller(ey, d_ey, ...
                psi_p, psi_sim, r_sim, r_ref(j_match), Vx, slope_deg(k), mu_road(k), dt, rl_state);
    end
    
    % Actuator Slew Rate and Saturation
    max_rate = 0.6 * dt;
    delta = prev_delta + max(min(delta_cmd - prev_delta, max_rate), -max_rate);
    delta = max(min(delta, delta_max), -delta_max);
    prev_delta = delta;
    
    % Tire Slip Angles & Forces
    alpha_f = delta - atan2((Vy_sim + lf * r_sim), Vx);
    alpha_r = - atan2((Vy_sim - lr * r_sim), Vx);
    
    Fyf = max(min(Cf * alpha_f, Fyf_max), -Fyf_max);
    Fyr = max(min(Cr * alpha_r, Fyr_max), -Fyr_max);
    
    % Vehicle Equations of Motion (2-DOF Bicycle Dynamics)
    ay = (Fyf * cos(delta) + Fyr) / m;
    d_Vy = ay - Vx * r_sim;
    d_r  = (lf * Fyf * cos(delta) - lr * Fyr) / Iz;
    d_psi = r_sim;
    
    d_X = Vx * cos(psi_sim) - Vy_sim * sin(psi_sim);
    d_Y = Vx * sin(psi_sim) + Vy_sim * cos(psi_sim);
    
    % Euler Integration
    Vy_sim  = Vy_sim + d_Vy * dt;
    r_sim   = r_sim + d_r * dt;
    psi_sim = psi_sim + d_psi * dt;
    X_sim   = X_sim + d_X * dt;
    Y_sim   = Y_sim + d_Y * dt;
    
    % Stream to EGGA Studio every 'steps_per_packet' steps
    if mod(k, steps_per_packet) == 0
        st_id = stage_id(k);
        frame = struct();
        frame.t = round(time_k, 3);
        frame.tick = k;
        frame.stage = st_id;
        frame.stage_name = stage_names{st_id};
        frame.stage_short = stage_short_names{st_id};
        frame.x = round(X_sim, 2);
        frame.y = round(Y_sim, 2);
        frame.x_ref = round(X_ref(j_match), 2);
        frame.y_ref = round(Y_ref(j_match), 2);
        frame.psi = round(psi_sim, 3);
        frame.psi_deg = round(rad2deg(psi_sim), 2);
        frame.psi_ref_deg = round(rad2deg(psi_p), 2);
        frame.epsi_deg = round(rad2deg(dpsi), 2);
        frame.vx = round(Vx, 2);
        frame.vy = round(Vy_sim, 2);
        frame.speed_kmh = round(Vx * 3.6, 1);
        frame.steer_deg = round(rad2deg(delta), 2);
        frame.lateral_error_cm = round(abs(ey) * 100, 1);
        frame.lateral_deviation_m = round(ey, 3);
        frame.slope_deg = round(slope_deg(k), 1);
        frame.mu = round(mu_road(k), 2);
        frame.ay_current = round(ay, 2);
        frame.Fzf = round(Fzf, 0);
        frame.Fzr = round(Fzr, 0);
        frame.alpha_f_deg = round(rad2deg(alpha_f), 2);
        frame.alpha_r_deg = round(rad2deg(alpha_r), 2);
        frame.Fyf = round(Fyf, 0);
        frame.Fyr = round(Fyr, 0);
        frame.controller = ctrl_label;
        frame.kp = round(gains(1), 2);
        frame.ki = round(gains(2), 3);
        frame.kd = round(gains(3), 3);
        frame.khead = round(gains(4), 2);
        
        % First frame: send reference trajectory coordinates once
        if k == steps_per_packet
            frame.ref_points_x = round(ref_pts(:, 1), 1);
            frame.ref_points_y = round(ref_pts(:, 2), 2);
        end
        
        % Transmit frame to Server via HTTP POST
        try
            resp = webwrite(stream_url, frame, options);
            if isfield(resp, 'stop_requested') && resp.stop_requested
                fprintf('Stop requested by server. Exiting loop.\n');
                break;
            end
            if isfield(resp, 'target_vx') && resp.target_vx > 0
                Vx = double(resp.target_vx);
            end
            if isfield(resp, 'speed_mult') && resp.speed_mult > 0
                speed_mult = double(resp.speed_mult);
            end
        catch err
            % If server temporarily busy, continue simulation
            if mod(k, 100) == 0
                fprintf('[MATLAB Bridge] Transmission note: %s\n', err.message);
            end
        end
        
        % Real-time synchronization pacing
        target_elapsed = (time_k / speed_mult);
        current_elapsed = toc;
        if current_elapsed < target_elapsed
            pause(target_elapsed - current_elapsed);
        end
    end
end
end


%% ========================================================================
% SUBFUNCTION: Master Dynamic Bicycle Simulation (from master_bicycle_simulation.m)
%% ========================================================================
function run_bicycle_master_simulation(stream_url, ctrl_req, traj_name, max_time, speed_mult, options, Vx_input)
dt = 0.01;
t = (0 : dt : max_time)';
N = length(t);

m = 1500; Iz = 3000; lf = 1.2; lr = 1.6; L = lf + lr;
Cf = 80000; Cr = 80000; delta_max = 0.5; Vx = max(double(Vx_input), 5.0); g = 9.81;

slope_deg = 0.0;
mu_road = 0.85;

% Generate Reference Trajectory
X_ref = zeros(N, 1);
Y_ref = zeros(N, 1);
psi_ref = zeros(N, 1);

for k = 1:N
    time_k = t(k);
    if strcmpi(traj_name, 'DoubleLaneChange')
        x_val = Vx * time_k;
        if x_val < 30
            y_val = 0;
        elseif x_val < 80
            y_val = 1.75 * (1 - cos(pi * (x_val - 30) / 50));
        elseif x_val < 130
            y_val = 3.5;
        elseif x_val < 180
            y_val = 3.5 - 1.75 * (1 - cos(pi * (x_val - 130) / 50));
        else
            y_val = 0;
        end
    elseif strcmpi(traj_name, 'Circle')
        R_circle = 40.0;
        omega = Vx / R_circle;
        x_val = R_circle * sin(omega * time_k);
        y_val = R_circle * (1 - cos(omega * time_k));
    else % Default SineWave
        x_val = Vx * time_k;
        y_val = 3.0 * sin(0.04 * x_val);
    end
    X_ref(k) = x_val;
    Y_ref(k) = y_val;
end

for k = 2:N-1
    psi_ref(k) = atan2(Y_ref(k+1) - Y_ref(k-1), X_ref(k+1) - X_ref(k-1));
end
psi_ref(1) = psi_ref(2);
psi_ref(N) = psi_ref(N-1);

X_sim = X_ref(1); Y_sim = Y_ref(1); psi_sim = psi_ref(1);
Vy_sim = 0; r_sim = 0; prev_delta = 0; int_ey = 0;

steps_per_packet = 4;
tic;
last_match_idx = 1;

for k = 1:N-1
    time_k = t(k);
    
    % Robust Frenet Orthogonal Path Projection (Velocity-Independent)
    w_min = max(1, last_match_idx - 5);
    w_max = min(N, last_match_idx + 120);
    dists_sq = (X_ref(w_min:w_max) - X_sim).^2 + (Y_ref(w_min:w_max) - Y_sim).^2;
    [~, best_rel] = min(dists_sq);
    j_match = w_min + best_rel - 1;
    last_match_idx = j_match;
    
    dx_p = X_ref(j_match) - X_sim;
    dy_p = Y_ref(j_match) - Y_sim;
    psi_p = psi_ref(j_match);
    
    ey = - dx_p * sin(psi_p) + dy_p * cos(psi_p);
    dpsi = atan2(sin(psi_p - psi_sim), cos(psi_p - psi_sim));
    
    % Controller: PID or Stanley
    if strcmpi(ctrl_req, 'stanley')
        k_stanley = 1.2;
        delta_cmd = dpsi + atan2(k_stanley * ey, Vx);
        ctrl_label = 'Stanley (MATLAB)';
    else
        int_ey = int_ey + ey * dt;
        delta_cmd = 0.85 * ey + 0.05 * int_ey + 1.10 * dpsi;
        ctrl_label = 'Dynamic Bicycle PID (MATLAB)';
    end
    
    delta = max(min(delta_cmd, delta_max), -delta_max);
    prev_delta = delta;
    
    alpha_f = delta - atan2((Vy_sim + lf * r_sim), Vx);
    alpha_r = - atan2((Vy_sim - lr * r_sim), Vx);
    Fyf = max(min(Cf * alpha_f, mu_road * m * g * 0.5), -mu_road * m * g * 0.5);
    Fyr = max(min(Cr * alpha_r, mu_road * m * g * 0.5), -mu_road * m * g * 0.5);
    
    ay = (Fyf * cos(delta) + Fyr) / m;
    d_Vy = ay - Vx * r_sim;
    d_r  = (lf * Fyf * cos(delta) - lr * Fyr) / Iz;
    d_psi = r_sim;
    
    Vy_sim  = Vy_sim + d_Vy * dt;
    r_sim   = r_sim + d_r * dt;
    psi_sim = psi_sim + d_psi * dt;
    X_sim   = X_sim + (Vx * cos(psi_sim) - Vy_sim * sin(psi_sim)) * dt;
    Y_sim   = Y_sim + (Vx * sin(psi_sim) + Vy_sim * cos(psi_sim)) * dt;
    
    if mod(k, steps_per_packet) == 0
        frame = struct();
        frame.t = round(time_k, 3);
        frame.tick = k;
        frame.stage = 1;
        frame.stage_name = ['BICYCLE MASTER: ' traj_name];
        frame.stage_short = traj_name;
        frame.x = round(X_sim, 2);
        frame.y = round(Y_sim, 2);
        frame.x_ref = round(X_ref(j_match), 2);
        frame.y_ref = round(Y_ref(j_match), 2);
        frame.psi = round(psi_sim, 3);
        frame.psi_deg = round(rad2deg(psi_sim), 2);
        frame.psi_ref_deg = round(rad2deg(psi_p), 2);
        frame.epsi_deg = round(rad2deg(dpsi), 2);
        frame.vx = round(Vx, 2);
        frame.vy = round(Vy_sim, 2);
        frame.speed_kmh = round(Vx * 3.6, 1);
        frame.steer_deg = round(rad2deg(delta), 2);
        frame.lateral_error_cm = round(abs(ey) * 100, 1);
        frame.lateral_deviation_m = round(ey, 3);
        frame.slope_deg = round(slope_deg, 1);
        frame.mu = round(mu_road, 2);
        frame.ay_current = round(ay, 2);
        frame.Fzf = round(m*g*0.5, 0);
        frame.Fzr = round(m*g*0.5, 0);
        frame.alpha_f_deg = round(rad2deg(alpha_f), 2);
        frame.alpha_r_deg = round(rad2deg(alpha_r), 2);
        frame.Fyf = round(Fyf, 0);
        frame.Fyr = round(Fyr, 0);
        frame.controller = ctrl_label;
        frame.kp = 0.85; frame.ki = 0.05; frame.kd = 0.05; frame.khead = 1.10;
        
        try
            resp = webwrite(stream_url, frame, options);
            if isfield(resp, 'stop_requested') && resp.stop_requested
                break;
            end
            if isfield(resp, 'target_vx') && resp.target_vx > 0
                Vx = double(resp.target_vx);
            end
            if isfield(resp, 'speed_mult') && resp.speed_mult > 0
                speed_mult = double(resp.speed_mult);
            end
        catch
        end
        
        target_elapsed = (time_k / speed_mult);
        current_elapsed = toc;
        if current_elapsed < target_elapsed
            pause(target_elapsed - current_elapsed);
        end
    end
end
end


%% ========================================================================
% SUBFUNCTION: Simulink Model Simulation & Streamer (BicyclePathTracking1.slx)
%% ========================================================================
function run_simulink_simulation(stream_url, legacy_dir, speed_mult, options, Vx_input)
fprintf('>>> Initializing MATLAB Simulink BicyclePathTracking1 Model...\n');

Vx = max(double(Vx_input), 5.0);

% Set up Base Workspace parameters for Simulink model
evalin('base', 'm   = 1500;');
evalin('base', 'Iz  = 3000;');
evalin('base', 'lf  = 1.2;');
evalin('base', 'lr  = 1.6;');
evalin('base', 'L   = 2.8;');
evalin('base', 'h   = 0.5;');
evalin('base', 'Cf  = 80000;');
evalin('base', 'Cr  = 80000;');
evalin('base', sprintf('vx  = %.2f;', Vx));
evalin('base', 'slope_deg = 0;');
evalin('base', 'mu        = 0.85;');
evalin('base', 'Kp           = 0.80;');
evalin('base', 'Ki           = 0.05;');
evalin('base', 'Kd           = 0.05;');
evalin('base', 'heading_gain = 1.00;');
evalin('base', 'delta_max    = 0.5;');

slx_compat = fullfile(legacy_dir, 'BicyclePathTracking1_R2025b.slx');
slx_orig   = fullfile(legacy_dir, 'BicyclePathTracking1.slx');

if exist(slx_compat, 'file')
    model_name = 'BicyclePathTracking1_R2025b';
    load_system(slx_compat);
else
    model_name = 'BicyclePathTracking1';
    load_system(slx_orig);
end

fprintf('>>> Running Simulink Model: %s...\n', model_name);
sim_out = sim(model_name);
fprintf('>>> Simulink model execution complete. Streaming live results to Studio...\n');

[t_vec, X_vec]       = extract_timeseries(sim_out.Xpos_log);
[~, Y_vec]           = extract_timeseries(sim_out.Ypos_log);
[~, Xref_vec]        = extract_timeseries(sim_out.Xref_log);
[~, Yref_vec]        = extract_timeseries(sim_out.Yref_log);
[~, ey_vec]          = extract_timeseries(sim_out.ey_log);
[~, psi_vec]         = extract_timeseries(sim_out.psi_log);
[~, r_vec]           = extract_timeseries(sim_out.r_log);
[~, vy_vec]          = extract_timeseries(sim_out.vy_log);
[~, ay_vec]          = extract_timeseries(sim_out.ay_log);
[~, delta_vec]       = extract_timeseries(sim_out.delta_log);
[~, alpha_f_vec]     = extract_timeseries(sim_out.alpha_f_log);
[~, alpha_r_vec]     = extract_timeseries(sim_out.alpha_r_log);
[~, Fyf_vec]         = extract_timeseries(sim_out.Fyf_log);
[~, Fyr_vec]         = extract_timeseries(sim_out.Fyr_log);

N = length(t_vec);
step_sample = max(1, floor(N / 120));
ref_pts_x = round(Xref_vec(1:step_sample:end), 2);
ref_pts_y = round(Yref_vec(1:step_sample:end), 2);

% Stream paced in real-time at dt = 0.04s (25 Hz)
dt_stream = 0.04;
t_max = t_vec(end);
t_stream = 0 : dt_stream : t_max;

tic;
for k = 1:length(t_stream)
    tk = t_stream(k);
    
    xk   = interp1(t_vec, X_vec, tk, 'linear', 'extrap');
    yk   = interp1(t_vec, Y_vec, tk, 'linear', 'extrap');
    xrk  = interp1(t_vec, Xref_vec, tk, 'linear', 'extrap');
    yrk  = interp1(t_vec, Yref_vec, tk, 'linear', 'extrap');
    eyk  = interp1(t_vec, ey_vec, tk, 'linear', 'extrap');
    psik = interp1(t_vec, psi_vec, tk, 'linear', 'extrap');
    rk   = interp1(t_vec, r_vec, tk, 'linear', 'extrap');
    vyk  = interp1(t_vec, vy_vec, tk, 'linear', 'extrap');
    ayk  = interp1(t_vec, ay_vec, tk, 'linear', 'extrap');
    delk = interp1(t_vec, delta_vec, tk, 'linear', 'extrap');
    afk  = interp1(t_vec, alpha_f_vec, tk, 'linear', 'extrap');
    ark  = interp1(t_vec, alpha_r_vec, tk, 'linear', 'extrap');
    fyfk = interp1(t_vec, Fyf_vec, tk, 'linear', 'extrap');
    fyrk = interp1(t_vec, Fyr_vec, tk, 'linear', 'extrap');
    
    frame = struct();
    frame.t = round(tk, 3);
    frame.tick = k;
    frame.stage = 1;
    frame.stage_name = 'SIMULINK MODEL (BicyclePathTracking1.slx)';
    frame.stage_short = 'SIMULINK';
    frame.x = round(xk, 2);
    frame.y = round(yk, 2);
    frame.x_ref = round(xrk, 2);
    frame.y_ref = round(yrk, 2);
    frame.psi = round(psik, 3);
    frame.psi_deg = round(rad2deg(psik), 2);
    frame.psi_ref_deg = round(rad2deg(atan2(gradient(Yref_vec, max(0.01, gradient(Xref_vec))), 1)), 2);
    frame.epsi_deg = round(rad2deg(psik - atan2(yrk - yk, max(0.1, xrk - xk))), 2);
    frame.vx = round(Vx, 2);
    frame.vy = round(vyk, 2);
    frame.speed_kmh = round(Vx * 3.6, 1);
    frame.steer_deg = round(rad2deg(delk), 2);
    frame.lateral_error_cm = round(abs(eyk) * 100, 1);
    frame.lateral_deviation_m = round(eyk, 3);
    frame.slope_deg = 0.0;
    frame.mu = 0.85;
    frame.ay_current = round(ayk, 2);
    frame.Fzf = 7357;
    frame.Fzr = 7357;
    frame.alpha_f_deg = round(rad2deg(afk), 2);
    frame.alpha_r_deg = round(rad2deg(ark), 2);
    frame.Fyf = round(fyfk, 0);
    frame.Fyr = round(fyrk, 0);
    frame.controller = 'Simulink PID Feedback (BicyclePathTracking1.slx)';
    frame.kp = 0.80; frame.ki = 0.05; frame.kd = 0.05; frame.khead = 1.00;
    frame.is_simulink = true;
    
    if k == 1
        frame.ref_points_x = ref_pts_x;
        frame.ref_points_y = ref_pts_y;
    end
    
    try
        resp = webwrite(stream_url, frame, options);
        if isfield(resp, 'stop_requested') && resp.stop_requested
            break;
        end
        if isfield(resp, 'speed_mult') && resp.speed_mult > 0
            speed_mult = double(resp.speed_mult);
        end
    catch
    end
    
    target_elapsed = (tk / speed_mult);
    current_elapsed = toc;
    if current_elapsed < target_elapsed
        pause(target_elapsed - current_elapsed);
    end
end
end


%% ========================================================================
% SUBFUNCTION: Standalone RL Gym Environment & Actor-Critic Evaluation
%% ========================================================================
function run_rl_env_simulation(stream_url, speed_mult, options, Vx_input)
fprintf('>>> Initializing Standalone RL Gym Environment (Actor-Critic)...\n');

dt = 0.01;
max_time = 60.0;
t = (0 : dt : max_time)';
N = length(t);

% Vehicle & Environment Parameters
m   = 1500; Iz = 3000; lf = 1.2; lr = 1.6; L = 2.8; h_cg = 0.5;
Cf  = 80000; Cr = 80000; Vx = max(double(Vx_input), 5.0); g = 9.81; delta_max = 0.5;

% Generate Dynamic Road Trajectory with Curvature Changes
X_ref = zeros(N, 1);
Y_ref = zeros(N, 1);
psi_ref = zeros(N, 1);
r_ref = zeros(N, 1);

for k = 1:N
    tk = t(k);
    X_ref(k) = Vx * tk;
    if tk < 15
        Y_ref(k) = 0.4 * sin(0.15 * tk);
    elseif tk < 35
        Y_ref(k) = 2.8 * sin(0.35 * (tk - 15));
    elseif tk < 50
        Y_ref(k) = 3.6 / (1 + exp(-1.2 * (tk - 40)));
    else
        Y_ref(k) = 3.6 + 0.3 * sin(0.20 * (tk - 50));
    end
    
    if k > 1
        dx = X_ref(k) - X_ref(k-1);
        dy = Y_ref(k) - Y_ref(k-1);
        psi_ref(k) = atan2(dy, dx);
        r_ref(k) = (psi_ref(k) - psi_ref(k-1)) / dt;
    end
end

% Pre-send reference track points
ref_pts_x = round(X_ref(1:150:end), 2);
ref_pts_y = round(Y_ref(1:150:end), 2);

% RL Simulation States
X_sim = 0; Y_sim = 0; psi_sim = 0; Vy_sim = 0; r_sim = 0;
prev_ey = 0;
rl_state = struct('int_e_y', 0, 'prev_delta', 0);

cumulative_reward = 0;
steps_per_packet = 4; % 25 Hz streaming
last_match_idx = 1;

tic;
for k = 1:N-1
    time_k = t(k);
    
    % Dynamic Terrain Scheduled per Episode
    if time_k < 15
        slope_deg = 0.0;
        mu_road = 0.85;
        stage_name = 'RL ENV: DRY ASPHALT NOMINAL';
    elseif time_k < 30
        slope_deg = 8.0;
        mu_road = 0.85;
        stage_name = 'RL ENV: UPHILL ADAPTATION (+8 deg)';
    elseif time_k < 45
        slope_deg = 8.0;
        mu_road = 0.45;
        stage_name = 'RL ENV: WET SLIPPERY SURFACE (mu=0.45)';
    else
        slope_deg = -6.0;
        mu_road = 0.30;
        stage_name = 'RL ENV: DOWNHILL ICE (mu=0.30, -6 deg)';
    end
    
    % Robust Frenet Orthogonal Path Projection (Velocity-Independent)
    w_min = max(1, last_match_idx - 5);
    w_max = min(N, last_match_idx + 120);
    dists_sq = (X_ref(w_min:w_max) - X_sim).^2 + (Y_ref(w_min:w_max) - Y_sim).^2;
    [~, best_rel] = min(dists_sq);
    j_match = w_min + best_rel - 1;
    last_match_idx = j_match;
    
    dx_p = X_ref(j_match) - X_sim;
    dy_p = Y_ref(j_match) - Y_sim;
    psi_p = psi_ref(j_match);
    
    ey = - dx_p * sin(psi_p) + dy_p * cos(psi_p);
    if k > 1
        d_ey = (ey - prev_ey) / dt;
    else
        d_ey = 0;
    end
    prev_ey = ey;
    dpsi = atan2(sin(psi_p - psi_sim), cos(psi_p - psi_sim));
    
    % Call Actor-Critic Policy Network (rl_adaptive_controller.m)
    [delta, gains_out, rl_state] = rl_adaptive_controller(...
        ey, d_ey, psi_p, psi_sim, r_sim, r_ref(j_match), Vx, slope_deg, mu_road, dt, rl_state);
    
    % Compute RL Reward from train_rl_agent.py
    step_reward = - (1.5 * ey^2 + 0.1 * dpsi^2 + 0.05 * delta^2);
    cumulative_reward = cumulative_reward + step_reward;
    
    % Dynamic Axle Loads & Slip Angles
    th_rad = deg2rad(slope_deg);
    Fzf = (m * g * (lr * cos(th_rad) - h_cg * sin(th_rad))) / L;
    Fzr = (m * g * (lf * cos(th_rad) + h_cg * sin(th_rad))) / L;
    Fyf_max = max(mu_road * Fzf, 100);
    Fyr_max = max(mu_road * Fzr, 100);
    
    alpha_f = delta - atan2(Vy_sim + lf * r_sim, Vx);
    alpha_r = - atan2(Vy_sim - lr * r_sim, Vx);
    Fyf = max(min(Cf * alpha_f, Fyf_max), -Fyf_max);
    Fyr = max(min(Cr * alpha_r, Fyr_max), -Fyr_max);
    
    ay = (Fyf * cos(delta) + Fyr) / m;
    d_Vy = ay - Vx * r_sim;
    d_r  = (lf * Fyf * cos(delta) - lr * Fyr) / Iz;
    d_psi = r_sim;
    
    Vy_sim  = Vy_sim + d_Vy * dt;
    r_sim   = r_sim + d_r * dt;
    psi_sim = psi_sim + d_psi * dt;
    X_sim   = X_sim + (Vx * cos(psi_sim) - Vy_sim * sin(psi_sim)) * dt;
    Y_sim   = Y_sim + (Vx * sin(psi_sim) + Vy_sim * cos(psi_sim)) * dt;
    
    if mod(k, steps_per_packet) == 0
        frame = struct();
        frame.t = round(time_k, 3);
        frame.tick = k;
        frame.stage = 1;
        frame.stage_name = stage_name;
        frame.stage_short = 'RL_ENV';
        frame.x = round(X_sim, 2);
        frame.y = round(Y_sim, 2);
        frame.x_ref = round(X_ref(j_match), 2);
        frame.y_ref = round(Y_ref(j_match), 2);
        frame.psi = round(psi_sim, 3);
        frame.psi_deg = round(rad2deg(psi_sim), 2);
        frame.psi_ref_deg = round(rad2deg(psi_p), 2);
        frame.epsi_deg = round(rad2deg(dpsi), 2);
        frame.vx = round(Vx, 2);
        frame.vy = round(Vy_sim, 2);
        frame.speed_kmh = round(Vx * 3.6, 1);
        frame.steer_deg = round(rad2deg(delta), 2);
        frame.lateral_error_cm = round(abs(ey) * 100, 1);
        frame.lateral_deviation_m = round(ey, 3);
        frame.slope_deg = round(slope_deg, 1);
        frame.mu = round(mu_road, 2);
        frame.ay_current = round(ay, 2);
        frame.Fzf = round(Fzf, 0);
        frame.Fzr = round(Fzr, 0);
        frame.alpha_f_deg = round(rad2deg(alpha_f), 2);
        frame.alpha_r_deg = round(rad2deg(alpha_r), 2);
        frame.Fyf = round(Fyf, 0);
        frame.Fyr = round(Fyr, 0);
        frame.controller = 'RL Actor-Critic Policy (Self-Tuning)';
        frame.kp = round(gains_out(1), 3);
        frame.ki = round(gains_out(2), 3);
        frame.kd = round(gains_out(3), 3);
        frame.khead = round(gains_out(4), 3);
        frame.reward = round(step_reward, 3);
        frame.cumulative_reward = round(cumulative_reward, 1);
        frame.is_rl_env = true;
        
        if k == steps_per_packet
            frame.ref_points_x = ref_pts_x;
            frame.ref_points_y = ref_pts_y;
        end
        
        try
            resp = webwrite(stream_url, frame, options);
            if isfield(resp, 'stop_requested') && resp.stop_requested
                break;
            end
            if isfield(resp, 'target_vx') && resp.target_vx > 0
                Vx = double(resp.target_vx);
            end
            if isfield(resp, 'speed_mult') && resp.speed_mult > 0
                speed_mult = double(resp.speed_mult);
            end
        catch
        end
        
        target_elapsed = (time_k / speed_mult);
        current_elapsed = toc;
        if current_elapsed < target_elapsed
            pause(target_elapsed - current_elapsed);
        end
    end
end
end

function [t, val] = extract_timeseries(sig)
if isa(sig, 'timeseries')
    t = sig.Time;
    val = squeeze(sig.Data);
elseif isstruct(sig) && isfield(sig, 'signals')
    t = sig.time;
    val = squeeze(sig.signals.values);
else
    t = [];
    val = double(sig);
end
val = val(:);
end
