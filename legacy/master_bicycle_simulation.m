%% =========================================================================
% MASTER AUTONOMOUS VEHICLE DYNAMIC BICYCLE SIMULATION WITH LIVE ANIMATION
% =========================================================================
% ALL-IN-ONE STANDALONE MATLAB SCRIPT
%
% This single file contains:
%   1. Parameter Configuration (Vehicle, Slopes, Friction, Controllers)
%   2. Reference Path Generation (Sine wave, Double Lane Change, Circle)
%   3. 2-DOF Dynamic Bicycle Model with Load Transfer and Friction Limits
%   4. Four Selectable Controllers: PID, Stanley, LQR, MPC
%   5. Real-Time Interactive Car Driving Animation with Dashboard HUD
%   6. Comprehensive Multi-Scope Plotting
%
% HOW TO RUN:
%   1. Open MATLAB.
%   2. Open this file: master_bicycle_simulation.m
%   3. Click "RUN" (or press F5) in MATLAB editor!
% =========================================================================

clear; clc; close all;

%% =========================================================================
% SECTION 1: USER CONFIGURATION (CHANGE YOUR PARAMETERS HERE!)
% =========================================================================

% --- LIVE VISUAL ANIMATION TOGGLE ---
enable_animation = true; % Set to true for live moving car animation!
animation_speed  = 3.0;  % Animation playback speed multiplier (1x, 2x, 5x)

% --- A. ROAD CONDITIONS (SLOPE & FRICTION) ---
slope_deg   = 0;        % Road slope: 0 = Flat, +10 = Uphill, -10 = Downhill
mu_friction = 0.85;     % Road friction: 0.85 = Dry, 0.50 = Wet, 0.25 = Icy

% --- B. CONTROLLER SELECTION ---
% Choose ONE: 'PID', 'STANLEY', 'LQR', or 'MPC'
controller_type = 'PID'; 

% --- C. TRAJECTORY SELECTION ---
% Choose ONE: 'SineWave', 'DoubleLaneChange', or 'Circle'
trajectory_type = 'SineWave'; 

% --- D. VEHICLE PARAMETERS (2-DOF Dynamic Bicycle Model) ---
m         = 1500;       % Vehicle mass (kg)
Iz        = 3000;       % Yaw moment of inertia (kg*m^2)
lf        = 1.2;        % CG to front axle distance (m)
lr        = 1.6;        % CG to rear axle distance (m)
L         = lf + lr;    % Wheelbase (2.8 m)
h_cg      = 0.5;        % Height of Center of Gravity (m)
Cf        = 80000;      % Front axle cornering stiffness (N/rad)
Cr        = 80000;      % Rear axle cornering stiffness (N/rad)
delta_max = 0.5;        % Max steering limit (rad, ~28.6 deg)
d_delta_max = 0.6;      % Max steering slew rate (rad/s)
g         = 9.81;       % Gravity (m/s^2)

% --- E. SIMULATION SETTINGS ---
Vx        = 10.0;       % Constant longitudinal velocity (m/s) [36 km/h]
T_final   = 30.0;       % Total simulation time (seconds)
dt        = 0.01;       % Time step (seconds)

% --- F. CONTROLLER TUNING GAINS ---
% 1. PID Tuning:
Kp_pid    = 0.80;       % Proportional gain on lateral error
Ki_pid    = 0.05;       % Integral gain
Kd_pid    = 0.05;       % Derivative gain
Khead_pid = 1.00;       % Heading feedforward gain

% 2. Stanley Tuning:
k_stanley = 1.50;       % Cross-track error gain
k_soft    = 1.00;       % Softening velocity term (m/s)

% 3. LQR Tuning:
Q_lqr     = diag([10, 1, 20, 1]); % State penalty [e_y, d_e_y, e_psi, d_e_psi]
R_lqr     = 5.0;                  % Steering effort penalty

% 4. MPC Tuning:
Np_mpc    = 15;         % Prediction horizon steps
Nc_mpc    = 5;          % Control horizon steps
Q_lat_mpc = 15;         % Lateral error penalty
Q_head_mpc= 25;         % Heading error penalty
R_u_mpc   = 2.0;        % Steering angle penalty
R_du_mpc  = 5.0;        % Steering rate penalty

%% =========================================================================
% SECTION 2: REFERENCE PATH GENERATION
% =========================================================================
t = (0 : dt : T_final)';
N = length(t);

X_ref = zeros(N, 1);
Y_ref = zeros(N, 1);
psi_ref = zeros(N, 1);
r_ref = zeros(N, 1);
delta_ref = zeros(N, 1);

switch lower(trajectory_type)
    case 'sinewave'
        freq = 2 * pi / 30;
        delta_ref = 0.05 * sin(freq * t);
        for k = 2:N
            dt_k = t(k) - t(k-1);
            r_ref(k-1) = (Vx / L) * tan(delta_ref(k-1));
            psi_ref(k) = psi_ref(k-1) + r_ref(k-1) * dt_k;
            X_ref(k) = X_ref(k-1) + Vx * cos(psi_ref(k-1)) * dt_k;
            Y_ref(k) = Y_ref(k-1) + Vx * sin(psi_ref(k-1)) * dt_k;
        end
        r_ref(N) = (Vx / L) * tan(delta_ref(N));
        
    case 'doublelanechange'
        for k = 1:N
            X_est = Vx * t(k);
            X_ref(k) = X_est;
            if X_est < 25
                Y_ref(k) = 0;
            elseif X_est <= 45
                s = (X_est - 25) / 20;
                Y_ref(k) = 3.5 * (3*s^2 - 2*s^3);
            elseif X_est <= 65
                Y_ref(k) = 3.5;
            elseif X_est <= 85
                s = (X_est - 65) / 20;
                Y_ref(k) = 3.5 * (1 - (3*s^2 - 2*s^3));
            else
                Y_ref(k) = 0;
            end
        end
        dY = gradient(Y_ref, t);
        dX = gradient(X_ref, t);
        psi_ref = atan2(dY, dX);
        r_ref = gradient(psi_ref, t);
        delta_ref = atan(L * r_ref / max(Vx, 1.0));
        
    case 'circle'
        R_circle = 40;
        omega = Vx / R_circle;
        r_ref = omega * ones(N, 1);
        psi_ref = omega * t;
        X_ref = R_circle * sin(omega * t);
        Y_ref = R_circle * (1 - cos(omega * t));
        delta_ref = atan(L / R_circle) * ones(N, 1);
end

%% =========================================================================
% SECTION 3: ROAD INCLINE LOAD TRANSFER & TIRE FRICTION LIMITS
% =========================================================================
theta_rad = deg2rad(slope_deg);

% Static normal axle loads under slope angle theta:
F_zf_static = (m * g * (lr * cos(theta_rad) - h_cg * sin(theta_rad))) / L;
F_zr_static = (m * g * (lf * cos(theta_rad) + h_cg * sin(theta_rad))) / L;

% Maximum lateral force capacity (Coulomb friction limit):
Fyf_max = max(mu_friction * F_zf_static, 100);
Fyr_max = max(mu_friction * F_zr_static, 100);

%% =========================================================================
% SECTION 4: PREPARE CONTROLLER MATRICES (LQR & MPC)
% =========================================================================
A_cont = [ 0,             1,                      0,                               0;
           0, -(Cf + Cr)/(m*Vx),             (Cf + Cr)/m,           (-Cf*lf + Cr*lr)/(m*Vx);
           0,             0,                      0,                               1;
           0, (-Cf*lf + Cr*lr)/(Iz*Vx),      (Cf*lf - Cr*lr)/Iz,    (-Cf*lf^2 - Cr*lr^2)/(Iz*Vx) ];

B_cont = [ 0; 
           Cf/m; 
           0; 
           (lf*Cf)/Iz ];

% 1. LQR Gain Matrix K:
try
    [K_lqr, ~] = lqr(A_cont, B_cont, Q_lqr, R_lqr);
catch
    Ad_lqr = eye(4) + A_cont * dt;
    Bd_lqr = B_cont * dt;
    P_mat = Q_lqr;
    for iter = 1:150
        P_next = Q_lqr + Ad_lqr'*P_mat*Ad_lqr - (Ad_lqr'*P_mat*Bd_lqr)/(R_lqr + Bd_lqr'*P_mat*Bd_lqr)*(Bd_lqr'*P_mat*Ad_lqr);
        if max(max(abs(P_next - P_mat))) < 1e-6, break; end
        P_mat = P_next;
    end
    K_lqr = (R_lqr + Bd_lqr'*P_mat*Bd_lqr) \ (Bd_lqr'*P_mat*Ad_lqr);
end

% 2. MPC Prediction Matrices:
Ad_mpc = eye(4) + A_cont * dt;
Bd_mpc = B_cont * dt;
C_mpc  = [1, 0, 0, 0; 0, 0, 1, 0];

Sx_mpc = zeros(2 * Np_mpc, 4);
Su_mpc = zeros(2 * Np_mpc, Nc_mpc);
A_pow = eye(4);
for i = 1:Np_mpc
    A_pow = A_pow * Ad_mpc;
    Sx_mpc((i-1)*2+1 : i*2, :) = C_mpc * A_pow;
    for j = 1:min(i, Nc_mpc)
        A_sub = eye(4);
        for kk = 1:(i-j), A_sub = A_sub * Ad_mpc; end
        Su_mpc((i-1)*2+1 : i*2, j) = C_mpc * A_sub * Bd_mpc;
    end
end

Q_blk_mpc = kron(eye(Np_mpc), diag([Q_lat_mpc, Q_head_mpc]));
R_blk_mpc = kron(eye(Nc_mpc), R_u_mpc);
D_rate_mpc = eye(Nc_mpc);
for i = 2:Nc_mpc, D_rate_mpc(i, i-1) = -1; end
R_rate_blk_mpc = kron(eye(Nc_mpc), R_du_mpc);

H_mpc = 2 * (Su_mpc' * Q_blk_mpc * Su_mpc + R_blk_mpc + D_rate_mpc' * R_rate_blk_mpc * D_rate_mpc);
H_mpc = (H_mpc + H_mpc') / 2 + 1e-4 * eye(Nc_mpc);

%% =========================================================================
% SECTION 5: INITIALIZE VEHICLE STATES & LOGGING ARRAYS
% =========================================================================
X         = zeros(N, 1);
Y         = zeros(N, 1);
psi       = zeros(N, 1);
Vy        = zeros(N, 1);
r         = zeros(N, 1);
ay        = zeros(N, 1);
delta     = zeros(N, 1);
alpha_f   = zeros(N, 1);
alpha_r   = zeros(N, 1);
Fyf       = zeros(N, 1);
Fyr       = zeros(N, 1);
e_y_body  = zeros(N, 1);
delta_psi = zeros(N, 1);

X(1)   = X_ref(1);
Y(1)   = Y_ref(1);
psi(1) = psi_ref(1);

int_ey = 0;
prev_delta_mpc = 0;

%% =========================================================================
% SECTION 6: MAIN SIMULATION LOOP (NUMERICAL INTEGRATION)
% =========================================================================
fprintf('Running 2-DOF Dynamic Bicycle Simulation (%s Controller)...\n', controller_type);

for k = 1:N-1
    % --- 6.1 Tracking Errors in Vehicle Body Frame ---
    ex_glob = X_ref(k) - X(k);
    ey_glob = Y_ref(k) - Y(k);
    e_y_body(k) = ey_glob * cos(psi(k)) - ex_glob * sin(psi(k));
    
    dpsi_raw = psi_ref(k) - psi(k);
    delta_psi(k) = atan2(sin(dpsi_raw), cos(dpsi_raw));
    
    if k > 1
        d_ey = (e_y_body(k) - e_y_body(k-1)) / dt;
    else
        d_ey = 0;
    end
    
    % --- 6.2 Controller Execution ---
    switch upper(controller_type)
        case 'PID'
            int_ey = int_ey + e_y_body(k) * dt;
            int_ey = max(min(int_ey, 2.0), -2.0);
            u_pid = Kp_pid * e_y_body(k) + Ki_pid * int_ey + Kd_pid * d_ey;
            delta_cmd = u_pid + Khead_pid * delta_psi(k);
            
        case 'STANLEY'
            e_front = e_y_body(k) + lf * sin(delta_psi(k));
            delta_cmd = delta_psi(k) + atan2(k_stanley * e_front, Vx + k_soft);
            
        case 'LQR'
            x_state = [e_y_body(k); d_ey; delta_psi(k); r(k) - r_ref(k)];
            kappa = r_ref(k) / max(Vx, 1.0);
            Kv = (lr*m/(2*Cf*L)) - (lf*m/(2*Cr*L));
            delta_ff = L * kappa + Kv * (Vx^2) * kappa;
            delta_cmd = -K_lqr * x_state + delta_ff;
            
        case 'MPC'
            x0_mpc = [e_y_body(k); d_ey; delta_psi(k); r(k) - r_ref(k)];
            f_mpc = 2 * (Su_mpc' * Q_blk_mpc * (Sx_mpc * x0_mpc)) - 2 * [R_du_mpc * prev_delta_mpc; zeros(Nc_mpc-1, 1)];
            
            U_opt = zeros(Nc_mpc, 1);
            alpha_step = 1 / max(eig(H_mpc));
            for qp_iter = 1:35
                grad = H_mpc * U_opt + f_mpc;
                U_opt = max(min(U_opt - alpha_step * grad, delta_max), -delta_max);
            end
            delta_cmd = U_opt(1);
            max_rate = d_delta_max * dt;
            delta_cmd = prev_delta_mpc + max(min(delta_cmd - prev_delta_mpc, max_rate), -max_rate);
            prev_delta_mpc = delta_cmd;
    end
    
    delta(k) = max(min(delta_cmd, delta_max), -delta_max);
    
    % --- 6.3 Dynamic Bicycle Slip Angles ---
    alpha_f(k) = delta(k) - atan2((Vy(k) + lf * r(k)), Vx);
    alpha_r(k) = - atan2((Vy(k) - lr * r(k)), Vx);
    
    % --- 6.4 Lateral Tire Forces with Friction Limits ---
    Fyf(k)  = max(min(Cf * alpha_f(k), Fyf_max), -Fyf_max);
    Fyr(k)  = max(min(Cr * alpha_r(k), Fyr_max), -Fyr_max);
    
    % --- 6.5 Dynamic Bicycle Equations of Motion ---
    ay(k) = (Fyf(k) * cos(delta(k)) + Fyr(k)) / m;
    d_Vy  = ay(k) - Vx * r(k);
    d_r   = (lf * Fyf(k) * cos(delta(k)) - lr * Fyr(k)) / Iz;
    d_psi = r(k);
    
    d_X = Vx * cos(psi(k)) - Vy(k) * sin(psi(k));
    d_Y = Vx * sin(psi(k)) + Vy(k) * cos(psi(k));
    
    % Forward Euler Integration
    Vy(k+1)  = Vy(k) + d_Vy * dt;
    r(k+1)   = r(k) + d_r * dt;
    psi(k+1) = psi(k) + d_psi * dt;
    X(k+1)   = X(k) + d_X * dt;
    Y(k+1)   = Y(k) + d_Y * dt;
end

e_y_body(N)  = (Y_ref(N) - Y(N))*cos(psi(N)) - (X_ref(N) - X(N))*sin(psi(N));
delta_psi(N) = delta_psi(N-1);
delta(N)     = delta(N-1);
alpha_f(N)   = alpha_f(N-1);
alpha_r(N)   = alpha_r(N-1);
Fyf(N)       = Fyf(N-1);
Fyr(N)       = Fyr(N-1);
ay(N)        = ay(N-1);

fprintf('Simulation calculations complete!\n');

%% =========================================================================
% SECTION 7: LIVE VEHICLE ANIMATION ON 2D TRACK (INTERACTIVE PLAYBACK)
% =========================================================================
if enable_animation
    fprintf('\n>>> PLAYING LIVE VEHICLE SIMULATION ANIMATION (Watch the car drive!)...\n');
    anim_fig = figure('Name', 'Live Vehicle Autonomous Driving Simulation', ...
                      'Color', [0.12, 0.12, 0.15], 'Position', [80, 80, 1050, 620]);
    
    % Trajectory Map Axis
    ax_map = subplot('Position', [0.08, 0.12, 0.58, 0.78]);
    plot(X_ref, Y_ref, '--', 'Color', [0.3, 0.75, 1.0], 'LineWidth', 2.0); hold on;
    h_trail = plot(X(1), Y(1), 'Color', [1.0, 0.25, 0.25], 'LineWidth', 2.2);
    
    % Vehicle Polygon (Car chassis: 4.2m long, 1.8m wide)
    car_W = 1.8;
    car_corners = [-lr, -car_W/2;
                    lf + 0.8, -car_W/2;
                    lf + 0.8,  car_W/2;
                   -lr,  car_W/2]';
    h_car = patch('XData', car_corners(1,:), 'YData', car_corners(2,:), ...
                  'FaceColor', [1.0, 0.82, 0.1], 'EdgeColor', 'w', 'LineWidth', 1.5);
    h_wheel_f = plot([0,0],[0,0], 'Color', [0.1, 1.0, 0.3], 'LineWidth', 4.5);
    
    grid on; box on; axis equal;
    set(ax_map, 'Color', [0.18, 0.18, 0.22], 'XColor', 'w', 'YColor', 'w', 'GridColor', [0.4, 0.4, 0.4]);
    xlabel('Global X Position (m)', 'Color', 'w', 'FontWeight', 'bold');
    ylabel('Global Y Position (m)', 'Color', 'w', 'FontWeight', 'bold');
    title(sprintf('Live Vehicle Trajectory Tracking (%s Controller)', controller_type), 'Color', 'w', 'FontSize', 12);
    legend({'Reference Path', 'Actual Vehicle Trail', 'Vehicle Body (CG)'}, 'TextColor', 'w', 'Location', 'northwest');
    xlim([min(X_ref)-10, max(X_ref)+15]);
    ylim([min(Y_ref)-15, max(Y_ref)+15]);
    
    % Telemetry Dashboard Box
    ax_hud = subplot('Position', [0.70, 0.12, 0.26, 0.78]);
    axis off;
    set(ax_hud, 'Color', [0.12, 0.12, 0.15]);
    
    text(0.05, 0.95, 'LIVE TELEMETRY', 'Color', [0.2, 0.9, 1.0], 'FontSize', 13, 'FontWeight', 'bold');
    text(0.05, 0.84, sprintf('Controller:  %s', controller_type), 'Color', 'w', 'FontSize', 11);
    text(0.05, 0.76, sprintf('Road Slope:  %.1f deg', slope_deg), 'Color', 'w', 'FontSize', 11);
    text(0.05, 0.68, sprintf('Friction mu: %.2f', mu_friction), 'Color', 'w', 'FontSize', 11);
    
    hud_time  = text(0.05, 0.55, 'Sim Time:    0.00 s', 'Color', [1, 1, 0.3], 'FontSize', 12, 'FontWeight', 'bold');
    hud_err   = text(0.05, 0.45, 'Cross Error: 0.00 cm', 'Color', [0.3, 1.0, 0.4], 'FontSize', 12, 'FontWeight', 'bold');
    hud_steer = text(0.05, 0.35, 'Steer Angle: 0.00 deg', 'Color', 'w', 'FontSize', 11);
    hud_yaw   = text(0.05, 0.25, 'Yaw Rate r:  0.00 rad/s', 'Color', 'w', 'FontSize', 11);
    hud_slip  = text(0.05, 0.15, 'Front Slip:  0.00 deg', 'Color', 'w', 'FontSize', 11);
    hud_ay    = text(0.05, 0.05, 'Lat Accel:   0.00 m/s^2', 'Color', 'w', 'FontSize', 11);
    
    % Step skip for smooth visual playback (~8 to 10 seconds animation)
    step_skip = 15; 
    for k_anim = 1:step_skip:N
        if ~ishandle(anim_fig), break; end
        
        % Update vehicle trail
        set(h_trail, 'XData', X(1:k_anim), 'YData', Y(1:k_anim));
        
        % Rotate and translate vehicle body polygon
        R_mat = [cos(psi(k_anim)), -sin(psi(k_anim)); sin(psi(k_anim)), cos(psi(k_anim))];
        poly_rot = R_mat * car_corners;
        set(h_car, 'XData', poly_rot(1,:) + X(k_anim), 'YData', poly_rot(2,:) + Y(k_anim));
        
        % Update front steer line
        wheel_dir = [cos(psi(k_anim)+delta(k_anim)), -sin(psi(k_anim)+delta(k_anim)); 
                     sin(psi(k_anim)+delta(k_anim)),  cos(psi(k_anim)+delta(k_anim))] * [1.2, -1.2; 0, 0];
        front_pos = [X(k_anim) + lf*cos(psi(k_anim)); Y(k_anim) + lf*sin(psi(k_anim))];
        set(h_wheel_f, 'XData', front_pos(1) + wheel_dir(1,:), 'YData', front_pos(2) + wheel_dir(2,:));
        
        % Update Dashboard Telemetry values
        set(hud_time,  'String', sprintf('Sim Time:    %.2f s', t(k_anim)));
        set(hud_err,   'String', sprintf('Cross Error: %+.2f cm', e_y_body(k_anim)*100));
        set(hud_steer, 'String', sprintf('Steer Angle: %+.2f deg', rad2deg(delta(k_anim))));
        set(hud_yaw,   'String', sprintf('Yaw Rate r:  %+.2f rad/s', r(k_anim)));
        set(hud_slip,  'String', sprintf('Front Slip:  %+.2f deg', rad2deg(alpha_f(k_anim))));
        set(hud_ay,    'String', sprintf('Lat Accel:   %+.2f m/s^2', ay(k_anim)));
        
        drawnow;
        pause(0.015); % Smooth pause for human eye
    end
end

%% =========================================================================
% SECTION 8: DISPLAY QUANTITATIVE PERFORMANCE METRICS
% =========================================================================
max_e_y = max(abs(e_y_body));
rms_e_y = sqrt(mean(e_y_body.^2));
max_psi_e = max(abs(delta_psi));
max_ay = max(abs(ay));
max_af = rad2deg(max(abs(alpha_f)));
max_ar = rad2deg(max(abs(alpha_r)));
effort = trapz(t, delta.^2);

fprintf('\n========================================================\n');
fprintf('  SIMULATION COMPLETED SUCCESSFULLY! RESULTS SUMMARY    \n');
fprintf('========================================================\n');
fprintf('Active Controller:            %s\n', controller_type);
fprintf('Road Condition:               Slope = %.1f deg, Friction mu = %.2f\n', slope_deg, mu_friction);
fprintf('--------------------------------------------------------\n');
fprintf('Max Lateral Tracking Error:   %.4f m  (%.2f cm)\n', max_e_y, max_e_y * 100);
fprintf('RMS Lateral Tracking Error:   %.4f m  (%.2f cm)\n', rms_e_y, rms_e_y * 100);
fprintf('Max Heading Angle Error:      %.4f rad (%.2f deg)\n', max_psi_e, rad2deg(max_psi_e));
fprintf('Peak Lateral Acceleration:    %.2f m/s^2 (%.2f g)\n', max_ay, max_ay / 9.81);
fprintf('Peak Front Tire Slip Angle:   %.2f deg\n', max_af);
fprintf('Peak Rear Tire Slip Angle:    %.2f deg\n', max_ar);
fprintf('Front Normal Axle Load (Fzf): %.1f N\n', F_zf_static);
fprintf('Rear Normal Axle Load (Fzr):  %.1f N\n', F_zr_static);
fprintf('Steering Effort (Integral):   %.4f rad^2*s\n', effort);
fprintf('========================================================\n\n');

%% =========================================================================
% SECTION 9: MULTI-PANEL SCOPE PLOTTING (MATCHING SIMULINK SCOPES)
% =========================================================================
fig = figure('Name', sprintf('Autonomous Vehicle Scopes - %s Controller', controller_type), ...
             'Color', 'w', 'Position', [60, 40, 1150, 780]);

% 1. Scope: Vehicle Dynamics (ay, r, vy)
subplot(3, 2, 1);
plot(t, ay, 'LineWidth', 1.8, 'Color', [0.85, 0.33, 0.1]); hold on;
plot(t, r, 'LineWidth', 1.8, 'Color', [0, 0.45, 0.74]);
plot(t, Vy, 'LineWidth', 1.5, 'Color', [0.47, 0.67, 0.19]);
grid on; box on;
xlabel('Time (s)', 'FontWeight', 'bold');
ylabel('Dynamic Units', 'FontWeight', 'bold');
title('(a) Scope Vehicle Dynamics (ay, r, vy)', 'FontWeight', 'bold');
legend('a_y (m/s^2)', 'r (rad/s)', 'v_y (m/s)', 'Location', 'best');

% 2. Scope: Body Lateral Tracking Error (ey_body)
subplot(3, 2, 2);
plot(t, e_y_body * 100, 'LineWidth', 2.0, 'Color', [0.64, 0.08, 0.18]);
grid on; box on;
xlabel('Time (s)', 'FontWeight', 'bold');
ylabel('Lateral Error e_y (cm)', 'FontWeight', 'bold');
title(sprintf('(b) Scope Body Lateral Error (Max = %.2f cm)', max_e_y * 100), 'FontWeight', 'bold');

% 3. Scope: Tire Lateral Forces (Fyf, Fyr)
subplot(3, 2, 3);
plot(t, Fyf, 'LineWidth', 1.8, 'Color', [0.93, 0.69, 0.13]); hold on;
plot(t, Fyr, 'LineWidth', 1.8, 'Color', [0, 0.45, 0.74]);
grid on; box on;
xlabel('Time (s)', 'FontWeight', 'bold');
ylabel('Lateral Force (N)', 'FontWeight', 'bold');
title('(c) Scope Tire Lateral Forces', 'FontWeight', 'bold');
legend('F_{yf} (Front)', 'F_{yr} (Rear)', 'Location', 'best');

% 4. Scope: Tire Slip Angles (alpha_f, alpha_r)
subplot(3, 2, 4);
plot(t, rad2deg(alpha_f), 'LineWidth', 1.8, 'Color', [0.93, 0.69, 0.13]); hold on;
plot(t, rad2deg(alpha_r), 'LineWidth', 1.8, 'Color', [0, 0.45, 0.74]);
grid on; box on;
xlabel('Time (s)', 'FontWeight', 'bold');
ylabel('Slip Angle (deg)', 'FontWeight', 'bold');
title('(d) Scope Slip Angles (\alpha_f, \alpha_r)', 'FontWeight', 'bold');
legend('\alpha_f (Front)', '\alpha_r (Rear)', 'Location', 'best');

% 5. Scope: Steering Input and Heading
subplot(3, 2, 5);
plot(t, rad2deg(delta), 'LineWidth', 1.8, 'Color', [0.49, 0.18, 0.56]); hold on;
plot(t, rad2deg(psi), 'LineWidth', 1.8, 'Color', [0, 0.45, 0.74]);
plot(t, rad2deg(psi_ref), '--k', 'LineWidth', 1.2);
grid on; box on;
xlabel('Time (s)', 'FontWeight', 'bold');
ylabel('Angle (deg)', 'FontWeight', 'bold');
title('(e) Scope Steering and Heading (\delta, \psi)', 'FontWeight', 'bold');
legend('\delta (Steering)', '\psi (Vehicle Heading)', '\psi_{ref} (Path Heading)', 'Location', 'best');

% 6. Scope: 2D Global Path Tracking (X-Y)
subplot(3, 2, 6);
plot(X_ref, Y_ref, '--k', 'LineWidth', 2.0); hold on;
plot(X, Y, 'r-', 'LineWidth', 1.8);
grid on; box on; axis equal;
xlabel('Global X (m)', 'FontWeight', 'bold');
ylabel('Global Y (m)', 'FontWeight', 'bold');
title(sprintf('(f) 2D Path Tracking (%s Controller)', controller_type), 'FontWeight', 'bold');
legend('Reference Path', 'Vehicle Trajectory', 'Location', 'best');

fprintf('All Scopes and Visualizations Rendered Successfully!\n');
