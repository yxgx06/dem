%% RUN_IN_SIMULINK.M - Loads Parameters & Runs the Simulink Model Directly
%
% This script automatically sets the MATLAB path, loads all vehicle variables
% into the Base Workspace, opens the scopes, and executes `BicyclePathTracking1.slx`.
%
% HOW TO USE:
%   1. Open MATLAB.
%   2. Run this script: run_in_simulink
%   3. Watch the Simulink simulation run with live scopes!

clear; clc;

fprintf('====================================================\n');
fprintf('  INITIALIZING SIMULINK BICYCLE PATH TRACKING       \n');
fprintf('====================================================\n');

% 1. Automatically set Current Directory and Path to this folder
current_folder = fileparts(mfilename('fullpath'));
if ~isempty(current_folder)
    cd(current_folder);
    addpath(current_folder);
end

% 2. Check that the Simulink file exists
slx_file = fullfile(current_folder, 'BicyclePathTracking1.slx');
if ~exist(slx_file, 'file') && ~exist('BicyclePathTracking1.slx', 'file')
    error('Cannot find BicyclePathTracking1.slx in current folder: %s', current_folder);
end

% 3. Load all Vehicle & Road Parameters into Base Workspace
fprintf('Loading vehicle parameters into MATLAB Workspace...\n');

% Vehicle & Geometry
m   = 1500;       % Mass (kg)
Iz  = 3000;       % Yaw inertia (kg*m^2)
lf  = 1.2;        % CG to front axle (m)
lr  = 1.6;        % CG to rear axle (m)
L   = lf + lr;    % Wheelbase (2.8 m)
h   = 0.5;        % CG height (m)
Cf  = 80000;      % Front cornering stiffness (N/rad)
Cr  = 80000;      % Rear cornering stiffness (N/rad)
vx  = 10.0;       % Forward speed (m/s)

% Road slope & friction
slope_deg = 0;    % 0 = Flat, +10 = Uphill, -10 = Downhill
mu        = 0.85; % 0.85 = Dry, 0.50 = Wet, 0.25 = Ice

% Path Tracking Controller (PID + Heading Gains)
Kp           = 0.80;
Ki           = 0.05;
Kd           = 0.05;
heading_gain = 1.00;
delta_max    = 0.5;

% 4. Open and Run the Simulink Model
fprintf('Opening Simulink Model: BicyclePathTracking1.slx...\n');

% Load system into memory
load_system('BicyclePathTracking1');
open_system('BicyclePathTracking1');

fprintf('Running Simulink Simulation...\n');
sim('BicyclePathTracking1');

fprintf('\n====================================================\n');
fprintf('  SIMULINK SIMULATION COMPLETED SUCCESSFULLY!       \n');
fprintf('====================================================\n');
fprintf('All scopes inside BicyclePathTracking1 have been updated with fresh simulation data.\n');
