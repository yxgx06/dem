%% build_mex.m
% Compiles the EGGA Supervisor C MEX S-Function for MATLAB/Simulink.
% Requirements: MATLAB R2020b or later with MinGW-w64 or Visual Studio compiler.

disp('--- Building EGGA Supervisor MEX S-Function ---');

repo_root = fileparts(fileparts(pwd));
c_dir = fullfile(repo_root, 'src', 'c');

sources = {
    'egga_supervisor_sfun.c', ...
    fullfile(c_dir, 'supervisor.c'), ...
    fullfile(c_dir, 'supervisor_envelope_data.c'), ...
    fullfile(c_dir, 'actor.c'), ...
    fullfile(c_dir, 'friction_circle.c'), ...
    fullfile(c_dir, 'jackknife_guard.c')
};

include_flag = ['-I' c_dir];

mex_cmd = [{'mex', '-O', include_flag}, sources];
disp(['Executing: ', strjoin(mex_cmd, ' ')]);

mex(mex_cmd{:});

disp('Build Complete: egga_supervisor_sfun MEX binary generated successfully.');
