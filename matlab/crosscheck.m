function crosscheck(outfile)
% CROSSCHECK Run the unmodified legacy mission (sections 1-4 only, no animation or plots)
% and write golden values for tests/test_matlab_crosscheck.py.
%   matlab -batch "cd matlab; crosscheck('../results/phase0/matlab_golden.json')"
if nargin < 1
    outfile = fullfile('..', 'results', 'phase0', 'matlab_golden.json');
end
legacy = fullfile(fileparts(mfilename('fullpath')), '..', 'legacy');
addpath(legacy);
txt = fileread(fullfile(legacy, 'mission_proving_ground_rl.m'));
cut = strfind(txt, '%% 5. LIVE ANIMATED');
[results, t] = run_core(txt(1:cut(1)-1));

names = {'pid', 'pd_ff', 'rl_handtyped'};
out = struct();
for i = 1:3
    r = results.(sprintf('c%d', i));
    s = struct();
    s.max_abs_ey_cm = r.max_error;
    s.rms_ey_cm = r.rms_error;
    idx = 1:100:numel(t);
    s.ey_m_every_1s = r.ey(idx)';
    out.(names{i}) = s;
end
out.matlab_version = version;
fid = fopen(outfile, 'w');
fprintf(fid, '%s', jsonencode(out, 'PrettyPrint', true));
fclose(fid);
fprintf('wrote %s\n', outfile);
end

function [results, t] = run_core(code) %#ok<INUSD>
% eval is intentional: it runs only this repo's own legacy script text (read from legacy/),
% never external input, so the legacy file stays untouched.
eval(code);
end
