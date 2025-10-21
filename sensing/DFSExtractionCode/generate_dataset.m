clc;clear;close all;
addpath("csi_tool_box");
addpath("tftb/");
%% batch_process_csi.m
% Walk a "date" directory whose subfolders are users (user1, user2, ...).
% For each *.dat file, run csi_get_all(filename) and save cfr_array/timestamp
% into a mirrored directory tree named "<date>_processed".
%
% Usage:
%   - Set input_root to your top-level "date" directory.
%   - Run this script. Processed .mat files will be written to "<date>_processed".
%
% Requirements:
%   - Your function csi_get_all.m must be on the MATLAB path.

%% --- CONFIG ---
input_root = '/home/myid/za70400/Documents/Lecture/2025Fall-4900/20181109';   % e.g., '/data/2025-10-21'
append_tag = '_processed';               % output root = [input_root append_tag]
save_version = '-v7.3';                  % use '-v7' if your arrays are small

%% --- DERIVE OUTPUT ROOT ---
[input_parent, input_name, ~] = fileparts(input_root);
if isempty(input_parent) && ~isempty(input_name)
    % Handle case like input_root = '2025-10-21' (relative path)
    output_root = [input_root append_tag];
else
    output_root = fullfile(input_parent, [input_name append_tag]);
end

% Make sure output root exists
make_dir(output_root);

%% --- LIST USER SUBDIRS ---
d_users = dir(input_root);
is_user_dir = [d_users.isdir] & ~ismember({d_users.name},{'.','..'});
user_dirs = d_users(is_user_dir);

if isempty(user_dirs)
    warning('No user subdirectories found under: %s', input_root);
end

%% --- LOGGING SETUP ---
log_file = fullfile(output_root, 'processing_log.txt');
log_fid = fopen(log_file, 'w');
logmsg(log_fid, 'Input root: %s', input_root);
logmsg(log_fid, 'Output root: %s', output_root);
logmsg(log_fid, 'Start time: %s', datestr(now));

%% --- PROCESS EACH USER ---
for ui = 1:numel(user_dirs)
    user_name = user_dirs(ui).name;
    in_user_dir  = fullfile(input_root,  user_name);
    out_user_dir = fullfile(output_root, user_name);
    make_dir(out_user_dir);

    % Find *.dat files in this user folder (non-recursive; add '**/*.dat' if needed on newer MATLABs)
    dat_list = dir(fullfile(in_user_dir, '*.dat'));
    if isempty(dat_list)
        logmsg(log_fid, '[SKIP] %s: no .dat files found.', in_user_dir);
        continue;
    end

    logmsg(log_fid, '[USER] %s: %d .dat files found.', user_name, numel(dat_list));

    % Loop files
    for fi = 1:numel(dat_list)
        dat_name = dat_list(fi).name;
        in_file  = fullfile(in_user_dir, dat_name);

        [~, base, ~] = fileparts(dat_name);
        out_file = fullfile(out_user_dir, [base '.mat']);

        try
            % --- PROCESS ---
            [cfr_array, timestamp] = csi_get_all(in_file);

            % --- SAVE (only the two variables) ---
            save(out_file, 'cfr_array', 'timestamp', save_version);

            logmsg(log_fid, '  [OK] %s -> %s', in_file, out_file);
        catch ME
            logmsg(log_fid, '  [ERR] %s :: %s', in_file, ME.message);
        end
    end
end

logmsg(log_fid, 'Done. End time: %s', datestr(now));
fclose(log_fid);

fprintf('All done.\nProcessed files are in:\n  %s\nA log is at:\n  %s\n', output_root, log_file);

%% --- HELPERS ---
function make_dir(p)
    if exist(p, 'dir') ~= 7
        ok = mkdir(p);
        if ~ok
            error('Failed to create directory: %s', p);
        end
    end
end

function logmsg(fid, fmt, varargin)
    msg = sprintf(fmt, varargin{:});
    ts  = datestr(now, 'yyyy-mm-dd HH:MM:SS.FFF');
    line = sprintf('[%s] %s\n', ts, msg);
    fprintf('%s', line);        % echo to console
    if fid > 0
        fprintf(fid, '%s', line);
    end
end
