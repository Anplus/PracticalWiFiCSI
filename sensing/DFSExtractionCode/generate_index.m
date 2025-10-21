clc;clear;close all;
%%
% read all files in the subfolders
% then generate shuffle.txt into the parent folder, where each line is a path of a data file
data_path = '/home/myid/za70400/Documents/Lecture/2025Fall-4900/20181109_processed';
% there are multiple user subfolders under data_path, collect all .mat files in the user subfolders
file_list = dir(fullfile(data_path, '**', '*.mat'));
num_files = length(file_list);
% generate a cell array to store the full paths
file_paths = cell(num_files, 1);
for i = 1:num_files
    file_paths{i} = fullfile(file_list(i).folder, file_list(i).name);
end
% shuffle the file paths
shuffled_indices = randperm(num_files);
shuffled_file_paths = file_paths(shuffled_indices);
% write to shuffle.txt
output_file = fullfile('../', 'shuffle.txt');
fileID = fopen(output_file, 'w');
for i = 1:num_files
    fprintf(fileID, '%s\n', shuffled_file_paths{i});
end
fclose(fileID);
disp(['Generated ', output_file]);
