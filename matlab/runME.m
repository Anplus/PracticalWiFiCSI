clearvars
close all;
%% Load data (channels, ap, ap_aoas, RSSI, labels, opt, d1, d2)
CHAN_DATA_DIR = './';
DATASET_NAME = 'channels_jacobs_July28.mat';
load(fullfile(CHAN_DATA_DIR, DATASET_NAME));
%% define parameters
D_VALS = -10:0.1:30; % Vector of Distance search space
THETA_VALS = -pi/2:0.01:pi/2; % Vector of AoA search space
y_len = length(yLabels);
x_len = length(xLabels);
% d1 and d2
d1 = xLabels;
d2 = yLabels;
%%
figure; hold on; grid on;
xlabel('X (m)'); ylabel('Y (m)');
title('AP antenna positions (4 APs × 4 antennas each)');
axis equal;
colors  = {'r','b','g','m'};   
markers = {'o','s','^','d'};

for k = 1:numel(ap)   % 
    pos = ap{k};      % 4x2 matrix
    plot(pos(:,1), pos(:,2), markers{k}, ...
        'MarkerFaceColor', colors{k}, ...
        'MarkerEdgeColor', 'k', ...
        'DisplayName', sprintf('AP %d', k));
    
    center = mean(pos,1);
    plot(center(1), center(2), 'x', ...
        'Color', colors{k}, 'MarkerSize', 10, ...
        'LineWidth', 2, 'HandleVisibility','off');
    
    text(center(1)+0.2, center(2), sprintf('AP%d',k), ...
        'Color', colors{k}, 'FontWeight','bold');
end
plot(labels(:,1), labels(:,2));
legend show;

%% 
% channels [n_datapoints x n_frequency x n_ant X n_ap]
[n_datapoints, n_freq, n_ant, n_ap] = size(channels);
ap_index = 3;
% for demo, only process the first 10 samples
n_datapoints = 10;
channels = channels(1:n_datapoints, :,:,:);
features = zeros(n_datapoints, n_ap, y_len, x_len);
%% generate features
% parfor i = 1:n_datapoints
for i = 1:n_datapoints
    features(i,:,:,:) = generate_features_from_channel(squeeze(channels(i,:,:,:)),ap,...
        THETA_VALS,D_VALS,d1,d2,ap_index,opt);
end

%% plot heatmap
figure; tiledlayout(2,2);
sample_index = 1;
for k = 1:4
    % Get 2-D matrix for AP k from features:
    C = squeeze(features(sample_index, k, :, :));        % try [ny x nx]
    % If sizes are swapped, transpose:
    if size(C,1) == numel(d1) && size(C,2) == numel(d2)
        C = C.';   % make it [ny x nx]
    end

    nexttile;
    imagesc(d1, d2, C);       % C must be [length(d2) x length(d1)]
    axis xy; axis image; colormap hot; colorbar;

    xlabel('X (m)'); ylabel('Y (m)');
    title(sprintf('AP %d heatmap', k));

    % plot AP position
    hold on;
    ap_k = ap{k};   % 4x2
    plot(ap_k(:,1), ap_k(:,2), 'wo', 'MarkerFaceColor','c', ...
        'MarkerSize',6, 'DisplayName','Antenna');
    % AP center
    center = mean(ap_k,1);
    plot(center(1), center(2), 'wx', 'MarkerSize',30, 'LineWidth',2);
    
    
    % plot the position of transmitter
    plot(labels(sample_index, 1), labels(sample_index,2), 'kp', 'MarkerFaceColor','y', ...
        'MarkerSize',20, 'DisplayName','Transmitter');
    hold off;
end