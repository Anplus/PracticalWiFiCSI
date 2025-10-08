clearvars
close all;
%% Load data (channels, ap, ap_aoa, RSSI, labels, opt, d1, d2)
CHAN_DATA_DIR = '/Users/za70400/Documents/Dataset/DLoc';
DATASET_NAME = 'channels_jacobs_July28.mat';
load(fullfile(CHAN_DATA_DIR, DATASET_NAME));
%% define parameters
D_VALS = -10:0.1:30;
THETA_VALS = -pi/2:0.01:pi/2;
y_len = length(yLabels);
x_len = length(xLabels);
ap_index = 3;
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


[n_datapoints, n_freq, n_ant, n_ap] = size(channels);
% for demo, only process the first 10 samples
n_datapoints = 10;
channels = channels(1:n_datapoints, :,:,:);
features = zeros(n_datapoints, n_ap, y_len, x_len);
% 11440 4 161 361
%% generate features
for i = 1:n_datapoints
    temp = generate_features_from_channel(channels(i,:,:,:),ap,...
        THETA_VALS,D_VALS,d1,d2,ap_index,opt);
    features(i,:,:,:) = temp;
end

%% plot heatmap
figure; tiledlayout(2,2);
sample_index = 10;
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
    plot(center(1), center(2), 'wx', 'MarkerSize',10, 'LineWidth',2);
    
    
    % plot the position of transmitter
    plot(labels(sample_index, 1), labels(sample_index,2), 'kp', 'MarkerFaceColor','y', ...
        'MarkerSize',10, 'DisplayName','Transmitter');
    hold off;
end