clc;close all;
%%
sample_id = 1;   
numAP = size(features_wo_offset,2);

% 
tx_xy = labels(sample_id,:);   % [x,y]

figure;
tiledlayout(2,numAP);

for k = 1:numAP
    % ---- no offset ----
    C1 = squeeze(features_wo_offset(sample_id, k, :, :));
    nexttile;
    imagesc(xLabels, yLabels, C1);
    axis xy; axis image;
    colormap hot; colorbar;
    hold on;
    plot(tx_xy(1), tx_xy(2), 'wp', 'MarkerSize',12, ...
         'MarkerFaceColor','y','DisplayName','Tx');
    text(tx_xy(1)+0.2, tx_xy(2), 'Tx', 'Color','w','FontWeight','bold');
    hold off;
    xlabel('X (m)'); ylabel('Y (m)');
    title(sprintf('AP %d - wo offset', k));

    % ---- with offset ----
    C2 = squeeze(features_w_offset(sample_id, k, :, :));
    nexttile;
    imagesc(xLabels, yLabels, C2);
    axis xy; axis image;
    colormap hot; colorbar;
    hold on;
    plot(tx_xy(1), tx_xy(2), 'wp', 'MarkerSize',12, ...
         'MarkerFaceColor','y','DisplayName','Tx');
    text(tx_xy(1)+0.2, tx_xy(2), 'Tx', 'Color','w','FontWeight','bold');
    hold off;
    xlabel('X (m)'); ylabel('Y (m)');
    title(sprintf('AP %d - w offset', k));
end