function [comm_ind, anomaly_cell, cor_cell, score_global, rank_anomaly, rank_local] = DSGD_CNMF_FindOutlier()

% Load data and factor matrices
load('disney/X2.mat');              % attribute matrix
load('disney/X2_H.mat');            % community attr property factor matrix
load('disney/X2_W.mat');            % community association factor matrix
% Ground truth for Disney data
true_anomaly = [18, 36, 45, 66, 80, 120] + 1;


thre = 0.2;                         % threshold for community assignment
num_comm = size(W, 2);

comm_ind = cell(1, num_comm);       % nodes id for each community
anomaly_cell = cell(1, num_comm);   % regularity of each node in each community
cor_cell = cell(1, num_comm);
regu_max = zeros(1, num_comm);      % most regular node in each community
rank_local = cell(1, num_comm);

w_diag = 1./sum(W, 2);
for i = 1:length(w_diag)
    W(i, :) = W(i, :) * w_diag(i);
end

% community assignment
[~, maxC] = max(W, [], 2);

for C = 1:num_comm
    comm_ind{C} = find(W(:,C) > thre);
    if size(comm_ind{C}, 1) == 0
        continue
    end
    
    attr_C = X(comm_ind{C},:);

    anomaly_factors = H(C,:);
    score_local = attr_C*anomaly_factors'./diag(sqrt(attr_C*attr_C'));
    regu_max(C) = sqrt(anomaly_factors*anomaly_factors');
    
    score_local = score_local/regu_max(C);
    anomaly_cell{C} = score_local';
    
    [~, indexes] = sort(anomaly_cell{C});
    rank_local{C} = comm_ind{C}(indexes) - 1;
end

score_global = ones(1, size(X, 1)) * realmax;
for C = 1:num_comm
    for i = 1:length(comm_ind{C})
        if anomaly_cell{C}(i) < score_global(comm_ind{C}(i))
            score_global(comm_ind{C}(i)) = anomaly_cell{C}(i);
        end
    end
end

[~, rank_anomaly] = sort(score_global);

if ~isempty(true_anomaly)
    [~, auc_PR] = BDSGD_AUC(rank_anomaly, true_anomaly);
    auc_PR
end