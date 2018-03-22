function [auc_ROC, auc_PR] = BDSGD_AUC(ret, act)

num_act = length(act);

ranks = zeros(1, num_act);
for a = 1:num_act
    ranks(a) = find(ret == act(a));
end
ranks = sort(ranks);

% ROC AUC
x1s = zeros(1, num_act);
y1s = zeros(1, num_act);
for a = 1:num_act
    rank_a = ranks(a);
    dr = a/(num_act);
    flr = (rank_a - a)/(length(ret) - num_act);
    x1s(a) = flr;
    y1s(a) = dr;
end

auc_ROC = trapz([0,x1s,1], [0,y1s,1]);

% PR AUC
x2s = zeros(1, num_act);
y2s = zeros(1, num_act);
for a = 1:num_act
    rank_a = ranks(a);
    prec = a/rank_a;
    rec = a/(num_act);
    x2s(a) = rec;
    y2s(a) = prec;
end

auc_PR = trapz([0,x2s],[y2s(1),y2s]);
