
clc
clear all
close all

%%
variables = load ('variables.mat');

files = ["PS1";"PS2";"PS3";"PS4";"PS5";"PS6";"TS1";"TS2";"TS3";"TS4";"FS1";"FS2";"CP";"CE";"EPS1";"SE";"VS1";];
ext = ["_mu", "_var", "_K", "_S", "_max"];

feat_name = [];
for i = 1 : 17
    for j = 1 : 5
        feat_name = [feat_name, files(i) + ext(j)];
    end
end

%{
features = [];
for i = 1 : 17
    x = importdata('./data/' + files(i)  + '.txt');
    x_norm = (x - (sum(x) ./ length(x))) ./ (max(x) - min(x));
    x1 = statmeasure_vec(x_norm);
    features = [features, x1];
    clear x1;
end

label = importdata('./data/l1.txt');
ground_truth = importdata('./data/profile.txt');
%}

features = variables.features;
features_t = features';
features_table = array2table(features);
features_table.Properties.VariableNames = cellstr(feat_name);

labels_col1 = variables.label1;
labels_col2 = variables.label2;
labels_col3 = variables.label3;
labels_col4 = variables.label4;
labels_col5 = variables.label5;

[idx_col1, score_col1] = fscmrmr(features, labels_col1);
[idx_col2, score_col2] = fscmrmr(features, labels_col2);
[idx_col3, score_col3] = fscmrmr(features, labels_col3);
[idx_col4, score_col4] = fscmrmr(features, labels_col4);
[idx_col5, score_col5] = fscmrmr(features, labels_col5);

selected_indices_col1 = find(idx_col1 < 15);
selected_indices_col2 = find(idx_col2 < 15);
selected_indices_col3 = find(idx_col3 < 15);
selected_indices_col4 = find(idx_col4 < 15);
selected_indices_col5 = find(idx_col5 < 15);

figure;
bar(score_col1(idx_col1));
title('Features ranked by metric for cooler condition');
xlabel('Predictor rank');
ylabel('Predictor importance score');
xticks([1 : 85]);
xticklabels(strrep(features_table.Properties.VariableNames(idx_col1), '_', '\_'))
xtickangle(90)

figure;
bar(score_col2(idx_col2));
xlabel('Predictor rank');
ylabel('Predictor importance score');
title('Features ranked by metric for valve condition');
xticks([1 : 85]);
xticklabels(strrep(features_table.Properties.VariableNames(idx_col2), '_', '\_'))
xtickangle(90)

figure;
bar(score_col3(idx_col3));
xlabel('Predictor rank');
ylabel('Predictor importance score');
title('Features ranked by metric for internal pump leakage');
xticks([1 : 85]);
xticklabels(strrep(features_table.Properties.VariableNames(idx_col3), '_', '\_'))
xtickangle(90)

figure;
bar(score_col4(idx_col4));
xlabel('Predictor rank');
ylabel('Predictor importance score');
title('Features ranked by metric for hydraulic accumulator');
xticks([1 : 85]);
xticklabels(strrep(features_table.Properties.VariableNames(idx_col4), '_', '\_'))
xtickangle(90)

figure;
bar(score_col5(idx_col5));
xlabel('Predictor rank');
ylabel('Predictor importance score');
title('Features ranked by metric for stable flag');
xticks([1 : 85]);
xticklabels(strrep(features_table.Properties.VariableNames(idx_col5), '_', '\_'))
xtickangle(90)

data_col1 = [features(:, selected_indices_col1), labels_col1];
data_col2 = [features(:, selected_indices_col2), labels_col2];
data_col3 = [features(:, selected_indices_col3), labels_col3];
data_col4 = [features(:, selected_indices_col4), labels_col4];
data_col5 = [features(:, selected_indices_col5), labels_col5];

test_col1 = ind2vec(labels_col1');
test_col2 = ind2vec(labels_col2');
test_col3 = ind2vec(labels_col3');
test_col4 = ind2vec(labels_col4');

test_col1_t = test_col1';
test_col2_t = test_col2';
test_col3_t = test_col3';
test_col4_t = test_col4';
test_col5_t = labels_col5';