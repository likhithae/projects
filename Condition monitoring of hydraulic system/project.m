% clc
% clear
% close 

%%
%{

f1 = statmeasure_vec(importdata('./data/PS1.txt'));
f2 = statmeasure_vec(importdata('./data/PS2.txt'));
f3 = statmeasure_vec(importdata('./data/PS3.txt'));
f4 = statmeasure_vec(importdata('./data/PS4.txt'));
f5 = statmeasure_vec(importdata('./data/PS5.txt'));
f6 = statmeasure_vec(importdata('./data/PS6.txt'));
f7 = statmeasure_vec(importdata('./data/TS1.txt'));
f8 = statmeasure_vec(importdata('./data/TS2.txt'));
f9 = statmeasure_vec(importdata('./data/TS3.txt'));
f10 = statmeasure_vec(importdata('./data/TS4.txt'));
f11 = statmeasure_vec(importdata('./data/FS1.txt'));
f12 = statmeasure_vec(importdata('./data/FS2.txt'));
f13 = statmeasure_vec(importdata('./data/CP.txt'));
f14 = statmeasure_vec(importdata('./data/CE.txt'));
f15 = statmeasure_vec(importdata('./data/EPS1.txt'));
f16 = statmeasure_vec(importdata('./data/SE.txt'));
f17 = statmeasure_vec(importdata('./data/VS1.txt'));

features1 = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17];

label = importdata('./data/l1.txt');

%features = load('features.mat');
%features = features.features;
%profile = load('profile.mat');
%profile = profile.profile;
train = features';
data = [features,label];
test = ind2vec(label');
test1 = test';

%}

[trainedClassifier_col3_lda, tenfold] = col3_lda(data_col3);
predicted_lda_col3 = trainedClassifier_col3_lda.predictFcn(features(:, selected_indices_col3));
f1metrics_lda_col3 = MyClassifyPerf(labels_col3, predicted_lda_col3)

[trainedClassifier_col3_svm, tenfold] = col3_svm(data_col3);
predicted_svm_col3 = trainedClassifier_col3_svm.predictFcn(features(:, selected_indices_col1));
f1metrics_svm_col3 = MyClassifyPerf(labels_col3, predicted_svm_col3)