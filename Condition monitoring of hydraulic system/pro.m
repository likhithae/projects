
clc
clear
close 

%%

files = ["PS1";"PS2";"PS3";"PS4";"PS5";"PS6";"TS1";"TS2";"TS3";"TS4";"FS1";"FS2";"CP";"CE";"EPS1";"SE";"VS1";];
features = [];
for i=1:17
    x = importdata('./data/'+files(i)+'.txt');
    x1 = statmeasure_vec(x);
    features = [features,x1];
    clear x1;
end

label = importdata('./data/profile.txt');

index=corrSel(features,label(:,2));
selected_features=features(:,index(:,1:20));


%{
[idx,scores] = fscmrmr(features,label);
figure;
bar(scores(idx));
xlabel('Predictor rank');
ylabel('Predictor importance score');

%}

feat = features';
%data = [features(:,1),features(:,40),features(:,41),features(:,42),features(:,43),features(:,44),features(:,16),features(:,26),features(:,9),features(:,1),features(:,33),label];
data = [selected_features,label(:,2)];
%test = ind2vec(label');
%test1 = test';


