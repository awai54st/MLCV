%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% experiment with Caltech101 dataset for image categorisation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

init;

% Select dataset
% we do bag-of-words technique to convert images to vectors (histogram of codewords)
% Set 'showImg' in getData.m to 0 to stop displaying training and testing images and their feature vectors

[data_train, data_test] = getData('Caltech_kmeans');
close all;%, classList_train



% Set the random forest parameters ...
param.num = 300;%10; % Number of trees
param.depth = 10; % trees depth
param.splitNum = 20;%3; % Number of split functions to try
param.split = 'IG'; % Currently support 'information gain' only
param.split_func = 1;

% Train Random Forest ...
duration = zeros(10,1);
accuracy_test = zeros(10,1);
for count = 1:10
    tic
[trees,ig_best] = growTrees(data_train,param);

% Evaluate/Test Random Forest ...
predictions = zeros(size(data_test,1),1);
p_rf_sum = zeros(10,size(data_test,1));

for n=1:size(data_test,1)
leaves = testTrees(data_test(n,:),trees,param);
% disp(leaves);
% average the class distributions of leaf nodes of all trees
p_rf = trees(1).prob(leaves,:);
p_rf_sum(:,n) = sum(p_rf)/length(trees);
[~,predictions(n)] = max(p_rf_sum(:,n));
end
duration(count) = toc
% show accuracy and confusion matrix ...
accuracy_test(count) = sum(predictions==data_test(:,end))/size(data_test,1);
% conf = confusionmat(data_test(:,end), predictions);
end

accuracy_test_mean = mean(accuracy_test)
duration_mean = mean(duration)

% str = input('conf200_4_5_1 ','s');
% save([str,'.mat'],'conf');
load handel
sound(y,Fs)

% for n=1:size(data_train,1)
% leaves = testTrees(data_train(n,:),trees,param);
% % disp(leaves);
% % average the class distributions of leaf nodes of all trees
% p_rf = trees(1).prob(leaves,:);
% p_rf_sum(:,n) = sum(p_rf)/length(trees);
% [~,predictions(n)] = max(p_rf_sum(:,n));
% end
% 
% % show accuracy and confusion matrix ...
% accuracy_train = sum(predictions==data_train(:,end))/size(data_train,1);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% random forest codebook for Caltech101 image categorisation
% .....

init;

[data_train, data_test] = getData('Caltech_RFCB');
close all;

% Set the random forest parameters ...
param.num = 350;%10; % Number of trees
param.depth = 9; % trees depth
param.splitNum = 10;%3; % Number of split functions to try
param.split = 'IG'; % Currently support 'information gain' only
param.split_func = 1;

% Train Random Forest ...
[trees,ig_best] = growTrees(data_train,param);

% Evaluate/Test Random Forest ...
predictions = zeros(size(data_test,1),1);
p_rf_sum = zeros(10,size(data_test,1));

for n=1:size(data_train,1)
leaves = testTrees(data_train(n,:),trees,param);
% disp(leaves);
% average the class distributions of leaf nodes of all trees
p_rf = trees(1).prob(leaves,:);
p_rf_sum(:,n) = sum(p_rf)/length(trees);
[~,predictions(n)] = max(p_rf_sum(:,n));
end

% show accuracy and confusion matrix ...
accuracy_train = sum(predictions==data_train(:,end))/size(data_train,1);

for n=1:size(data_test,1)
leaves = testTrees(data_test(n,:),trees,param);
% disp(leaves);
% average the class distributions of leaf nodes of all trees
p_rf = trees(1).prob(leaves,:);
p_rf_sum(:,n) = sum(p_rf)/length(trees);
[~,predictions(n)] = max(p_rf_sum(:,n));
end

% show accuracy and confusion matrix ...
accuracy_test = sum(predictions==data_test(:,end))/size(data_test,1);
% 0.52 0.9867

