% Simple Random Forest Toolbox for Matlab
% written by Mang Shao and Tae-Kyun Kim, June 20, 2014.
% updated by Tae-Kyun Kim, Feb 09, 2017

% This is a guideline script of simple-RF toolbox.
% The codes are made for educational purposes only.
% Some parts are inspired by Karpathy's RF Toolbox

% Under BSD Licence

%%
clear all; close all; clc;
% Initialisation
init;

% Select dataset
[data_train, data_test] = getData('Toy_Spiral'); % {'Toy_Gaussian', 'Toy_Spiral', 'Toy_Circle', 'Caltech'}


%%%%%%%%%%%%%
% check the training and testing data
% data_train(:,1:2) : [num_data x dim] Training 2D vectors
% data_train(:,3) : [num_data x 1] Labels of training data, {1,2,3}

plot_toydata(data_train);

% data_test(:,1:2) : [num_data x dim] Testing 2D vectors, 2D points in the
% uniform dense grid within the range of [-1.5, 1.5]
% data_train(:,3) : N/A

scatter(data_test(:,1),data_test(:,2),'.b');

%% Question 1 Generate 4 subsets
%%%%%%%%%%%%%%%%%%%%% Bagging %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
frac = 1 - 1/exp(1);
[N,D] = size(data_train);
for subset = 1:4
    idx{subset} = randsample(N,ceil(N*frac),1);
end
subset1 = data_train(idx{1,1}(:,1),:);
subset2 = data_train(idx{1,2}(:,1),:);
subset3 = data_train(idx{1,3}(:,1),:);
subset4 = data_train(idx{1,4}(:,1),:);
figure
subplot(2,2,1)
plot_toydata(subset1)
title('subset 1 by bagging')
subplot(2,2,2)
plot_toydata(subset2)
title('subset 2 by bagging')
subplot(2,2,3)
plot_toydata(subset3)
title('subset 3 by bagging')
subplot(2,2,4)
plot_toydata(subset4)
title('subset 4 by bagging')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Question 1 IG against Randomness (Using pre-generated subset 1)
clear all; close all; clc;
init;
load('subsets.mat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% Very Important! %%%%%%%%
%%% Do not do bagging for this test
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set the random forest parameters for instance,
param.num = 100;%10;         % Number of trees
param.depth = 5;        % trees depth
param.splitNum = 20;%3;     % Number of split functions to try
param.split = 'IG';     % Currently support 'information gain' only
param.split_func = 1;

%%%%%%%%%%%%%%%%%%%%%%
% Train Random Forest
ig_best_split_rho = zeros(4,50);
for rho = 1:200
    for split = 1:4
        param.split_func = split;
        param.splitNum = rho;
        [trees,ig_best] = growTrees_nobag(subset1,param);
        ig_best_split_rho(split,rho) = ig_best;
    end
end

figure
plot(1:size(ig_best_split_rho,2),ig_best_split_rho(1,:),'--','LineWidth',1);hold on
plot(1:size(ig_best_split_rho,2),ig_best_split_rho(2,:),':','lineWidth',1);
plot(1:size(ig_best_split_rho,2),ig_best_split_rho(3,:),'-*','lineWidth',1);
plot(1:size(ig_best_split_rho,2),ig_best_split_rho(4,:),'-','lineWidth',1);hold off
set(gca, 'LineWidth',2,'FontSize',18)
title('Information gain vs randomness')
legend('axis-aligned','linear','conic','two-pixel test')
xlabel('Number of Splitting Trials')
ylabel('Information Gain')
grid on

figure;
visualise_leaf(trees);

% Grow all trees
% [trees,ig_best] = growTrees(data_train,param);
% ig_best_cpr = zeros(50,1);
% for rho = 1:50
%     param.splitNum = rho;
%     [trees,ig_best] = growTrees(data_train,param);
%     ig_best_cpr(rho) = ig_best;
% end

% ig_best_vs_spfunc = zeros(4,1); %rho set to be 30
% for split = 1:4
%     param.split_func = split;
%      [trees,ig_best] = growTrees(data_train,param);
%     ig_best_vs_spfunc(split) = ig_best;
% end

%% Question 2 part 1 - four testing points
clear all; close all; clc;
init;
% Select dataset
[data_train, data_test] = getData('Toy_Spiral'); % {'Toy_Gaussian', 'Toy_Spiral', 'Toy_Circle', 'Caltech'}


param.num = 4;%10;         % Number of trees
param.depth = 5;        % trees depth
param.splitNum = 20;%3;     % Number of split functions to try
param.split = 'IG';     % Currently support 'information gain' only
param.split_func = 3;

%%%%%%%%%%%%%%%%%%%%%%
% Train Random Forest
[trees,ig_best] = growTrees(data_train,param);
test_point = [-.5 -.7 0; .4 .3 0; -.7 .4 0; .5 -.5 0];
for n=1:length(test_point)
    leaves = testTrees(test_point(n,:),trees,param);
    % disp(leaves);
    % average the class distributions of leaf nodes of all trees
    p_rf = trees(1).prob(leaves,:);
    p_rf_sum(:,n) = sum(p_rf)/length(trees);
    [~,test_point(n,3)] = max(p_rf_sum(:,n));
    
end



for n=1:length(data_test)
    leaves = testTrees(data_test(n,:),trees,param);
    % disp(leaves);
    % average the class distributions of leaf nodes of all trees
    p_rf = trees(1).prob(leaves,:);
    p_rf_sum(:,n) = sum(p_rf)/length(trees);
    [~,data_test(n,3)] = max(p_rf_sum(:,n));
    
end

figure
h_mesh = scatter(data_test(data_test(:,end)==1,1), data_test(data_test(:,end)==1,2), '.', 'MarkerEdgeColor', [.9 .5 .5]);%red
hold on; scatter(data_test(data_test(:,end)==2,1), data_test(data_test(:,end)==2,2), '.', 'MarkerEdgeColor', [.5 .9 .5]);%green
hold on; scatter(data_test(data_test(:,end)==3,1), data_test(data_test(:,end)==3,2), '.', 'MarkerEdgeColor', [.5 .5 .9]);
axis([-1.5 1.5 -1.5 1.5])
h_train = plot(data_train(data_train(:,end)==1,1), data_train(data_train(:,end)==1,2), 'o', 'MarkerFaceColor', [.9 .5 .5], 'MarkerEdgeColor','k');
hold on; %red
plot(data_train(data_train(:,end)==2,1), data_train(data_train(:,end)==2,2), 'o', 'MarkerFaceColor', [.5 .9 .5], 'MarkerEdgeColor','k');
hold on;%green
plot(data_train(data_train(:,end)==3,1), data_train(data_train(:,end)==3,2), 'o', 'MarkerFaceColor', [.5 .5 .9], 'MarkerEdgeColor','k');
axis([-1.5 1.5 -1.5 1.5]);%blue
hold on;

h_test = scatter(test_point(test_point(:,end)==1,1), test_point(test_point(:,end)==1,2), 'd', 'MarkerFaceColor', [.9 .5 .5], 'MarkerEdgeColor','k');%red
hold on; scatter(test_point(test_point(:,end)==2,1), test_point(test_point(:,end)==2,2), 'd', 'MarkerFaceColor', [.5 .9 .5], 'MarkerEdgeColor','k');%green
hold on; scatter(test_point(test_point(:,end)==3,1), test_point(test_point(:,end)==3,2), 'd', 'MarkerFaceColor', [.5 .5 .9], 'MarkerEdgeColor','k'); hold on
axis([-1.5 1.5 -1.5 1.5])
title('Four Test Points Classification Results')
legend([h_train,h_test,h_mesh],'Training Point','Test Point','Prediction Mesh');
%% Question 2 Evaluate the tree

%%%%%%%%%%%%%%%%%%%%
% Evaluate/Test Random Forest

clear all; close all; clc;
init;
[data_train, data_test] = getData('Toy_Spiral'); % {'Toy_Gaussian', 'Toy_Spiral', 'Toy_Circle', 'Caltech'}


% param.num = 250;%10;         % Number of trees
% param.depth = 5;        % trees depth
% param.splitNum = 20;%3;     % degree of randomness; Number of split functions trials to try
param.split = 'IG';     % Currently support 'information gain' only
param.split_func = 3;
% [trees,ig_best] = growTrees(data_train,param);

% grab the few data points and evaluate them one by one by the leant RF
%
% test_point = [-.5 -.7 0; .4 .3 0; -.7 .4 0; .5 -.5 0];
test_point = data_test(1:end,:);
p_rf_sum = zeros(3,length(test_point));
figure
counter = 0;
labels = zeros(length(test_point),15);
% for i = [100,200,300]
i = 100;
for j = [5 10 15]
    for k = [50 100 150 200 250]
        counter = counter+1;
        param.num = i;
        param.depth = j;
        param.splitNum = k;
        [trees,ig_best] = growTrees(data_train,param);
        
        for n=1:length(test_point)
            leaves = testTrees(test_point(n,:),trees,param);
            % disp(leaves);
            % average the class distributions of leaf nodes of all trees
            p_rf = trees(1).prob(leaves,:);
            p_rf_sum(:,n) = sum(p_rf)/length(trees);
            [~,test_point(n,3)] = max(p_rf_sum(:,n));
            
        end
        labels(:,counter) = test_point(:,3); %col1 is the label for i = 5 k = 50;
        subplot(4,5,counter)
        plot_toydata(data_train);hold on;
        scatter(test_point(test_point(:,end)==1,1), test_point(test_point(:,end)==1,2), '.', 'MarkerFaceColor', [.9 .5 .5]);%red
        hold on; scatter(test_point(test_point(:,end)==2,1), test_point(test_point(:,end)==2,2), '.', 'MarkerFaceColor', [.5 .9 .5]);%green
        hold on; scatter(test_point(test_point(:,end)==3,1), test_point(test_point(:,end)==3,2), '.', 'MarkerFaceColor', [.5 .5 .9]);
        axis([-1.5 1.5 -1.5 1.5])
        title([num2str(i) 'trees,', num2str(j) 'levels,', num2str(k) 'split trials' ])
        hold off
        
        %         end
    end
end
% Test on the dense 2D grid data, and visualise the results ...
% Change the RF parameter values and evaluate ...





%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%experiment with Caltech101 dataset for image categorisation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

init;

% Select dataset
% we do bag-of-words technique to convert images to vectors (histogram of codewords)
% Set 'showImg' in getData.m to 0 to stop displaying training and testing images and their feature vectors
[data_train, data_test] = getData('Caltech');
close all;



% Set the random forest parameters ...
% Train Random Forest ...
% Evaluate/Test Random Forest ...
% show accuracy and confusion matrix ...


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% random forest codebook for Caltech101 image categorisation
% .....



