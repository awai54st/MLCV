function [ data_train, data_query ] = getData( MODE )
% Generate training and testing data

% Data Options:
%   4. Caltech 101
data_train = [];
data_query = [];
showImg = 0; % Show training & testing images and their image feature vector (histogram representation)

PHOW_Sizes = [4 8 10]; % Multi-resolution, these values determine the scale of each layer.
PHOW_Step = 8; % The lower the denser. Select from {2,4,8,16}

switch MODE
    case 'Caltech_kmeans' % Caltech dataset
        close all;
        imgSel = [15 15]; % randomly select 15 images each class without replacement. (For both training & testing)
        folderName = './Caltech_101/101_ObjectCategories';
        classList = dir(folderName);
        classList = {classList(3:end).name} % 10 classes
        
        disp('Loading training images...')
        % Load Images -> Description (Dense SIFT)
        cnt = 1;
        if showImg
            figure('Units','normalized','Position',[.05 .1 .4 .9]);
            suptitle('Training image samples');
        end
        for c = 1:length(classList) %1 to 10 classes
            subFolderName = fullfile(folderName,classList{c});
            imgList = dir(fullfile(subFolderName,'*.jpg'));
            imgIdx{c} = randperm(length(imgList));
            imgIdx_tr = imgIdx{c}(1:imgSel(1));
            imgIdx_te = imgIdx{c}(imgSel(1)+1:sum(imgSel));
            
            for i = 1:length(imgIdx_tr)
                I = imread(fullfile(subFolderName,imgList(imgIdx_tr(i)).name));
                
                % Visualise
                if i < 6 & showImg %five columns of picture
                    subaxis(length(classList),5,cnt,'SpacingVert',0,'MR',0);
                    imshow(I);
                    cnt = cnt+1;
                    drawnow;
                end
                
                if size(I,3) == 3
                    I = rgb2gray(I); % PHOW work on gray scale image
                end
                
                % For details of image description, see http://www.vlfeat.org/matlab/vl_phow.html
                [~, desc_tr{c,i}] = vl_phow(single(I),'Sizes',PHOW_Sizes,'Step',PHOW_Step); %  extracts PHOW features (multi-scaled Dense SIFT)
                
            end     
        end
        
        disp('Building visual codebook...')
        % Build visual vocabulary (codebook) for 'Bag-of-Words method'
        desc_sel = single(vl_colsubset(cat(2,desc_tr{:}), 10e4)); % Randomly select 100k SIFT descriptors for clustering
        %desc_sel is a 128x100k (nxp data) (observation x variables)
        
        % K-means clustering
        numBins = 256; % for instance,
        cluster_centers = kmeans(desc_sel,numBins);
        
        %Training data: vectors are histograms, one from each training image
        % write your own codes here
        % ...
        

        %histogram(hist_tr{1,1},numBins);
        
        disp('Encoding Images...')
        % Vector Quantisation
        
        % write your own codes here
        % ...
        numClasses = length(classList);
        sizeClasses = length(imgIdx_tr);
        training_data = zeros(numClasses*sizeClasses,numBins);
        training_label = zeros(numClasses*sizeClasses,1);
        
        for idx_tr = 1:numClasses
            label = idx_tr;
            for idy_tr = 1:sizeClasses
                %hist_tmp = knnsearch(cluster_centers',single(desc_tr{idx_tr,idy_tr})');
                [~,hist_tmp] = min(vl_alldist(single(desc_tr{idx_tr,idy_tr}), cluster_centers),[],2) ;
                training_data(idy_tr+(idx_tr-1)*sizeClasses,:) = histcounts(hist_tmp,numBins);
                training_label(idy_tr+(idx_tr-1)*sizeClasses) = label;
                %hist_tr{idx_tr,idy_tr} = cluster_list; % Archive cluster lists
            end
        end

        data_train = [training_data,training_label];
        
        % Clear unused varibles to save memory
        clearvars desc_tr desc_sel hist_tmp I training_data training_label
        
    case 'Caltech_RFCB'
        
        close all;
        imgSel = [15 15]; % randomly select 15 images each class without replacement. (For both training & testing)
        folderName = './Caltech_101/101_ObjectCategories';
        classList = dir(folderName);
        classList = {classList(3:end).name} % 10 classes
        
        disp('Loading training images...')
        % Load Images -> Description (Dense SIFT)
        cnt = 1;
        if showImg
            figure('Units','normalized','Position',[.05 .1 .4 .9]);
            suptitle('Training image samples');
        end
        for c = 1:length(classList) %1 to 10 classes
            subFolderName = fullfile(folderName,classList{c});
            imgList = dir(fullfile(subFolderName,'*.jpg'));
            imgIdx{c} = randperm(length(imgList));
            imgIdx_tr = imgIdx{c}(1:imgSel(1));
            imgIdx_te = imgIdx{c}(imgSel(1)+1:sum(imgSel));
            
            for i = 1:length(imgIdx_tr)
                I = imread(fullfile(subFolderName,imgList(imgIdx_tr(i)).name));
                
                % Visualise
                if i < 6 & showImg %five columns of picture
                    subaxis(length(classList),5,cnt,'SpacingVert',0,'MR',0);
                    imshow(I);
                    cnt = cnt+1;
                    drawnow;
                end
                
                if size(I,3) == 3
                    I = rgb2gray(I); % PHOW work on gray scale image
                end
                
                % For details of image description, see http://www.vlfeat.org/matlab/vl_phow.html
                [~, desc_tr{c,i}] = vl_phow(single(I),'Sizes',PHOW_Sizes,'Step',PHOW_Step); %  extracts PHOW features (multi-scaled Dense SIFT)
                
            end     
        end
        
        disp('Building visual codebook...')
        % Build visual vocabulary (codebook) for 'Bag-of-Words method'
        
%         [desc_sel,desc_idx] = vl_colsubset(cat(2,desc_tr{:}), 10e4);
        
        [numClass,sizeClass] = size(desc_tr);
        training_desc = [];
        for idx = 1:numClass
            for idy = 1:sizeClass
                desc_size = size(desc_tr{idx,idy},2);
                training_desc_label = repmat(idy+(idx-1)*sizeClass,1,666);
                desc_tmp = single(vl_colsubset(desc_tr{idx,idy}, 666));
%                 desc_tmp = [desc_tmp;training_desc_label];
                training_desc = [training_desc,[desc_tmp;training_desc_label]];
            end
        end
        
        % Set the random forest parameters ...
        param_codebook.num = 20;%10; % Number of trees
        param_codebook.depth = 5; % trees depth
        param_codebook.splitNum = 50;%3; % Number of split functions to try
        param_codebook.split = 'IG'; % Currently support 'information gain' only
        param_codebook.split_func = 1;
        
        % Train Random Forest ...
        training_desc = training_desc';
        [tree_codebook,~] = growTrees(training_desc,param_codebook);
        
        disp('Encoding Images...')
        % Vector Quantisation
        
        % write your own codes here
        % ...
        numBins = size(tree_codebook(1).prob,1);

        training_data = zeros(numClass*sizeClass,numBins);
        training_label = zeros(numClass*sizeClass,1);
        
        for idx_tr = 1:numClass
            label = idx_tr;
            for idy_tr = 1:sizeClass
                desc_tmp = [single(desc_tr{idx_tr,idy_tr}'),ones(size(desc_tr{idx_tr,idy_tr},2),1)];
                k = randperm(size(desc_tr{idx_tr,idy_tr},2));
                desc_tmp = desc_tmp(k(1:1000),:);
                leaves = testTrees(desc_tmp,tree_codebook,param_codebook);
                training_data(idy_tr+(idx_tr-1)*sizeClass,:) = histcounts(leaves(:),numBins);
                training_label(idy_tr+(idx_tr-1)*sizeClass) = label;
                %hist_tr{idx_tr,idy_tr} = cluster_list; % Archive cluster lists
            end
        end
        
        data_train = [training_data,training_label];
        
        % Clear unused varibles to save memory
        clearvars desc_tr training_desc training_data training_label
        
end


switch MODE
    case 'Caltech_kmeans'
        if showImg
            figure('Units','normalized','Position',[.05 .1 .4 .9]);
            suptitle('Testing image samples');
        end
        disp('Processing testing images...');
        cnt = 1;
        % Load Images -> Description (Dense SIFT)
        for c = 1:length(classList)
            subFolderName = fullfile(folderName,classList{c});
            imgList = dir(fullfile(subFolderName,'*.jpg'));
            imgIdx_te = imgIdx{c}(imgSel(1)+1:sum(imgSel));
            
            for i = 1:length(imgIdx_te)
                I = imread(fullfile(subFolderName,imgList(imgIdx_te(i)).name));
                
                % Visualise
                if i < 6 & showImg
                    subaxis(length(classList),5,cnt,'SpacingVert',0,'MR',0);
                    imshow(I);
                    cnt = cnt+1;
                    drawnow;
                end
                
                if size(I,3) == 3
                    I = rgb2gray(I);
                end
                [~, desc_te{c,i}] = vl_phow(single(I),'Sizes',PHOW_Sizes,'Step',PHOW_Step);
                 %desc_sel1{c,i} 
                 %temp= single(vl_colsubset(cat(2,desc_te{c,i}), 2000)); % Randomly select 100k SIFT descriptors for clustering
        
                
            end
        end
        %        suptitle('Testing image samples');
        if showImg
            figure('Units','normalized','Position',[.5 .1 .4 .9]);
            suptitle('Testing image representations: 256-D histograms');
        end
        
        % Quantisation
        
        % write your own codes here
        % ...
        numClasses = length(classList);
        sizeClasses = length(imgIdx_te);
        test_data = zeros(numClasses*sizeClasses,numBins);
        test_label = zeros(numClasses*sizeClasses,1);
        
        for idx_te = 1:numClasses
            label = idx_te;
            for idy_te = 1:sizeClasses
                %hist_tmp = knnsearch(cluster_centers',single(desc_te{idx_te,idy_te})');
                [~,hist_tmp] = min(vl_alldist(single(desc_te{idx_te,idy_te}), cluster_centers),[],2) ;
                test_data(idy_te+(idx_te-1)*sizeClasses,:) = histcounts(hist_tmp,numBins);
                test_label(idy_te+(idx_te-1)*sizeClasses) = label;
                %hist_tr{idx_tr,idy_tr} = cluster_list; % Archive cluster lists
            end
        end
        
        data_query = [test_data,test_label];

        
        
    case 'Caltech_RFCB'
        
        if showImg
            figure('Units','normalized','Position',[.05 .1 .4 .9]);
            suptitle('Testing image samples');
        end
        disp('Processing testing images...');
        cnt = 1;
        % Load Images -> Description (Dense SIFT)
        for c = 1:length(classList)
            subFolderName = fullfile(folderName,classList{c});
            imgList = dir(fullfile(subFolderName,'*.jpg'));
            imgIdx_te = imgIdx{c}(imgSel(1)+1:sum(imgSel));
            
            for i = 1:length(imgIdx_te)
                I = imread(fullfile(subFolderName,imgList(imgIdx_te(i)).name));
                
                % Visualise
                if i < 6 & showImg
                    subaxis(length(classList),5,cnt,'SpacingVert',0,'MR',0);
                    imshow(I);
                    cnt = cnt+1;
                    drawnow;
                end
                
                if size(I,3) == 3
                    I = rgb2gray(I);
                end
                [~, desc_te{c,i}] = vl_phow(single(I),'Sizes',PHOW_Sizes,'Step',PHOW_Step);
                 %desc_sel1{c,i} 
                 %temp= single(vl_colsubset(cat(2,desc_te{c,i}), 2000)); % Randomly select 100k SIFT descriptors for clustering
        
                
            end
        end
        %        suptitle('Testing image samples');
        if showImg
            figure('Units','normalized','Position',[.5 .1 .4 .9]);
            suptitle('Testing image representations: 256-D histograms');
        end
        
        % Quantisation
        
        % write your own codes here
        % ...
        [numClass,sizeClass] = size(desc_te);
        numBins = size(tree_codebook(1).prob,1);

        test_data = zeros(numClass*sizeClass,numBins);
        test_label = zeros(numClass*sizeClass,1);
        
        for idx_te = 1:numClass
            label = idx_te;
            for idy_te = 1:sizeClass
                desc_tmp = [single(desc_te{idx_te,idy_te}'),ones(size(desc_te{idx_te,idy_te},2),1)];
                k = randperm(size(desc_te{idx_te,idy_te},2));
                desc_tmp = desc_tmp(k(1:1000),:);
                
%                 desc_tmp = [single(desc_te{idx_te,idy_te}'),ones(size(desc_te{idx_te,idy_te},2),1)];
                leaves = testTrees(desc_tmp,tree_codebook,param_codebook);
                test_data(idy_te+(idx_te-1)*sizeClass,:) = histcounts(leaves(:),numBins);
                test_label(idy_te+(idx_te-1)*sizeClass) = label;
                %hist_tr{idx_tr,idy_tr} = cluster_list; % Archive cluster lists
            end
        end
        
        data_query = [test_data,test_label];
            
end

