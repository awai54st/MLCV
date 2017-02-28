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
    case 'Caltech' % Caltech dataset
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
        training_data = zeros(numBins,numClasses*sizeClasses);
        training_label = zeros(1,numClasses*sizeClasses);
        
        for idx_tr = 1:numClasses
            label = idx_tr;
            for idy_tr = 1:sizeClasses
                hist_tmp = knnsearch(cluster_centers',single(desc_tr{idx_tr,idy_tr})');
                
                training_data(:,idy_tr+(idx_tr-1)*sizeClasses) = histcounts(hist_tmp,numBins)';
                training_label(idy_tr+(idx_tr-1)*sizeClasses) = label;
                %hist_tr{idx_tr,idy_tr} = cluster_list; % Archive cluster lists
            end
        end

        data_train = [training_data;training_label]';
        
        % Clear unused varibles to save memory
        clearvars desc_tr desc_sel hist_tmp I training_data training_label
end


switch MODE
    case 'Caltech'
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
        test_data = zeros(numBins,numClasses*sizeClasses);
        test_label = zeros(1,numClasses*sizeClasses);
        
        for idx_te = 1:numClasses
            label = idx_te;
            for idy_te = 1:sizeClasses
                hist_tmp = knnsearch(cluster_centers',single(desc_te{idx_te,idy_te})');
                
                test_data(:,idy_te+(idx_te-1)*sizeClasses) = histcounts(hist_tmp,numBins)';
                test_label(idy_te+(idx_te-1)*sizeClasses) = label;
                %hist_tr{idx_tr,idy_tr} = cluster_list; % Archive cluster lists
            end
        end
        
        data_query = [test_data;test_label]';

        
        
        %     otherwise % Dense point for 2D toy data
        %         xrange = [-1.5 1.5];
        %         yrange = [-1.5 1.5];
        %         inc = 0.02;
        %         [x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
        %         data_query = [x(:) y(:) zeros(length(x)^2,1)];
        % end
end

