function [ data_train, data_query ] = getData( MODE )
% Generate training and testing data

% Data Options:
%   4. Caltech 101
data_train = [];
data_query = [];
showImg = 1; % Show training & testing images and their image feature vector (histogram representation)

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
                   
%                 bag = bagOfFeatures(I,'Verbose',false);
            end     
        end
        
        disp('Building visual codebook...')
        % Build visual vocabulary (codebook) for 'Bag-of-Words method'
        desc_sel = single(vl_colsubset(cat(2,desc_tr{:}), 10e4)); % Randomly select 100k SIFT descriptors for clustering
        %desc_sel is a 128x100k (nxp data) (observation x variables)
        
        % K-means clustering
        numBins = 256; % for instance,
        
        %Training data: vectors are histograms, one from each training image
        % write your own codes here
        % ...
        cluster_centers = kmeans(desc_sel,numBins);

        
        disp('Encoding Images...')
        % Vector Quantisation
        
        % write your own codes here
        % ...
        desc_tr = single(vl_colsubset(cat(2,desc_tr{:}), 10e4));

        
        
        % Clear unused varibles to save memory
        clearvars desc_tr desc_sel
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
                 temp= single(vl_colsubset(cat(2,desc_te{c,i}), 2000)); % Randomly select 100k SIFT descriptors for clustering
        
                
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
        
        
        

        
        
        %     otherwise % Dense point for 2D toy data
        %         xrange = [-1.5 1.5];
        %         yrange = [-1.5 1.5];
        %         inc = 0.02;
        %         [x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
        %         data_query = [x(:) y(:) zeros(length(x)^2,1)];
        % end
end

