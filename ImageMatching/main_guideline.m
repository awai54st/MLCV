%% Import and show tsukuba images

clear all; close all; clc;

imageset = 0;
if imageset==0
    % Tsukuba imageset
    [ ~ , C{1} ] = readppm('tsukuba/scene1.row3.col1.ppm');
    [ ~ , C{2} ] = readppm('tsukuba/scene1.row3.col2.ppm');
    [ ~ , C{3} ] = readppm('tsukuba/scene1.row3.col3.ppm');
    [ ~ , C{4} ] = readppm('tsukuba/scene1.row3.col4.ppm');
    [ ~ , C{5} ] = readppm('tsukuba/scene1.row3.col5.ppm');
    groundTruthImage = imread('tsukuba/truedisp.row3.col3.pgm');

    image1 = C{1}(:,:,1);
    image2 = C{5}(:,:,1);
% HG imageset
elseif imageset==1 % original size
    image1 = rgb2gray(imread('HG/img1.JPG'));
    image2 = rgb2gray(imread('HG/img2.JPG'));
elseif imageset==2 % scaled size
    image1 = imresize(rgb2gray(imread('HG/img2.JPG')),0.125);
    image2 = imresize(rgb2gray(imread('HG/img3.JPG')),0.125);
    
% FD imageset
elseif imageset==3 % original size
    image1 = rgb2gray(imread('FD1/img1.JPG'));
    image2 = rgb2gray(imread('FD1/img2.JPG'));
elseif imageset==4 % scaled size
    image1 = imresize(rgb2gray(imread('FD1/img1.JPG')),0.125);
    image2 = imresize(rgb2gray(imread('FD1/img2.JPG')),0.125);
end
% figure;imshow(groundTruthImage);

% Design 1
% harrisFeatureDetector_customBlur(C{1}(:,:,1));

% Design 2
% corners = detectharrisFeatureFeatures(C{1}(:,:,1));
% 
% imshow(C{1}(:,:,1)); hold on;
% plot(corners.selectStrongest(150));

% https://github.com/gokhanozbulak/harrisFeature-Detector.git

feat_switch = 0;
if feat_switch==0
    % harrisFeature Feature Extraction
    features1_raw = harrisFeature(image1, 0.04, 20000);
    features2_raw = harrisFeature(image2, 0.04, 20000);
elseif feat_switch==1
    % Features by Clicking on Image
    [features1_raw,features2_raw] = clickImg(image1,image2);
end

% Descriptor Extraction
desc_switch = 1;
% Design 1: 11x11 Sampled Patches
if desc_switch==0
    descriptor1 = sampledPatchDesc(image1, features1_raw);
    descriptor2 = sampledPatchDesc(image2, features2_raw);
    sim_th = 1.5;
% Design 2: 11x11 Intensity Histograms
elseif desc_switch==1
    descriptor1 = histDesc(image1, features1_raw);
    descriptor2 = histDesc(image2, features2_raw);
    sim_th = 50;
end
% if size(descriptor1,1)>size(descriptor2,1)
%     [matches, scores] = knnsearch(descriptor2,descriptor1); % descriptor1(1:length(corr_idx)) and descriptor2(corr_idx) are NN
%     features2 = features2(:,matches);
% %     descriptor2 = descriptor2(corr_idx,:);
% else
%     matches = knnsearch(descriptor1,descriptor2); % descriptor1(1:length(corr_idx)) and descriptor2(corr_idx) are NN
%     features1 = features1(:,matches);
% %     descriptor1 = descriptor1(corr_idx,:);
% end
[matches, scores] = knnsearch(descriptor2,descriptor1); % descriptor1(1:length(corr_idx)) and descriptor2(corr_idx) are NN
matches = [1:size(descriptor1,1);matches'];

features1 = [];features2 = [];
feat_len = size(matches,2);
idx = 1;
for i = 1:feat_len
    if scores(i)<sim_th
        features1(:,idx) = features1_raw(1:2,matches(1,i));
        features2(:,idx) = features2_raw(1:2,matches(2,i));
        idx = idx + 1;
    end
end

% Messy fix: Correct the order of x and y coordinates
features1 = [features1(2,:);features1(1,:)];
features2 = [features2(2,:);features2(1,:)];

% Show matched points (matched features)
figure;
showMatchedFeatures(image1,image2,features1',features2','montage','PlotOptions',{'ro','go','y--'});
title('Putative point matches');

% Calculate Homography Matrix using SVD
H = homographySolve(features1, features2);

projection = homographyTransform(features1,H);

HA = mean(sqrt(sum((projection - features2).^2)));

% Calculate Fundamental Matrix
[F, inliers] = estimateFundamentalMatrix(features1',features2','NumTrials',4000);

% Epipolar Lines
figure;
subplot(121);
imshow(image1);
title('Features and Epipolar Lines in First Image'); hold on;
plot(features1(1,inliers),features1(2,inliers),'go');
epiLines = epipolarLine(F',features1(:,inliers)');
points = lineToBorderPoints(epiLines,size(image1));
line(points(:,[1,3])',points(:,[2,4])');

subplot(122);
imshow(image2);
title('Features and Epipolar Lines in Second Image'); hold on;
plot(features2(1,inliers),features2(2,inliers),'go');
epiLines = epipolarLine(F,features2(:,inliers)');
points = lineToBorderPoints(epiLines,size(image2));
line(points(:,[1,3])',points(:,[2,4])');

%% TB for homographySolve

clear all; close all; clc;

features1 = [[1;1],[1;2],[2;1],[2;2],[3;1]];
features2 = [[1;6],[1;7],[2;6],[2;7],[3;6]];
H = homographySolve(features1, features2);
projection = homographyTransform(features1,H);
HA = mean(sqrt(sum((projection - features2).^2)));

%% TB for fundamentalSolve

clear all; close all; clc;

% features1 = [[1;1],[1;2],[2;1],[2;2],[3;1]];
% features2 = [[1;6],[1;7],[2;6],[2;7],[3;6]];
features1 = randi(10,2,18);
features2 = features1+1;
n = size(features1,2);
[F1, ~] = fundamentalSolve(features1, features2);
F2 = estimateFundamentalMatrix(features1',features2','NumTrials',4000);
e1 = [features2;ones(1,n)]'*F1*[features1;ones(1,n)];
e2 = [features2;ones(1,n)]'*F2*[features1;ones(1,n)];

%% vl_feat SIFT

clear all; close all; clc;

init;

imageset = 4;
if imageset==0
    % Tsukuba imageset
    [ ~ , C{1} ] = readppm('tsukuba/scene1.row3.col1.ppm');
    [ ~ , C{2} ] = readppm('tsukuba/scene1.row3.col2.ppm');
    [ ~ , C{3} ] = readppm('tsukuba/scene1.row3.col3.ppm');
    [ ~ , C{4} ] = readppm('tsukuba/scene1.row3.col4.ppm');
    [ ~ , C{5} ] = readppm('tsukuba/scene1.row3.col5.ppm');
    groundTruthImage = imread('tsukuba/truedisp.row3.col3.pgm');

    image1 = C{1}(:,:,1);
    image2 = C{5}(:,:,1);
% HG imageset
elseif imageset==1 % original size
    image1 = rgb2gray(imread('HG/img1.JPG'));
    image2 = rgb2gray(imread('HG/img2.JPG'));
elseif imageset==2 % scaled size
    image1 = imresize(rgb2gray(imread('HG/img2.JPG')),0.5);
    image2 = imresize(rgb2gray(imread('HG/img3.JPG')),0.5);
    
% FD imageset
elseif imageset==3 % original size
    image1 = rgb2gray(imread('FD1/img1.JPG'));
    image2 = rgb2gray(imread('FD1/img2.JPG'));
elseif imageset==4 % scaled size
    image1 = imresize(rgb2gray(imread('FD1/img1.JPG')),0.5);
    image2 = imresize(rgb2gray(imread('FD1/img3.JPG')),0.5);
end

% vl_sift Descriptors
[frame1, descriptor1] = vl_sift(single(image1)) ;
[frame2, descriptor2] = vl_sift(single(image2)) ;

% Nearest Neighbour for Descriptor Matching
[matches, scores] = vl_ubcmatch(descriptor1, descriptor2) ;
features1_raw = frame1(1:2,:); features2_raw = frame2(1:2,:);

feat_len = size(matches,2);
idx = 1;
for i = 1:feat_len
    if scores(i)<8000
        features1(:,idx) = features1_raw(1:2,matches(1,i));
        features2(:,idx) = features2_raw(1:2,matches(2,i));
        idx = idx + 1;
    end
end

% Show matched points (matched features)
figure;
showMatchedFeatures(image1,image2,features1',features2','montage','PlotOptions',{'ro','go','y--'});
title('Putative point matches');

% Calculate Homography Matrix using SVD
H = homographySolve(features1, features2);
projection = homographyTransform(features1,H);

% Calculate Homography Accuracy as Average Euclidean Projection Error
HA = mean(sqrt(sum((projection - features2).^2)));

% Calculate Fundamental Matrix
[F, inliers] = estimateFundamentalMatrix(features1',features2','NumTrials',4000);

% Epipolar Lines
figure;
subplot(121);
imshow(image1);
title('Features and Epipolar Lines in First Image'); hold on;
plot(features1(1,inliers),features1(2,inliers),'go');
epiLines = epipolarLine(F',features1(:,inliers)');
points = lineToBorderPoints(epiLines,size(image1));
line(points(:,[1,3])',points(:,[2,4])');

subplot(122);
imshow(image2);
title('Features and Epipolar Lines in Second Image'); hold on;
plot(features2(1,inliers),features2(2,inliers),'go');
epiLines = epipolarLine(F,features2(:,inliers)');
points = lineToBorderPoints(epiLines,size(image2));
line(points(:,[1,3])',points(:,[2,4])');