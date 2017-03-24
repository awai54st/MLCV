%% Import and show tsukuba images

clear all; close all; clc;

[ ~ , C{1} ] = readppm('tsukuba/scene1.row3.col1.ppm');
[ ~ , C{2} ] = readppm('tsukuba/scene1.row3.col2.ppm');
[ ~ , C{3} ] = readppm('tsukuba/scene1.row3.col3.ppm');
[ ~ , C{4} ] = readppm('tsukuba/scene1.row3.col4.ppm');
[ ~ , C{5} ] = readppm('tsukuba/scene1.row3.col5.ppm');
groundTruthImage = imread('tsukuba/truedisp.row3.col3.pgm');
figure;imshow(C{1}(:,:,1));
% figure;imshow(C{2}(:,:,1));
% figure;imshow(C{3}(:,:,1));
% figure;imshow(C{4}(:,:,1));
figure;imshow(C{5}(:,:,1));
% figure;imshow(groundTruthImage);

% Design 1
% harrisDetector_customBlur(C{1}(:,:,1));

% Design 2
% corners = detectHarrisFeatures(C{1}(:,:,1));
% 
% imshow(C{1}(:,:,1)); hold on;
% plot(corners.selectStrongest(150));

% https://github.com/gokhanozbulak/Harris-Detector.git

image1 = C{1}(:,:,1);
image2 = C{5}(:,:,1);

% Harris Feature Extraction
features1_raw = harris(image1, 0.04, 20000);
features2_raw = harris(image2, 0.04, 20000);

% Descriptor Extraction
desc_switch = 1;
% Design 1: 11x11 Sampled Patches
if desc_switch==0
    descriptor1 = sampledPatchDesc(image1, features1_raw);
    descriptor2 = sampledPatchDesc(image2, features2_raw);
elseif desc_switch==1
    descriptor1 = histDesc(image1, features1_raw);
    descriptor2 = histDesc(image2, features2_raw);
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
    if scores(i)<40
        features1(:,idx) = features1_raw(1:2,matches(1,i));
        features2(:,idx) = features2_raw(1:2,matches(2,i));
        idx = idx + 1;
    end
end
% Calculate Homography Matrix using SVD
H = homography_solve(features1, features2);

projection = homography_transform(features1,H);

HA = mean(sqrt(sum((projection - features2).^2)));

%% TB for homography_solve

clear all; close all; clc;

features1 = [[1;1],[1;2],[2;1],[2;2],[3;1]];
features2 = [[1;6],[1;7],[2;6],[2;7],[3;6]];
H = homography_solve(features1, features2);
projection = homography_transform(features1,H);
HA = mean(sqrt(sum((projection - features2).^2)));

%% vl_feat SIFT

clear all; close all; clc;

init;

[ ~ , C{1} ] = readppm('tsukuba/scene1.row3.col1.ppm');
[ ~ , C{2} ] = readppm('tsukuba/scene1.row3.col2.ppm');
[ ~ , C{3} ] = readppm('tsukuba/scene1.row3.col3.ppm');
[ ~ , C{4} ] = readppm('tsukuba/scene1.row3.col4.ppm');
[ ~ , C{5} ] = readppm('tsukuba/scene1.row3.col5.ppm');
groundTruthImage = imread('tsukuba/truedisp.row3.col3.pgm');

image1 = C{1}(:,:,1);
image2 = C{5}(:,:,1);

% PHOW_Sizes = [4 8 10]; % Multi-resolution, these values determine the scale of each layer.
% PHOW_Step = 8; % The lower the denser. Select from {2,4,8,16}
% [features1, descriptor1] = vl_phow(single(image1),'Sizes',PHOW_Sizes,'Step',PHOW_Step); %  extracts PHOW features (multi-scaled Dense SIFT)
% [features2, descriptor2] = vl_phow(single(image2),'Sizes',PHOW_Sizes,'Step',PHOW_Step); %  extracts PHOW features (multi-scaled Dense SIFT)
% descriptor1 = descriptor1';
% descriptor2 = descriptor2';

[frame1, descriptor1] = vl_sift(single(image1)) ;
[frame2, descriptor2] = vl_sift(single(image2)) ;
[matches, scores] = vl_ubcmatch(descriptor1, descriptor2) ;
features1_raw = frame1(1:2,:); features2_raw = frame2(1:2,:);

feat_len = size(matches,2);
% features1 = zeros(2,feat_len); features2 = zeros(2,feat_len);
idx = 1;
for i = 1:feat_len
    if scores(i)<10000
        features1(:,idx) = features1_raw(1:2,matches(1,i));
        features2(:,idx) = features2_raw(1:2,matches(2,i));
        idx = idx + 1;
    end
end

% Calculate Homography Matrix using SVD
H = homography_solve(features1, features2);

projection = homography_transform(features1,H);

HA = mean(sqrt(sum((projection - features2).^2)));