%% Question 2 Image Geometry

%% 1.Homography(using images HG)
% http://www.vlfeat.org/applications/sift-mosaic-code.html
% https://people.cs.umass.edu/~elm/Teaching/ppt/370/370_10_RANSAC.pptx.pdf
%% (a)
% Find interest points in one image by using method from Q1.1 or Q1.2.a. Reduce the
% size of the image by a factor of 2 and run detector again. Compare the interest points
% obtained in these two cases using HA error.

clear all; close all; clc;

init;

% HG imageset
image1 = rgb2gray(imread('HG/img1.JPG')); % original image
image2 = imresize(rgb2gray(imread('HG/img1.JPG')),0.5); % scaled image

% vl_sift Descriptors
[frame1, descriptor1] = vl_sift(single(image1)) ;
[frame2, descriptor2] = vl_sift(single(image2)) ;

% Nearest Neighbour for Descriptor Matching
[matches, scores] = vl_ubcmatch(descriptor1, descriptor2, 2.5) ;
features1_raw = frame1(1:2,:); features2_raw = frame2(1:2,:);
features1 = features1_raw(:,matches(1,:));
features2 = features2_raw(:,matches(2,:));

% Show matched points (matched features)
figure;
showMatchedFeatures(image1,image2,features1',features2','montage','PlotOptions',{'ro','go','y--'});
title('Putative point matches');

% RANSAC to Filter out Outliers
numMatches = size(matches,2);
for iter = 1:100
    % estimate homography
    subset = vl_colsubset(1:numMatches, 4) ;
    feat1_tmp = features1(:,subset);
    feat2_tmp = features2(:,subset);
    
    % Calculate Homography Matrix using SVD
    H_tmp{iter} = homography_solve(feat1_tmp, feat2_tmp);
    
    % Calculate Homography Accuracy as Average Euclidean Projection Error
    projection = homography_transform(features1,H_tmp{iter});
    HA_tmp{iter} = mean(sqrt(sum((projection - features2).^2)));
    inlier_tmp{iter} = sqrt(sum((projection - features2).^2)) < 8*8;
end
[score, best_idx] = min(cell2mat(HA_tmp)) ;
inlier = inlier_tmp{best_idx};

% Evaluate final Homography Matrix
H = homography_solve(features1(:,inlier), features2(:,inlier));
projection = homography_transform(features1(:,inlier),H);
HA = mean(sqrt(sum((projection - features2(:,inlier)).^2)));

%% (b)
% Using method from Q1.3.a estimate homography from the manually established
% correspondences with Q1.1 and compare to the homography from the list of
% correspondences obtained automatically with Q1.2. Use function from Q1.3.c to
% visualise and validate your homographies. Analyse and compare geometric
% transformation parameters that can be derived from the two homographies.

clear all; close all; clc;

init;

% HG imageset
image1 = imresize(rgb2gray(imread('HG/img1.JPG')),0.5);
image2 = imresize(rgb2gray(imread('HG/img2.JPG')),0.5);

% Part1: Manual Features
[features1,features2] = clickImg(image1,image2);

% Show matched points (matched features)
figure;
showMatchedFeatures(image1,image2,features1,features2,'montage','PlotOptions',{'ro','go','y--'});
title('Putative point matches');

% Calculate Homography Matrix using SVD
H_manual = homography_solve(features1', features2');

% Calculate Homography Accuracy as Average Euclidean Projection Error
projection = homography_transform(features1',H_manual);
HA_manual = mean(sqrt(sum((projection' - features2).^2)));

% Part2: Automatic Features
[frame1, descriptor1] = vl_sift(single(image1)) ;
[frame2, descriptor2] = vl_sift(single(image2)) ;

% Nearest Neighbour for Descriptor Matching
[matches, scores] = vl_ubcmatch(descriptor1, descriptor2, 2.5) ;
features1_raw = frame1(1:2,:); features2_raw = frame2(1:2,:);
features1 = features1_raw(:,matches(1,:));
features2 = features2_raw(:,matches(2,:));

% RANSAC to Filter out Outliers
numMatches = size(matches,2);
for iter = 1:100
    % estimate homography
    subset = vl_colsubset(1:numMatches, 4) ;
    feat1_tmp = features1(:,subset);
    feat2_tmp = features2(:,subset);
    
    % Calculate Homography Matrix using SVD
    H_tmp{iter} = homography_solve(feat1_tmp, feat2_tmp);
    
    % Calculate Homography Accuracy as Average Euclidean Projection Error
    projection = homography_transform(features1,H_tmp{iter});
    HA_tmp{iter} = mean(sqrt(sum((projection - features2).^2)));
    inlier_tmp{iter} = sqrt(sum((projection - features2).^2)) < 8*8;
end
[score, best_idx] = min(cell2mat(HA_tmp)) ;
inlier = inlier_tmp{best_idx};

% Show matched points (matched features)
figure;
showMatchedFeatures(image1,image2,features1(:,inlier)',features2(:,inlier)','montage','PlotOptions',{'ro','go','y--'});
hold on
hx = scatter(features1(1,~inlier)',features1(2,~inlier)','rx');hold off
title('Putative point matches');

% Evaluate final Homography Matrix
H_auto = homography_solve(features1(:,inlier), features2(:,inlier));
projection = homography_transform(features1(:,inlier),H_auto);
HA_auto = mean(sqrt(sum((projection - features2(:,inlier)).^2)));

%% (c)
% Estimate homography from different number of correspondences from Q1.2 starting
% from the minimum number up to the maximum number of available pairs. Report and
% discuss HA for different number of correspondences. Find the number of outliers in
% your list of automatic correspondences and explain your approach to that.

clear all; close all; clc;

init;

% HG imageset
image1 = imresize(rgb2gray(imread('HG/img1.JPG')),0.5);
image2 = imresize(rgb2gray(imread('HG/img2.JPG')),0.5);

% vl_sift Descriptors
[frame1, descriptor1] = vl_sift(single(image1)) ;
[frame2, descriptor2] = vl_sift(single(image2)) ;

% Nearest Neighbour for Descriptor Matching
[matches, scores] = vl_ubcmatch(descriptor1, descriptor2, 2.5) ;
features1_raw = frame1(1:2,:); features2_raw = frame2(1:2,:);
features1 = features1_raw(:,matches(1,:));
features2 = features2_raw(:,matches(2,:));

% RANSAC to Filter out Outliers
numMatches = size(matches,2);
for iter = 1:100
    % estimate homography
    subset = vl_colsubset(1:numMatches, 4) ;
    feat1_tmp = features1(:,subset);
    feat2_tmp = features2(:,subset);
    
    % Calculate Homography Matrix using SVD
    H_tmp{iter} = homography_solve(feat1_tmp, feat2_tmp);
    
    % Calculate Homography Accuracy as Average Euclidean Projection Error
    projection = homography_transform(features1,H_tmp{iter});
    HA_tmp{iter} = mean(sqrt(sum((projection - features2).^2)));
    inlier_tmp{iter} = sqrt(sum((projection - features2).^2)) < 8*8;
end
[score, best_idx] = min(cell2mat(HA_tmp)) ;
inlier = inlier_tmp{best_idx};

% Show matched points (matched features)
figure;
showMatchedFeatures(image1,image2,features1(:,inlier)',features2(:,inlier)','montage','PlotOptions',{'ro','go','y--'});
hold on
hx = scatter(features1(1,~inlier)',features1(2,~inlier)','rx');hold off
title('Putative point matches');

% Evaluate final Homography Matrix
H = homography_solve(features1(:,inlier), features2(:,inlier));
projection = homography_transform(features1(:,inlier),H);
HA = mean(sqrt(sum((projection - features2(:,inlier)).^2)));

