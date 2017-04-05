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
    H_tmp{iter} = homographySolve(feat1_tmp, feat2_tmp);
    
    % Calculate Homography Accuracy as Average Euclidean Projection Error
    projection = homographyTransform(features1,H_tmp{iter});
    HA_tmp{iter} = mean(sqrt(sum((projection - features2).^2)));
    inlier_tmp{iter} = sqrt(sum((projection - features2).^2)) < 8*8;
end
[score, best_idx] = min(cell2mat(HA_tmp)) ;
inlier = inlier_tmp{best_idx};

% Evaluate final Homography Matrix
H = homographySolve(features1(:,inlier), features2(:,inlier));
projection = homographyTransform(features1(:,inlier),H);
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
H_manual = homographySolve(features1', features2');

% Calculate Homography Accuracy as Average Euclidean Projection Error
projection = homographyTransform(features1',H_manual);
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
    H_tmp{iter} = homographySolve(feat1_tmp, feat2_tmp);
    
    % Calculate Homography Accuracy as Average Euclidean Projection Error
    projection = homographyTransform(features1,H_tmp{iter});
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
H_auto = homographySolve(features1(:,inlier), features2(:,inlier));
projection = homographyTransform(features1(:,inlier),H_auto);
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
    H_tmp{iter} = homographySolve(feat1_tmp, feat2_tmp);
    
    % Calculate Homography Accuracy as Average Euclidean Projection Error
    projection = homographyTransform(features1,H_tmp{iter});
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
H = homographySolve(features1(:,inlier), features2(:,inlier));
projection = homographyTransform(features1(:,inlier),H);
HA = mean(sqrt(sum((projection - features2(:,inlier)).^2)));

%% 2.Image Geometry

%% (a) & (b)
% Estimate fundamental matrix using list of correspondences from Q1.1 or Q1.2.a.
% Calculate the epipoles for images A and B. Show epipolar lines and epipoles on the
% images if possible.

clear all; close all; clc;

init;

% FD1 imageset
image1 = imresize(rgb2gray(imread('FD1/img1.JPG')),0.125);
image2 = imresize(rgb2gray(imread('FD1/img2.JPG')),0.125);
% image1 = histeq(image1);
% image2 = histeq(image2);


% [ ~ , C{1} ] = readppm('tsukuba/scene1.row3.col1.ppm');
% [ ~ , C{2} ] = readppm('tsukuba/scene1.row3.col2.ppm');
% [ ~ , C{3} ] = readppm('tsukuba/scene1.row3.col3.ppm');
% [ ~ , C{4} ] = readppm('tsukuba/scene1.row3.col4.ppm');
% [ ~ , C{5} ] = readppm('tsukuba/scene1.row3.col5.ppm');
% groundTruthImage = imread('tsukuba/truedisp.row3.col3.pgm');
% 
% image1 = C{1}(:,:,1);
% image2 = C{5}(:,:,1);

% vl_sift Descriptors
[frame1, descriptor1] = vl_sift(single(image1)) ;
[frame2, descriptor2] = vl_sift(single(image2)) ;

% Nearest Neighbour for Descriptor Matching
[matches, scores] = vl_ubcmatch(descriptor1, descriptor2, 1.5) ;
features1_raw = frame1(1:2,:); features2_raw = frame2(1:2,:);
features1 = features1_raw(:,matches(1,:));
features2 = features2_raw(:,matches(2,:));

% Calculate Fundamental Matrix
% [F, inliers] = estimateFundamentalMatrix(features1',features2','NumTrials',4000);

% RANSAC to Filter out Outliers
numMatches = size(matches,2);
for iter = 1:4000
    % estimate homography
    subset = vl_colsubset(1:numMatches, 8) ;
    feat1_tmp = features1(:,subset);
    feat2_tmp = features2(:,subset);
    
    % Calculate Homography Matrix using SVD
    [F_tmp{iter}, e_tmp{iter}] = fundamentalSolve(feat1_tmp, feat2_tmp);
    
    % Calculate Homography Accuracy as Average Euclidean Projection Error
%     FA{iter} = sum(abs(dot((F_tmp{iter} * [features2;ones(1,numMatches)]), [features1;ones(1,numMatches)])));
%     inlier_tmp{iter} = abs(dot((F_tmp{iter} * [features2;ones(1,numMatches)]), [features1;ones(1,numMatches)])) < FA{iter};
    
    epipolar_lines = fundamentalTransform(features1,F_tmp{iter});
    FA_tmp{iter} = abs(dot( [features2; ones(1, numMatches)], epipolar_lines ));
    inlier_tmp{iter} = FA_tmp{iter} < 0.001;
    FA_tmp{iter} = mean(FA_tmp{iter});
end
% [score, best_idx] = min(cell2mat(e_tmp)) ;
[score, best_idx] = min(cell2mat(FA_tmp)) ;
inliers = inlier_tmp{best_idx};
% [F, ~] = fundamentalSolve(features1(:,inliers), features2(:,inliers));
F = F_tmp{best_idx};
% [F, inliers] = estimateFundamentalMatrix(features1',features2','NumTrials',4000);

% Epipolar Lines
figure;
subplot(121);
imshow(image1);
title('Features and Epipolar Lines in First Image'); hold on;
plot(features1(1,inliers),features1(2,inliers),'go');
epiLines = fundamentalTransform(features1(:,inliers),F);
% epiLines = epipolarLine(F',features1(:,inliers)');
points = lineToBorderPoints(epiLines',size(image1));
line(points(:,[1,3])',points(:,[2,4])');

subplot(122);
imshow(image2);
title('Features and Epipolar Lines in Second Image'); hold on;
plot(features2(1,inliers),features2(2,inliers),'go');
epiLines = fundamentalTransform(features1(:,inliers),F);
points = lineToBorderPoints(epiLines',size(image2));
line(points(:,[1,3])',points(:,[2,4])');

%% (c)
% Calculate disparity map between images A and B.

disparityRange = [0 96];
disparityMap = disparity(image1,image2,'BlockSize',...
    15,'DisparityRange',disparityRange);
figure;
imshow(disparityMap,disparityRange);
title('Disparity Map');
% colormap jet
colorbar

% https://github.com/owlbread/MATLAB-stereo-image-disparity-map/blob/master/disparitymap.m


