
sigma = 2;
g = fspecial('gaussian',max(1,fix(6*sigma)), sigma); %%%%%% Gaussien Filter
given_filter = [0.03 0.105 0.222 0.286 0.222 0.105 0.03];
g1 = given_filter'*given_filter;

%% Import and show tsukuba images

clear all; close all; clc;

[ ~ , C{1} ] = readppm('tsukuba/scene1.row3.col1.ppm');
[ ~ , C{2} ] = readppm('tsukuba/scene1.row3.col2.ppm');
[ ~ , C{3} ] = readppm('tsukuba/scene1.row3.col3.ppm');
[ ~ , C{4} ] = readppm('tsukuba/scene1.row3.col4.ppm');
[ ~ , C{5} ] = readppm('tsukuba/scene1.row3.col5.ppm');
groundTruthImage = imread('tsukuba/truedisp.row3.col3.pgm');
% figure;imshow(C{1}(:,:,1));
% figure;imshow(C{2}(:,:,1));
% figure;imshow(C{3}(:,:,1));
% figure;imshow(C{4}(:,:,1));
% figure;imshow(C{5}(:,:,1));
% figure;imshow(groundTruthImage);

% Design 1
% harrisDetector_customBlur(C{1}(:,:,1));

% Design 2
% corners = detectHarrisFeatures(C{1}(:,:,1));
% 
% imshow(C{1}(:,:,1)); hold on;
% plot(corners.selectStrongest(150));

% https://github.com/gokhanozbulak/Harris-Detector.git

harris(C{1}(:,:,1));