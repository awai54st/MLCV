% Please run this script under the root folder

clearvars -except N;
close all;

% addpaths
addpath('../RandomForest/external');
addpath('../RandomForest/external/libsvm-3.18/matlab');

% initialise external libraries
run('../RandomForest/external/vlfeat-0.9.18/toolbox/vl_setup.m'); % vlfeat library
cd('../RandomForest/external/libsvm-3.18/matlab'); % libsvm library
run('make');
cd('../../../../ImageMatching/');

% tested on Ubuntu 12.04, 64-bit, Intel® Core�?i7-3820 CPU @ 3.60GHz × 8 