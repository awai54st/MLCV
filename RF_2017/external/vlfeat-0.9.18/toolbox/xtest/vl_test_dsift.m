function results = vl_test_dsift(varargin)
% VL_TEST_DSIFT
vl_test_init ;

function s = setup()
I = im2double(imread(fullfile(vl_root,'data','spots.jpg'))) ;
s.I = rgb2gray(single(I)) ;

function test_fast_slow(s)
binSize     = 4 ; % bin size in pixels
magnif      = 3 ; % bin size / keypoint scale
scale       = binSize / magnif ;
windowSize  = 5 ;

[f, d] = vl_dsift(vl_imsmooth(s.I, sqrt(scale.^2 - .25)), ...
                  'size', binSize, ...
                  'step', 10, ...
                  'bounds', [20,20,210,140], ...
                  'windowsize', windowSize, ...
                  'floatdescriptors') ;

[f_, d_] = vl_dsift(vl_imsmooth(s.I, sqrt(scale.^2 - .25)), ...
                    'size', binSize, ...
                    'step', 10, ...
                    'bounds', [20,20,210,140], ...
                    'windowsize', windowSize, ...
                    'floatdescriptors', ...
                    'fast') ;

error = std(d_(:) - d(:)) / std(d(:)) ;
assert(error < 0.1,  'dsift fast approximation not close') ;

function test_sift(s)
binSize     = 4 ; % bin size in pixels
magnif      = 3 ; % bin size / keypoint scale
scale       = binSize / magnif ;

windowSizeRange = [1 1.2 5] ;
for wi = 1:length(windowSizeRange)
  windowSize = windowSizeRange(wi) ;

  [f, d] = vl_dsift(vl_imsmooth(s.I, sqrt(scale.^2 - .25)), ...
                    'size', binSize, ...
                    'step', 10, ...
                    'bounds', [20,20,210,140], ...
                    'windowsize', windowSize, ...
                    'floatdescriptors') ;

  numKeys = size(f, 2) ;
  f_ = [f ; ones(1, numKeys) * scale ; zeros(1, numKeys)] ;

  [f_, d_] = vl_sift(s.I, ...
                     'magnif', magnif, ...
                     'frames', f_, ...
                     'firstoctave', -1, ...
                     'levels', 5, ...
                     'floatdescriptors', ...
                     'windowsize', windowSize) ;

  error = std(d_(:) - d(:)) / std(d(:)) ;
  assert(error < 0.1,  'dsift and sift equivalence') ;
end
