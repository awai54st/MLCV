function [image1_coor,image2_coor] = clickImg(image1,image2)
% This function takes in two greyscale images for feature matching, and returns two sequences of feature coordinates for image A and B respectively. A user interface will pop out, asking user to manually click on interest points from image A and then click the same points , in the same order, in image B.

% Show image A
image1_coor = [];
figure;
imshow(image1); hold on
[x,y] = ginput(1);
scatter(x,y,'r');
image1_coor = [image1_coor;[x,y]];
set(gcf,'currentchar',' ')         % set a dummy character
% Click on all features in image A. Press ESC to go to image B
while get(gcf,'currentchar')==' '
   [x,y] = ginput(1);
   scatter(x,y,'r');
   image1_coor = [image1_coor;[x,y]];
end
hold off
% Show image B
image2_coor = [];
figure;
imshow(image2); hold on
[x,y] = ginput(1);
scatter(x,y,'r');
image2_coor = [image2_coor;[x,y]];
set(gcf,'currentchar',' ')         % set a dummy character
% Click on all features in image B. Press ESC finish
while get(gcf,'currentchar')==' '
   [x,y] = ginput(1);
   scatter(x,y,'r');
   image2_coor = [image2_coor;[x,y]];
end
hold off
end