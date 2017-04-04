function [image1_coor,image2_coor] = clickImg(image1,image2)

% Show image A
image1_coor = [];
figure;
imshow(image1); hold on
[x,y] = ginput(1);
scatter(x,y,'r');
image1_coor = [image1_coor;[x,y]];
set(gcf,'currentchar',' ')         % set a dummy character
while get(gcf,'currentchar')==' '  % which gets changed when key is pressed
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
while get(gcf,'currentchar')==' '  % which gets changed when key is pressed
   [x,y] = ginput(1);
   scatter(x,y,'r');
   image2_coor = [image2_coor;[x,y]];
end
hold off


end






