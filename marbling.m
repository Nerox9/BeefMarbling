addpath('Scripts');
dataset_path = fullfile(pwd, 'Beef Dataset');
image_path = fullfile(dataset_path, 'Choice-Moderate0.png');

%% Get image
image = imread(image_path);
padded = padarray(image,[3 3],'replicate','both');
gray = rgb2gray(padded);

%% Median Blurring
blurred = medfilt2(gray, [5 5]);
unpadded_size = size(blurred);
blurred = blurred(4:unpadded_size(1)-3, 4:unpadded_size(2)-3);

%% Region Growing method
background = regionGrowing(double(blurred),unpadded_size(1)-6,unpadded_size(2)-6,10);

%% Thresholding
otsulevel = graythresh(blurred);
binimage = im2bw(blurred,otsulevel);
imshow(binimage.*(1-background))