addpath('Scripts');
dataset_path = fullfile(pwd, 'Beef Dataset');
image_path = fullfile(dataset_path, 'Choice-Moderate_havuz.png');

%% Get image
image = imread(image_path);
padded = padarray(image,[3 3],'replicate','both');
gray = rgb2gray(padded);

%% Median Blurring
blurred = medfilt2(gray, [3 3]);
unpadded_size = size(blurred);
blurred = blurred(4:unpadded_size(1)-3, 4:unpadded_size(2)-3);

%% Region Growing method
background = regionGrowing(double(blurred),unpadded_size(1)-6,unpadded_size(2)-6,10);

%% Thresholding
otsulevel = graythresh(blurred);
binimage = im2bw(blurred,otsulevel);

%% Image Subtraction
fatimage = binimage.*(~background);
meatTissue = xor(fatimage,~background);

%% Print
subplot(2,5,1)
imshow(image)
title('(1) Original Image')
subplot(2,5,2)
imshow(gray)
title('(2) Grayscale Image')
subplot(2,5,3)
imshow(blurred)
title('(3) Median Blurring')
subplot(2,5,4)
imshow(background)
title('(4) Background(Region Growing Method)')
subplot(2,5,5)
imshow(binimage)
title('(5) Binarisation')
subplot(2,5,6)
imshow(fatimage)
title('(6) 5 * ~4')
subplot(2,5,7)
meatimage = bwareafilt(~binimage,1);
imshow(meatimage)
title('(7) Only Biggest Area Remains of ~6')
subplot(2,5,8)
filledmeat = imfill(meatimage, 'holes');
imshow(filledmeat)
title('(8) Fill Holes')
subplot(2,5,9)
onlyfat = and(filledmeat, fatimage);
imshow(onlyfat)
title('(9) 8 AND 6')
subplot(2,5,10)
onlymeat = xor(onlyfat, filledmeat);
imshow(onlymeat)
title('(10) 9 XOR 8')

fatcount = bwconncomp(onlyfat)
fatcount.NumObjects

fatarea = bwarea(onlyfat)

meatarea = bwarea(onlymeat)

fattomeatratio = fatarea/meatarea
%% BPNN
%%train_bpnn(0)