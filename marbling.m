addpath('Scripts');
dataset_path = fullfile(pwd, 'Beef Dataset');
image_path = fullfile(dataset_path, '0.jpg');

% Get image
image = imread(image_path);
padded = padarray(image,[3 3],'replicate','both');
gray = rgb2gray(padded);

% Median Blurring
blurred = medfilt2(gray, [5 5]);
unpadded_size = size(blurred);
blurred = blurred(4:unpadded_size(1)-3, 4:unpadded_size(2)-3);

%% Region Growing method
[P J]=regionGrowing(gray)
