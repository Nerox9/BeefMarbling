addpath('Scripts');
dataset_path = fullfile(pwd, 'Beef Dataset');
image_path = fullfile(dataset_path, 'Choice-Moderate0.png');

images = dir(fullfile(dataset_path, '*.png'));
for i=1:length(images)
    if strfind(images(i).name, 'Prime')
        images(i).label = 2;
    elseif strfind(images(i).name, 'Choice')
        images(i).label = 1;
    elseif strfind(images(i).name, 'Select')
        images(i).label = 0;
    end
    
    %% Get image
    image = imread(fullfile(dataset_path,images(i).name));
    padded = padarray(image,[3 3],'replicate','both');
    gray = rgb2gray(padded);

    %% Median Blurring
    %blurred = medfilt2(gray, [3 3]);
    %unpadded_size = size(blurred);
    %blurred = blurred(4:unpadded_size(1)-3, 4:unpadded_size(2)-3);
    blurred = gray;

    %% Region Growing method
    background = regionGrowing(double(blurred),unpadded_size(1)-6,unpadded_size(2)-6,10);
    se = strel('diamond',1);
    background = imerode(background,se);

    %% Thresholding
    otsulevel = graythresh(blurred);
    binimage = im2bw(blurred,otsulevel-0.2);

    %% Image Subtraction
    fatimage = and(binimage, ~background);

    %% Article Steps
    % 
    % % Fat Extractor by Article
    % CC = bwconncomp(fatimage);
    % numPixels = cellfun(@numel,CC.PixelIdxList);
    % outfat = zeros(size(fatimage));
    % [biggest,idx] = max(numPixels);
    % outfat(CC.PixelIdxList{idx}) = 1;
    % 
    % outliner = ~bwareaopen(~outfat, 1000);
    % segmented = ~binimage-outliner;
    % 
    % % Print
    % subplot(4,2,1)
    % imshow(image)
    % subplot(4,2,2)
    % imshow(gray)
    % subplot(4,2,3)
    % imshow(blurred)
    % subplot(4,2,4)
    % imshow(background)
    % subplot(4,2,5)
    % imshow(binimage)
    % subplot(4,2,6)
    % imshow(fatimage)
    % subplot(4,2,7)
    % imshow(outliner)
    % subplot(4,2,8)
    % imshow(segmented)
    % figure(1)
    % imshow(segmented)
    %% Our Implementation

    % Print
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

    fatCC = bwconncomp(onlyfat)
    fatcount = fatCC.NumObjects

    fatarea = bwarea(onlyfat)

    meatarea = bwarea(onlymeat)

    fattomeatratio = fatarea/meatarea

    images(i).fatCount = fatcount
    images(i).ratio = fattomeatratio
end

%% Features
numInputs = 1;
numLayers = 5;

X = [images.ratio]
T = [images.label]

%% CNN
opts = 'sgdm';
layers = [
    imageInputLayer([1 1]); % Input is an "Image" 1x36 floating point vector
    fullyConnectedLayer(200);
    reluLayer();
    fullyConnectedLayer(100);
    reluLayer();
    fullyConnectedLayer(50);
    reluLayer();
    fullyConnectedLayer(24);
    regressionLayer();
];
trainedNet = trainNetwork(X,T,layers,opts)
%% BPNN

net = feedforwardnet(3);
net.numInputs = numInputs;
trainedNet = train(net,X,T,'useGPU','yes');
save('network.mat', 'trainedNet');