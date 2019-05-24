*********************************************
*		  marbling.m		    *
*********************************************
The main script of meat marbling has some subpart which is separated in the script as dataset and label setting, segmentation methods, feature setting and backpropagation neural network training.
Also final trained network is saved as network.mat and it can be used as tranedNet({input}).

Dataset images are PNG files which are in Beef Dataset folder.(subfolders are not included)

Report and script comments has more detail on segmentation and other implementations.

*********************************************
*		regionGrowing.m		    *
*********************************************
This script is a copy from https://www.mathworks.com/matlabcentral/fileexchange/19084-region-growing by Dirk-Jan Kroon.

Simple but effective "Region Growing" method from a single seed point.
The region is iteratively grown by comparing all unallocated neighbouring pixels to the region. 
The difference between a pixel's intensity value and the region's mean, is used as a measure of similarity. 
The pixel with the smallest difference measured this way is allocated to the region.
This process stops when the intensity difference between region mean and new pixel becomes larger than a certain treshold.

It has 4 input and 1 output as:

J = regiongrowing(I,x,y,t) 

I : input image 
J : logical output image of region
x,y : the position of the seedpoint (if not given uses function getpts)
t : maximum intensity distance (defaults to 0.2)