import cv2
import os
import numpy as np

dataset_path = os.path.join(os.getcwd(), "Beef Dataset")
image_path = os.path.join(dataset_path, "0.jpg")

image = cv2.imread(image_path, 0)
blurred = cv2.medianBlur(image, 5)
#background = region_growing(image, image.shape[:2])
ret, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
canny = cv2.Canny(blurred, 20, 200)
kernel = np.ones((3,3), np.uint8)
dilation = cv2.dilate(canny, kernel, iterations=2)

h, w = dilation.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)
im_floodfill = dilation.copy()
cv2.floodFill(im_floodfill, mask, (0,0), 255)
cat = np.concatenate((image, im_floodfill), axis=1)

cv2.imshow("",cat)
cv2.waitKey(0)