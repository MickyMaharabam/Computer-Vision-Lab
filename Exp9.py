#Experiment 9: Image Segmentation 
import cv2
import numpy as np

# Read input image
img = cv2.imread("sample.jpg")
if img is None:
    print("Image not found")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# 1. Classical Segmentation


# ----- 1a: Simple Thresholding -----
_, thresh_img = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

# ----- 1b: Adaptive Thresholding -----
adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 11, 2)

# ----- 1c: Region-based Segmentation (Watershed) -----
# Convert to binary for watershed
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Noise removal
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

# Sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

# Unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labeling
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown==255] = 0

# Apply watershed
ws_img = img.copy()
markers = cv2.watershed(ws_img, markers)
ws_img[markers == -1] = [0,0,255]  # Mark boundaries in red


# Display Results
cv2.imshow("Original Image", img)
cv2.imshow("Thresholding", thresh_img)
cv2.imshow("Adaptive Thresholding", adaptive_thresh)
cv2.imshow("Region-based Segmentation (Watershed)", ws_img)

cv2.waitKey(0)
cv2.destroyAllWindows()