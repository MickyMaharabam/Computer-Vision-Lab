import cv2
import numpy as np

# Read image
img = cv2.imread("sample.jpg")
if img is None:
    print("Image not found")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ------------------ CLASSICAL METHODS ------------------

# 1. Simple Thresholding
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 2. Region-based Segmentation (Watershed)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
sure_bg = cv2.dilate(opening, kernel, iterations=3)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0
ws_img = img.copy()
markers = cv2.watershed(ws_img, markers)
ws_img[markers == -1] = [0, 0, 255]

# ------------------ LEARNING-BASED METHOD ------------------

# K-Means Clustering
Z = img.reshape((-1,3))
Z = np.float32(Z)
K = 3
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
_, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
kmeans_img = center[label.flatten()].reshape(img.shape)

# ------------------ DISPLAY RESULTS ------------------

cv2.imshow("Original Image", img)
cv2.imshow("Thresholding", thresh)
cv2.imshow("Watershed Segmentation", ws_img)
cv2.imshow("K-Means Segmentation", kmeans_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
