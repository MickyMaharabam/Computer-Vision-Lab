#Experiment 7: Corner Detection 
import cv2
import numpy as np

# Read the image
img = cv2.imread("sample.jpg")
if img is None:
    print("Image not found")
    exit()


# 1. Harris Corner Detection

gray_harris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_harris = np.float32(gray_harris)

# Harris corner detection
dst = cv2.cornerHarris(gray_harris, blockSize=2, ksize=3, k=0.04)

# Dilate result to mark corners
dst = cv2.dilate(dst, None)

# Threshold for marking corners in red
harris_img = img.copy()
harris_img[dst > 0.01 * dst.max()] = [0, 0, 255]  # Red color

cv2.imshow("Harris Corners", harris_img)


# 2. SIFT Keypoint Detection

gray_sift = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()            # Create SIFT detector
keypoints = sift.detect(gray_sift, None)

# Draw keypoints on image
sift_img = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("SIFT Keypoints", sift_img)

cv2.waitKey(0)
cv2.destroyAllWindows()