#Experiment 5: Image Enhancement
import cv2
import numpy as np

# Step 1: Read the image
img = cv2.imread("face1.png")  # Change to your image file
cv2.imshow("Original Image", img)

# ------------------ SMOOTHING ------------------
# Apply Gaussian Blur to reduce noise
smoothed = cv2.GaussianBlur(img, (5, 5), 0)
cv2.imshow("Smoothed Image", smoothed)

# ------------------ SHARPENING ------------------
# Sharpening kernel
sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

# Apply sharpening to the smoothed image
enhanced = cv2.filter2D(smoothed, -1, sharpen_kernel)
cv2.imshow("Enhanced Image (Smoothed + Sharpened)", enhanced)

# Wait for key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
