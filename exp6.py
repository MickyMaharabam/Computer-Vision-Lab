#Experiment 6: Convolution, Filtering, and Motion Blur in Digital Images 
# Sunil Angom NDU202400105
import cv2
import numpy as np

# Read the image
img = cv2.imread("con.jpg")

if img is None:
    print("Image not found")
    exit()


# 1. Convolution / Filtering

# Example: Edge detection kernel
edge_kernel = np.array([[-1, -1, -1],
                        [-1,  8, -1],
                        [-1, -1, -1]])
edge_filtered = cv2.filter2D(img, -1, edge_kernel)


# 2. Motion Blur Simulation

# Create a horizontal motion blur kernel
size = 15  # length of motion blur
motion_kernel = np.zeros((size, size))
motion_kernel[int((size-1)/2), :] = np.ones(size)
motion_kernel = motion_kernel / size

motion_blurred = cv2.filter2D(img, -1, motion_kernel)


# Display Results
cv2.imshow("Original Image", img)
cv2.imshow("Edge Filtered Image", edge_filtered)
cv2.imshow("Motion Blurred Image", motion_blurred)

cv2.waitKey(0)
cv2.destroyAllWindows()
motion_blur_combined = cv2.filter2D(blur_horizontal, -1, kernel_vertical)

# Show results
cv2.imshow('Combined Motion Blur', motion_blur_combined)

cv2.waitKey(0)
cv2.destroyAllWindows()
