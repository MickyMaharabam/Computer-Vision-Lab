#Experiment 4: Edge Detection 
import cv2
import numpy as np
# Load image in grayscale
image_path = "sample.jpg"  # Replace with your image file
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if image is loaded
if img is None:
    print(f" Error: Failed to load image '{image_path}'")
    exit()
# Set thresholds
threshold1 = 50
threshold2 = 150

# Apply Canny edge detection
edges = cv2.Canny(img, threshold1, threshold2)
cv2.imshow("Original Image", img)
cv2.imshow("Canny Edges", edges)

cv2.waitKey(0) 
cv2.destroyAllWindows()  
