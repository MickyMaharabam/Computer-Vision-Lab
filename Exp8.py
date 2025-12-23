# Experiment 8: Texture Segmentation using Gabor Filters.
import cv2
import numpy as np

# Read the image in grayscale
img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Image not found")
    exit()

# Define Gabor filter parameters
ksize = 31        # Size of the filter
sigma = 4.0       # Standard deviation of the gaussian
theta_values = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Orientations
lambd = 10.0      # Wavelength of the sinusoidal factor
gamma = 0.5       # Spatial aspect ratio
psi = 0           # Phase offset

# Apply a bank of Gabor filters
filtered_images = []
for theta in theta_values:
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
    fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
    filtered_images.append(fimg)
    cv2.imshow(f"Gabor Theta {theta:.2f}", fimg)

cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()