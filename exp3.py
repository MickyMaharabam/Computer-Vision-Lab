#Experiment 3: Histogram Analysis 
# Python program using OpenCV to apply convolution-based filtering techniques and simulate motion blur.
import cv2
import cv2
import matplotlib.pyplot as plt

# --- Load the image in grayscale ---
image_path = "sample.png"  # Make sure this image is in the same folder
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Image not found.")
    exit()

# --- Compute the histogram of the original image ---
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

# --- Perform histogram equalization ---
equalized_img = cv2.equalizeHist(img)
equalized_hist = cv2.calcHist([equalized_img], [0], None, [256], [0, 256])

# --- Display images using OpenCV ---
cv2.imshow("Original Image", img)
cv2.imshow("Equalized Image", equalized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- Plot histograms using Matplotlib ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(hist, color='blue')
plt.title("Original Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.plot(equalized_hist, color='green')
plt.title("Equalized Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.show()


