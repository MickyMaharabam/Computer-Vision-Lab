# Experiment 2: Image Transformations 
#  Python program using OpenCV to enhance the quality of a digital image by applying spatial domain filtering techniques such as smoothing and sharpening
import cv2
import numpy as np
import sys

# -------------------- FUNCTION DEFINITIONS --------------------

def scale_image(img, scale_x=1.0, scale_y=1.0):
    """
    Scales the image by scale_x and scale_y factors
    """
    height, width = img.shape[:2]
    new_size = (int(width * scale_x), int(height * scale_y))
    scaled_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
    return scaled_img

def translate_image(img, shift_x=0, shift_y=0):
    """
    Translates the image by shift_x and shift_y
    """
    height, width = img.shape[:2]
    translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    translated_img = cv2.warpAffine(img, translation_matrix, (width, height))
    return translated_img

def rotate_image(img, angle=0):
    """
    Rotates the image around its center by the given angle (degrees)
    """
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)  # scale=1.0
    rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
    return rotated_img

# -------------------- MAIN PROGRAM --------------------

if __name__ == "__main__":
    IMAGE_PATH = "sample.jpg"  

    # Load original image
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f" Error: Failed to load image '{IMAGE_PATH}'")
        sys.exit(1)

    print("Original image loaded successfully")

    # -------------------- SCALING --------------------
    scaled_img = scale_image(img, scale_x=1.5, scale_y=1.5)

    # -------------------- TRANSLATION --------------------
    translated_img = translate_image(img, shift_x=100, shift_y=50)

    # -------------------- ROTATION --------------------
    rotated_img = rotate_image(img, angle=45)

    # -------------------- DISPLAY RESULTS --------------------
    cv2.imshow("Original Image", img)
    cv2.imshow("Scaled Image", scaled_img)
    cv2.imshow("Translated Image", translated_img)
    cv2.imshow("Rotated Image", rotated_img)

    print("Press any key to close all image windows")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(" Geometric transformations completed")

