#Experiment 1: Image /Video Reading and Display
import cv2
import sys

# -------------------- IMAGE ACQUISITION --------------------

def demonstrate_image_acquisition(image_path):
    """
    Reads an image in multiple modes and displays them
    """
    # Read image in different modes
    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_unchanged = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Check if images loaded successfully
    if img_color is None or img_gray is None or img_unchanged is None:
        print(f" Error: Failed to load image '{image_path}'")
        sys.exit(1)

    print("Images loaded successfully")

    # Display images
    cv2.imshow("Color Image", img_color)
    cv2.imshow("Grayscale Image", img_gray)
    cv2.imshow("Unchanged Image", img_unchanged)

    print("Press any key to close the image windows")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Image windows closed\n")


# -------------------- VIDEO ACQUISITION --------------------

def demonstrate_video_acquisition(output_file='output_video.avi', frame_width=640, frame_height=480, fps=20.0):
    """
    Captures live video from webcam, displays it, and stores it to a file
    """
    # Open default webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" Error: Cannot open webcam")
        sys.exit(1)

    # Set frame width and height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # Define video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    print("Live video capture started")
    print("Press 'q' to stop recording")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Error: Failed to grab frame")
            break

        # Display the frame
        cv2.imshow('Live Video', frame)

        # Write frame to file
        out.write(frame)

        # Exit if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Video recording stopped by user")
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved as '{output_file}'\n")


# -------------------- MAIN PROGRAM --------------------

if __name__ == "__main__":
    IMAGE_PATH = "sample.jpg"  # Make sure this image is in the same folder as this script

    print("=== IMAGE ACQUISITION ===")
    demonstrate_image_acquisition(IMAGE_PATH)

    print("=== VIDEO ACQUISITION ===")
    demonstrate_video_acquisition()
    print("Practical completed successfully ")


