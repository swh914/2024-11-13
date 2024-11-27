# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 18:35:23 2024
@author: UOU
"""

import cv2
import numpy as np
import os

# Function to detect objects and draw rectangles automatically
def detect_objects(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to segment the image
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # List to store rectangles
    rectangles = []
    for contour in contours:
        # Get bounding rectangle for each contour
        x, y, w, h = cv2.boundingRect(contour)
        if w >= 10 and h >= 10:  # Minimum rectangle size
            rectangles.append((x, y, w, h))
    return rectangles

# Function to display a single image with segmentation
def segment_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {image_path}")
        return False, False

    # Detect objects and get rectangles
    rectangles = detect_objects(image)

    # Create a clone of the image for annotation display
    annotated_image = image.copy()

    while True:
        # Draw rectangles on the image
        temp_image = annotated_image.copy()
        for x, y, w, h in rectangles:
            cv2.rectangle(temp_image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
            # Display the rectangle dimensions
            cv2.putText(temp_image, f"({x}, {y}, {w}, {h})", 
                        (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (0, 255, 0), 
                        1)

        # Display the image with annotations
        cv2.imshow("Image Segmentation", temp_image)
        
        # Press 's' to save annotations, 'q' to quit current or all
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            # Save annotations
            annotation_file = os.path.splitext(image_path)[0] + "_annotations.txt"
            with open(annotation_file, "w") as f:
                for x, y, w, h in rectangles:
                    f.write(f"{x}, {y}, {w}, {h}\n")
            print(f"Annotations saved to {annotation_file}")
        elif key == ord("q"):
            return True  # Signal to quit
        elif key == ord("c"):
            # Clear annotations
            rectangles.clear()
            annotated_image = image.copy()
            print("Annotations cleared.")

    cv2.destroyAllWindows()
    return False

# Function to loop through images in a folder
def segment_images_in_folder(image_folder):
    # Get a list of all image files in the folder
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if not image_files:
        print("No images found in the folder.")
        return
    
    index = 0  # Start with the first image
    
    while 0 <= index < len(image_files):
        image_path = image_files[index]
        print(f"Processing: {image_path}")
        
        # Call segment_image for current image
        exit_program = segment_image(image_path)
        
        if exit_program:
            print("Exiting program.")
            break

        # Handle navigation with arrow keys
        key = cv2.waitKey(0) & 0xFF
        if key == 81:  # Left arrow key
            index = max(0, index - 1)  # Move to the previous image
        elif key == 83:  # Right arrow key
            index = min(len(image_files) - 1, index + 1)  # Move to the next image
        elif key == ord("q"):  # Quit
            print("Exiting...")
            break

    cv2.destroyAllWindows()

# Main program
if __name__ == "__main__":
    # Path to the folder containing images
    image_folder = r"C:\Users\cic\Desktop\SW_Dev\Online_Repo-main\Online_Repo-main\Image_dataset"
    
    # Start the segmentation and viewing process
    segment_images_in_folder(image_folder)
