import cv2
import numpy as np

def dull_razor(image, kernel_size=15, threshold=50):
    """
    Apply the Dull-Razor algorithm to remove hair from dermoscopy images.

    Parameters:
    image (numpy.ndarray): Input image.
    kernel_size (int): Size of the kernel for morphological operations.
    threshold (int): Threshold value to detect hair pixels.

    Returns:
    numpy.ndarray: Image with hair removed.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect dark hair-like structures
    _, blackhat = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(blackhat, cv2.MORPH_CLOSE, kernel)

    # Repair the image
    repaired = cv2.inpaint(image, blackhat, kernel_size, cv2.INPAINT_TELEA)

    # # Uncomment if you need to show the processed image
    # cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('Hair Removed Image', cv2.WINDOW_NORMAL)
    # cv2.imshow('Original Image', image)
    # cv2.imshow('Hair Removed Image', repaired)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return repaired

