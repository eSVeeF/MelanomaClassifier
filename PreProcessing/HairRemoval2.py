import cv2
import numpy as np

def morphological_black_hat_transformation(image):
    # Convert the original image to grayscale
    grayScale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Make sure to use BGR2GRAY for color conversion

    # Kernel for the morphological filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))  # Use cv2.MORPH_RECT for the kernel

    # Perform the blackHat filtering on the grayscale image to find the hair contours
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

    # Intensify the hair contours in preparation for the inpainting algorithm
    ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # Inpaint the original image depending on the mask
    repaired = cv2.inpaint(image, thresh2, 1, cv2.INPAINT_TELEA)

    # # Uncomment if you need to show the processed image
    # cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('Hair Removed Image', cv2.WINDOW_NORMAL)
    # cv2.imshow('Original Image', image)
    # cv2.imshow('Hair Removed Image', repaired)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return repaired


