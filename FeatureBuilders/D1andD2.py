import numpy as np
import cv2
FEATURE_NAME = "D1andD2"
READY = True
IMAGE_TYPE = "LESION"  # Options: "NORMAL", "LESION", "BOTH"

def build(image_normal=None, image_lesion=None):
    if IMAGE_TYPE == "NORMAL":
        return calculate_feature(image_normal)
    elif IMAGE_TYPE == "LESION":
        return calculate_feature(image_lesion)
    elif IMAGE_TYPE == "BOTH":
        return combine_features(image_normal, image_lesion)

def calculate_feature(image):
    center, axes, angle = calculate_elipse(image)
    a = axes[1]
    b = axes[0]
    area = np.sum(image)
    d1 = (4*area/3.1415926)**0.5
    d2 = (2*a+2*b)/2
    D1 = (d1+d2)/2
    D2 = 2*(a-b)
    return np.array([D1, D2])

def combine_features(image_normal, image_lesion):
    # Implement logic to combine features from both image types
    pass

def calculate_elipse(image):
    # Find contours in the binary image
    contours, _ = cv2.findContours(image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Fit ellipse to the largest contour
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        ellipse = cv2.fitEllipse(largest_contour)

        # Draw the ellipse on a blank image
        ellipse_image = np.zeros_like(image, dtype=np.uint8)
        cv2.ellipse(ellipse_image, ellipse, 255, 1)

        # Access ellipse parameters
        center, axes, angle = ellipse
    else:
        center, axes, angle = (0.0, 0.0), (0.0, 0.0), 0
    return center, axes, angle
