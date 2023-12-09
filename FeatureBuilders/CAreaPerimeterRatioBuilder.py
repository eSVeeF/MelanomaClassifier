import numpy as np
import scipy.ndimage

FEATURE_NAME = "PerimeterRatioAreaBuilder"
READY = True
IMAGE_TYPE = "LESION"  # Options: "NORMAL", "LESION", "BOTH"

def build(image_normal=None, image_lesion=None):
    if IMAGE_TYPE == "NORMAL":
        return calculate_feature(image_normal)
    elif IMAGE_TYPE == "LESION":
        return calculate_feature(image_lesion)
    elif IMAGE_TYPE == "BOTH":
        return combine_features(image_normal, image_lesion)

def calculate_area(lesion_mask):
    """
    Calculate the area of the lesion.
    """
    return np.sum(lesion_mask)

def calculate_perimeter(lesion_mask):
    """
    Calculate the perimeter of the lesion using a basic method.
    """
    structure = np.array([[1,1,1], [1,1,1], [1,1,1]])  # 8-connectivity
    eroded_image = scipy.ndimage.binary_erosion(lesion_mask, structure)
    boundary = np.logical_xor(lesion_mask, eroded_image)
    return np.sum(boundary)

def calculate_area_to_perimeter_ratio(lesion_mask):
    """
    Calculate the Area to Perimeter Ratio of a lesion.
    """
    area = calculate_area(lesion_mask)
    perimeter = calculate_perimeter(lesion_mask)
    return area / perimeter if perimeter else float('inf')

def calculate_feature(image):
    # Implement feature extraction logic here
    area = np.sum(image)
    return calculate_area_to_perimeter_ratio(image)

def combine_features(image_normal, image_lesion):
    # Implement logic to combine features from both image types
    pass
