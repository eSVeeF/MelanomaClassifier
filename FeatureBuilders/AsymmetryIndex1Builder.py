import numpy as np
from scipy import ndimage
from skimage import transform
from skimage.measure import regionprops
from skimage.morphology import label

FEATURE_NAME = "AsymmetryIndex1"
READY = True
IMAGE_TYPE = "LESION"  # Options: "NORMAL", "LESION", "BOTH"

def build(image_normal=None, image_lesion=None):
    if IMAGE_TYPE == "NORMAL":
        return calculate_feature(image_normal)
    elif IMAGE_TYPE == "LESION":
        return calculate_feature(image_lesion)
    elif IMAGE_TYPE == "BOTH":
        return combine_features(image_normal, image_lesion)

def calculate_orientation(lesion_mask):
    """
    Calculate the orientation of a lesion from its binary mask (across major axis).

    Parameters:
    lesion_mask (numpy.ndarray): Binary mask of the lesion.

    Returns:
    float: Orientation angle in radians.
    """
    labeled_mask = label(lesion_mask)
    properties = regionprops(labeled_mask)
    if properties:
        orientation = properties[0].orientation
    else:
        orientation = 0  # Default orientation if no region is found

    return orientation


def calculate_asymmetry_index(lesion_mask):
    """
    Calculate the asymmetry index of a lesion from its binary mask.
    
    Parameters:
    lesion_mask (numpy.ndarray): Binary mask of the lesion.

    Returns:
    float: Asymmetry index.
    """
    # Ensure the lesion mask is binary
    lesion_mask = (lesion_mask > 0).astype(np.uint8)

    # Check if there are any lesion pixels
    if np.sum(lesion_mask) == 0:
        return 0  # No lesion present, asymmetry index is 0

    # Calculate the centroid of the lesion
    centroid = ndimage.center_of_mass(lesion_mask)

    # Calculate the orientation angle of the lesion
    orientation = calculate_orientation(lesion_mask)

    # Rotate the lesion to align the major axis with the x-axis
    rotated_lesion = transform.rotate(lesion_mask, -np.degrees(orientation), center=centroid, preserve_range=True)
    rotated_lesion_binary = (rotated_lesion > 0.5).astype(np.uint8)  # Convert to binary mask

    # Check if the rotated lesion still contains lesion pixels
    if np.sum(rotated_lesion_binary) == 0:
        return 0  # No lesion present after rotation

    # Flip the lesion across the major axis (vertical flip)
    flipped_lesion = np.flipud(rotated_lesion_binary)

    # Calculate non-overlapping region
    non_overlapping_region = np.bitwise_xor(rotated_lesion_binary, flipped_lesion)

    # Calculate the asymmetry index
    asymmetry_index = np.sum(non_overlapping_region) / np.sum(rotated_lesion_binary)

    return asymmetry_index


def calculate_feature(image):

    return calculate_asymmetry_index(image)

def combine_features(image_normal, image_lesion):
    # Implement logic to combine features from both image types
    pass
