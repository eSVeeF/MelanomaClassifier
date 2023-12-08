import numpy as np

FEATURE_NAME = "Perimeter"
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
    #use numpy.diff to find differences along rows and columns
    d1 = np.diff(image, axis=1)
    d0 = np.diff(image, axis=0)

    #Count non-zero (True) values to find perimeter
    perimeter = np.count_nonzero(d1) + np.count_nonzero(d0)

    # count border pixels
    perimeter += np.count_nonzero(image[0, :])  # Top border
    perimeter += np.count_nonzero(image[-1, :])  # Bottom border
    perimeter += np.count_nonzero(image[:, 0])  # Left border
    perimeter += np.count_nonzero(image[:, -1])  # Right border

    return perimeter

def combine_features(image_normal, image_lesion):
    # Implement logic to combine features from both image types
    pass