import numpy as np

FEATURE_NAME = "Area"
READY = False
IMAGE_TYPE = "LESION"  # Options: "NORMAL", "LESION", "BOTH"

def build(image_normal=None, image_lesion=None):
    if IMAGE_TYPE == "NORMAL":
        return calculate_feature(image_normal)
    elif IMAGE_TYPE == "LESION":
        return calculate_feature(image_lesion)
    elif IMAGE_TYPE == "BOTH":
        return combine_features(image_normal, image_lesion)

def calculate_feature(image):
    # Implement feature extraction logic here
    area = np.sum(image)
    return area

def combine_features(image_normal, image_lesion):
    # Implement logic to combine features from both image types
    pass
