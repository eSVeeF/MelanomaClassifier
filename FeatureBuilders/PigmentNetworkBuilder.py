import numpy as np
# Example: FeatureBuilders/PigmentNetworkBuilder.py

FEATURE_NAME = "Pigment Network"
READY = True
IMAGE_TYPE = "NORMAL"  # Options: "NORMAL", "LESION", "BOTH"

def build(image_normal, image_lesion=None):
    if IMAGE_TYPE == "NORMAL":
        return calculate_feature(image_normal)
    elif IMAGE_TYPE == "LESION":
        return calculate_feature(image_lesion)
    elif IMAGE_TYPE == "BOTH":
        return combine_features(image_normal, image_lesion)

def calculate_feature(image):
    # Implement feature extraction logic here
    # Example: Calculate area of white regions (pigment network)
    return np.sum(image == 255)

def combine_features(image_normal, image_lesion):
    # Implement logic to combine features from both image types
    pass
