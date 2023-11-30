import numpy as np
# Example: FeatureBuilders/DotsGlobulesAreaBuilder.py

FEATURE_NAME = "Dots/Globules"
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
    # Example: # Calculate area of black regions (dots/globules)
    return np.sum(image == 0)

def combine_features(image_normal, image_lesion):
    # Implement logic to combine features from both image types
    pass

