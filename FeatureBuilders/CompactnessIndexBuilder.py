import numpy as np
import importlib
import math


FEATURE_NAME = "CompactnessIndex"
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
    module_area = importlib.import_module(f'FeatureBuilders.AreaBuilder')
    module_perimeter = importlib.import_module(f'FeatureBuilders.PerimeterBuilder')
    feature_area = module_area.build(image_lesion=image)
    feature_perimeter = module_perimeter.build(image_lesion=image)
    compactness = 4*math.pi*feature_area/(feature_perimeter**2)
    return compactness

def combine_features(image_normal, image_lesion):
    # Implement logic to combine features from both image types
    pass
