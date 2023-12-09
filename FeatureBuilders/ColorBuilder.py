import numpy as np

FEATURE_NAME = "Colors"
READY = True
IMAGE_TYPE = "NORMAL"  # Options: "NORMAL", "LESION", "BOTH"

def build(image_normal=None, image_lesion=None):
    if IMAGE_TYPE == "NORMAL":
        return calculate_feature(image_normal)
    elif IMAGE_TYPE == "LESION":
        return calculate_feature(image_lesion)
    elif IMAGE_TYPE == "BOTH":
        return combine_features(image_normal, image_lesion)

def calculate_feature(image):
    # Normalize RGB values
    normalized_image = image / 255.0

    # Precompute masks
    masks = {
        'White': np.all(normalized_image >= 0.8, axis=2),
        'Red': (normalized_image[:, :, 0] >= 0.588) & (normalized_image[:, :, 1] < 0.2) & (normalized_image[:, :, 2] < 0.2),
        'Light brown': (0.588 <= normalized_image[:, :, 0]) & (normalized_image[:, :, 0] <= 0.94) & (0.196 <= normalized_image[:, :, 1]) & (normalized_image[:, :, 1] <= 0.588) & (normalized_image[:, :, 2] < 0.392),
        'Dark brown': (0.243 < normalized_image[:, :, 0]) & (normalized_image[:, :, 0] < 0.56) & (normalized_image[:, :, 1] < 0.392) & (normalized_image[:, :, 2] < 0.392),
        'Blue-gray': (normalized_image[:, :, 0] <= 0.588) & (0.392 <= normalized_image[:, :, 1]) & (normalized_image[:, :, 1] <= 0.588) & (0.490 <= normalized_image[:, :, 2]) & (normalized_image[:, :, 2] <= 0.588),
        'Black': np.all(normalized_image <= 0.243, axis=2)
    }

    # Calculate color percentages
    total_pixels = image.shape[0] * image.shape[1]
    percentage_threshold = total_pixels * 0.05
    counter = [1 if np.sum(masks[color]) >= percentage_threshold else 0 for color in masks]

    return np.sum(np.array(counter))

def combine_features(image_normal, image_lesion):
    # Implement logic to combine features from both image types
    pass
