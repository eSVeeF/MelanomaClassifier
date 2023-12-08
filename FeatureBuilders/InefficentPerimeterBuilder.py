import numpy as np

FEATURE_NAME = "InefficientPerimeter"
READY = False
IMAGE_TYPE = "LESION"  # Options: "NORMAL", "LESION", "BOTH"

def build(image_normal=None, image_lesion=None):
    if IMAGE_TYPE == "NORMAL":
        return calculate_feature(image_normal)
    elif IMAGE_TYPE == "LESION":
        return calculate_feature(image_lesion)
    elif IMAGE_TYPE == "BOTH":
        return combine_features(image_normal, image_lesion)

def calculate_feature(image): # maybe we can do this at the same time we build the colors to save computation time
    rows, cols = image.shape
    perimeter = 0

    for i in range(rows):
        for j in range(cols):
            if image[i, j]:  # If the pixel is white
                neighbors = [
                    image[max(0, i - 1), j],
                    image[min(rows - 1, i + 1), j],
                    image[i, max(0, j - 1)],
                    image[i, min(cols - 1, j + 1)]
                ]

                perimeter += neighbors.count(False)

                # pixel is at the border of the image we also have to count as perimeter
                if i == 0 or i == rows - 1:
                    perimeter += 1
                if j == 0 or j == cols - 1:
                    perimeter += 1

    return perimeter


def combine_features(image_normal, image_lesion):
    # Implement logic to combine features from both image types
    pass
