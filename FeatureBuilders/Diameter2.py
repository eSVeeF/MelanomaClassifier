import numpy as np

FEATURE_NAME = "D2"
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
    a, b = best_fit_ellipse(image)
    D2 = 2 * (a - b)
    return D2

def combine_features(image_normal, image_lesion):
    # Implement logic to combine features from both image types
    pass

def raw_moments(image, order_i, order_j):
    rows, cols = image.shape
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))

    # Raw moments, Eq (2)
    M_ij = np.sum(x**order_i * y**order_j * image)

    return M_ij

def central_moments(image, order_i, order_j):
    rows, cols = image.shape
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    M_00 = raw_moments(image, order_i, order_j)
    # Centroid coordinates, Eq (4)
    x_0 = np.sum(x * image) / M_00 # M_10 /M_00
    y_0 = np.sum(y * image) / M_00 # M_01 /M_00

    # Central moments, Eq (3)
    m_ij = np.sum((x - x_0)**order_i * (y - y_0)**order_j * image)

    return m_ij


def best_fit_ellipse(image):
    m_20 = central_moments(image, 2, 0)
    m_02 = central_moments(image, 0, 2)
    m_11 = central_moments(image, 1, 1)

    # Eq (5)
    eigenvalue_1 = (m_20+m_02)/2 + (((m_20-m_02)**2 + 4*(m_11**2))**0.5)/2
    eigenvalue_2 = (m_20+m_02)/2 - (((m_20-m_02)**2 + 4*(m_11**2))**0.5)/2

    eigenvalue_1 = int(eigenvalue_1)
    eigenvalue_2 = int(eigenvalue_2)
    a = 2*(eigenvalue_1**0.5)
    b = 2*(eigenvalue_2**0.5)
    return a, b