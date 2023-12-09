import cv2

def apply_median_filter(image, kernel_size=5):
    """
    Apply a median filter to remove noise and small artifacts from images.

    Parameters:
    image (numpy.ndarray): Input image.
    kernel_size (int): Size of the kernel for the median filter.

    Returns:
    numpy.ndarray: Image after applying the median filter.
    """

    image_median_filtered = cv2.medianBlur(image, kernel_size)

    # # Uncomment if you need to show the processed image
    # cv2.imshow('Original Image', image)
    # cv2.imshow('Median Filtered Image', image_median_filtered)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return image_median_filtered
