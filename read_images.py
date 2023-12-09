from PIL import Image
from PreProcessing import HairRemoval, MedianFilter, HairRemoval2
import os
import matplotlib.pyplot as plt
import numpy as np

class ImageLoader:
    """Initialize the ImageLoader with the given data folder."""
    def __init__(self, data_folder, preprocess=False):
        try:
            self.preprocess = preprocess
            self.full_path = os.path.abspath(os.path.join(os.getcwd(), data_folder))
            self.bmp_files = sorted([file for file in os.listdir(self.full_path) if file.lower().endswith('.bmp')])
            self.images, self.images_arrays = self.read_bmps()
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    """Read BMP files from the specified folder in the specified target size and return images and arrays."""
    def read_bmps(self, target_size=(199, 137)):
        # target_size=(766, 575), not all the images have the same pixels
        # all 200 images stored here
        images = []
        # all pixels of the 200 images stored here, each image is a matrix with 575 rows (height of the image)
        # and 766 columns (length of the image), each pixel is an array with RGB values ex:[R= 54, G = 200, B = 150]
        # in the case of the lesion images (Black and white), the pixels are only False for black and True for White
        images_arrays = []

        # Iterate through the bmp files and display each one
        for bmp_file in self.bmp_files:
             # Construct the full path to the bmp file
            image_path = os.path.join(self.full_path, bmp_file)
            # Use Pillow to open the image
            ind_image = Image.open(image_path)

            # Resize the image to the target size
            resized_image = ind_image.resize(target_size)

            # Convert PIL image to a numpy array
            resized_image_np = np.array(resized_image)

            # Process the image
            if self.preprocess:
                processed_image_np = MedianFilter.apply_median_filter(HairRemoval2.morphological_black_hat_transformation(resized_image_np))
                # processed_image_np = HairRemoval2.morphological_black_hat_transformation(resized_image_np)
                # Convert the processed numpy array back to PIL image if necessary
                processed_image = Image.fromarray(processed_image_np)
                images.append(processed_image)
                images_arrays.append(processed_image_np)
            else:
                images.append(resized_image)
                images_arrays.append(resized_image_np)

        return images, images_arrays

    """Display the image at the specified index."""
    def display_image(self, index):
        # Display the image at the given index
        plt.imshow(self.images[index])
        plt.axis('on')
        plt.show()

    """Get the pixels of the image at the specified index and position."""
    def get_all_pixels(self, index):
        # Get the pixels of the image at the given index
        return self.images_arrays[index]

    """Get the pixels of the image at the specified index and position."""
    def get_one_pixel(self, index, x, y):
        # Get the pixels of the image at the given index
        return self.images_arrays[index][x][y]
