from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

class ImageLoader:
    """Initialize the ImageLoader with the given data folder."""
    def __init__(self, data_folder):
        try:
            self.full_path = os.path.abspath(os.path.join(os.getcwd(), data_folder))
            self.bmp_files = [file for file in os.listdir(self.full_path) if file.lower().endswith('.bmp')]
            self.images, self.images_arrays = self.read_bmps()
        except FileNotFoundError as e:
            print(f"Error: {e}. Please check if the specified folder exists.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    """Read BMP files from the specified folder in the specified target size and return images and arrays."""
    def read_bmps(self, target_size=(100, 100)):
        # target_size=(761, 553), not all the images have the same pixels, the smallest one is 553 height
        # and 761 length
        # all 200 images stored here
        images = []
        # all pixels of the 200 images stored here, each image is a matrix with 576 rows (height of the image)
        # and 768 columns (length of the image), each pixel is an array with RGB values ex:[R= 54, G = 200, B = 150]
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

            images.append(resized_image)
            # Convert the image to a NumPy array
            images_arrays.append(np.array(resized_image))

        return images, images_arrays

    """Display the image at the specified index."""
    def display_image(self, index):
        # Display the image at the given index
        plt.imshow(self.images[index])
        plt.axis('on')
        plt.show()

    """Get the pixels of the image at the specified index and position."""
    def get_image_pixels(self, index, x, y):
        # Get the pixels of the image at the given index
        return self.images_arrays[index][x][y]
