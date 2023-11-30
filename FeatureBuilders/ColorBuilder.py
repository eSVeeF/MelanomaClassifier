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
    counter = []
    for name_color in ['White','Red','Light brown','Dark brown','Blue-gray', 'Black']:
        counter.append(color_percentage(image,name_color, 0.05))
    return np.array(counter)

# table 1 Thresholding of Red, Green and Blue channels for creating six colors
def color_percentage(image,name_color, percentage):
    counter = 0 #from 0 to 199
    for pixel_row in image:
        for pixel in pixel_row:
            R, G, B = pixel[0]/255, pixel[1]/255, pixel[2]/255 # R,G,B Normalize the values to the range [0, 1]
            if name_color == 'White':
                if R >= 0.8 and G >= 0.8 and B >= 0.8: 
                    counter += 1
            elif name_color == 'Red':
                if R >= 0.588 and G < 0.2 and B < 0.2: 
                    counter += 1
            elif name_color == 'Light brown':
                if 0.588 <= R <= 0.94 and 0.196 <= G <= 0.588 and 0 <= B < 0.392: 
                    counter += 1
            elif name_color == 'Dark brown':
                if 0.243 < R < 0.56 and 0 <= G < 0.392 and 0 < B < 0.392: 
                    counter += 1
            elif name_color == 'Blue-gray':
                if 0 <= R <= 0.588 and 0.392 <= G <= 0.588 and 0.490 <= B <= 0.588: 
                    counter += 1
            elif name_color == 'Black':
                if R <= 0.243 and G <= 0.243 and B <= 0.243: 
                    counter += 1
                
    # total pixels length of rows + length of columns, we use the 1st image for computation        
    total_pixels = image.shape[0] + image.shape[1]
    # check if more that percentage %
    if counter >= (total_pixels*percentage):
        counter = 1
    else:
        counter = 0
    return counter

def combine_features(image_normal, image_lesion):
    # Implement logic to combine features from both image types
    pass