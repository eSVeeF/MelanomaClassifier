import numpy as np
from read_images import ImageLoader
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd  
import warnings # avoid warnings of pandas
warnings.filterwarnings('ignore') # warnings.filterwarnings('default')

relative_path = 'PH2Dataset\mod_PH2_dataset.csv'
current_directory = os.getcwd()
full_path = os.path.join(current_directory, relative_path)
full_path = os.path.abspath(full_path)
# Load your dataset with image names and ground truth labels
dataset = pd.read_csv(full_path)

# Trim leading and trailing whitespaces from the "Name" column
dataset['Name'] = dataset['Name'].str.strip()
# Add the ".bmp" extension to the "Name" column
dataset['Name'] = dataset['Name'] + '.bmp'

# better binary labels, 0 = no_melanoma, 1 = melanoma
index = -1
for label in dataset['Label']:
    index += 1
    if label == "no_melanoma":
        dataset.loc[index,'Label'] = 0
    elif label == "melanoma":
        dataset.loc[index,'Label'] = 1

# create a dataset but keep only this two features
columns_to_keep = [dataset.columns[0], dataset.columns[-1]]
# new dataset
df = dataset[columns_to_keep]
df.loc[80,:]

# Load your images using the ImageLoader
image_loader = ImageLoader('PH2Dataset/Custom Images/Normal')
# Add the pixels as a column
df['Pixels'] = image_loader.images_arrays 
# if want to check we did it correctly 
# aux_dataset["Pixels"][128] == ImageLoader.get_all_pixels(image_loader, 128)

# Load lesion images
image_loader_lesion = ImageLoader('PH2Dataset/Custom Images/Lesion')
# Add the pixels as a column
df['Pixels Lesion'] = image_loader_lesion.images_arrays # Ignore Warning

# computation of features, page 5 of 13
# add 'Area' feature that is number of white pixels, M_00, eq (2)
df['Area'] = 0

# eq
def M_ij(pixels_lesion, i,j):
    index = -1 #from 0 to 199
    for image in pixels_lesion:
        x = 0 # x coordinate of the pixel
        y = 0 # y coordinate of the pixel
        index += 1
        for pixel_row in image:
            x += 1
            for pixel in pixel_row:
                y += 1
                if pixel == True:
                    df.loc[index,'Area'] += 1

# compute area, tsting with only for the first 6 images because it takes time to do all 200                 
M_ij(df['Pixels Lesion'][0:6],0,0) 

# computation of color feature, page 7 of 13

# add each color feature, 0 means that less than 5% of pixels of that color,
# 1 means that 5% or more of the pixels are of that color
df[['White','Red','Light brown','Dark brown','Blue-gray', 'Black']] = 0

# table 1 Thresholding of Red, Green and Blue channels for creating six colors
def color_percentage(pixels_color,name_color, percentage):
    index = -1 #from 0 to 199
    for image in pixels_color:
        index += 1
        for pixel_row in image:
            for pixel in pixel_row:
                R, G, B = pixel[0]/255, pixel[1]/255, pixel[2]/255 # R,G,B Normalize the values to the range [0, 1]
                if name_color == 'White':
                    if R >= 0.8 and G >= 0.8 and B >= 0.8: 
                        df.loc[index, name_color] += 1
                elif name_color == 'Red':
                    if R >= 0.588 and G < 0.2 and B < 0.2: 
                        df.loc[index, name_color] += 1
                elif name_color == 'Light brown':
                    if 0.588 <= R <= 0.94 and 0.196 <= G <= 0.588 and 0 <= B < 0.392: 
                        df.loc[index, name_color] += 1
                elif name_color == 'Dark brown':
                    if 0.243 < R < 0.56 and 0 <= G < 0.392 and 0 < B < 0.392: 
                        df.loc[index, name_color] += 1
                elif name_color == 'Blue-gray':
                    if 0 <= R <= 0.588 and 0.392 <= G <= 0.588 and 0.490 <= B <= 0.588: 
                        df.loc[index, name_color] += 1
                elif name_color == 'Black':
                    if R <= 0.243 and G <= 0.243 and B <= 0.243: 
                        df.loc[index, name_color] += 1
                    
        # total pixels length of rows + length of columns, we use the 1st image for computation        
        total_pixels = len(pixels_color[0])+len(pixels_color[0][0])
        # check if more that percentage %
        if df.loc[index, name_color] >= (total_pixels*percentage):
            df.loc[index, name_color] = 1
        else:
            df.loc[index, name_color] = 0 

# compute colors, only 12 frist
for name_color in ['White','Red','Light brown','Dark brown','Blue-gray', 'Black']:
    color_percentage(df['Pixels'][0:13],name_color, 0.05)


df.loc[0:13,['Name','White','Red','Light brown','Dark brown','Blue-gray', 'Black']]
