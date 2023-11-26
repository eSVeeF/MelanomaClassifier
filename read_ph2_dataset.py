import os

# Specify the path to the folder containing bmp files
# Define the relative path
relative_path = 'MelanomaClassifier'
current_directory = os.getcwd()
full_path = os.path.join(current_directory, relative_path)
full_path = os.path.abspath(full_path)

# Read the content of the file
with open(full_path + "/PH2_dataset.txt", 'r') as file:
    lines = file.readlines()

# Replace "||" with "|" for smoother read
modified_lines = [line.replace('||', '|') for line in lines]

# Read up to line 201, after 201 is just the legends
modified_content = ''.join(modified_lines[:201])

# Write the modified content back to the file
with open(full_path + "/mod_PH2_dataset.txt", 'w') as file:
    file.write(modified_content)

import pandas as pd   
# read text file into pandas DataFrame 
df = pd.read_csv(full_path +"/mod_PH2_dataset.txt", sep="|") 

#drop mistaken column
df = df.drop(df.columns[0], axis=1)
df = df.drop(df.columns[10], axis=1)

#rename columns
df.columns = ['Name', 'Histological Diagnosis', 'Clinical Diagnosis',
       'Asymmetry', 'Pigment Network', 'Dots/Globules', 'Streaks',
       'Regression Areas', 'Blue-Whitish Veil', 'Colors']

#Remove white spaces
aux = ['Name', 'Histological Diagnosis', 'Pigment Network', 'Dots/Globules', 'Streaks',
       'Regression Areas', 'Blue-Whitish Veil', 'Colors']

for column in aux: 
    df[column] = df[column].str.strip()
df