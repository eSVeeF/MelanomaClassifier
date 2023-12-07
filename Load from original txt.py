import os
import pandas as pd
# Load dataset
# Specify the path to the folder containing bmp files
# Define the relative path
relative_dir = 'PH2Dataset'
current_directory = os.getcwd()
full_path = os.path.join(current_directory, relative_dir)
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

# read text file into pandas DataFrame
dataset = pd.read_csv(full_path +"/mod_PH2_dataset.txt", sep="|")

#keep only the name and the label because the paper only uses the images
dataset = dataset.iloc[:,[1,3]]

#rename columns
dataset.columns = ['Name', 'Label']
# strip
dataset['Name'] = dataset['Name'].str.strip() + '.bmp'
# Extract the number of the Name
dataset['Number'] = dataset['Name'].str.extract('(\d+)').astype(int)
# Sort the DataFrame based on the 'Number' to match with the order of ImageLoader
dataset = dataset.sort_values(by='Number')
# create dict
labels_dict = dict(zip(dataset['Name'], dataset['Label']))
