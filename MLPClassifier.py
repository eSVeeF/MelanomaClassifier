import numpy as np
from read_images import ImageLoader
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load your dataset with image names and ground truth labels
dataset = pd.read_csv('mod_PH2_dataset.csv')

# Trim leading and trailing whitespaces from the "Name" column
dataset['Name'] = dataset['Name'].str.strip()
# Add the ".bmp" extension to the "Name" column
dataset['Name'] = dataset['Name'] + '.bmp'

# Create a dictionary to map image names to labels
labels_dict = dict(zip(dataset['Name'], dataset['Label']))

# Load your images using the ImageLoader
image_loader = ImageLoader('PH2Dataset/Custom Images/Normal')

# Create a list of labels based on the image names
labels = [labels_dict[image_loader.bmp_files[i]] for i in range(len(image_loader.bmp_files))]

# Load your images and labels
image_loader = ImageLoader('PH2Dataset/Custom Images/Normal')

# Flatten the image arrays for each image
flattened_images = [image.flatten() for image in image_loader.images_arrays]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(flattened_images, labels, test_size=0.2, random_state=42)

# Initialize the MLP Classifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

# Train the classifier
mlp_classifier.fit(X_train, y_train)

# Make predictions on the test set
predictions = mlp_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# Create a confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)

# Display the confusion matrix using seaborn
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()