import numpy as np
from read_images import ImageLoader
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load your images and labels
image_loader = ImageLoader('PH2Dataset/Custom Images/Normal')

# Assuming you have labels for your images (0 for benign, 1 for malignant)
# This is just a placeholder, replace it with your actual labels
labels = np.random.randint(2, size=len(image_loader.bmp_files))

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
