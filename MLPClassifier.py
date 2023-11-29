import numpy as np
import pandas as pd
import os
import importlib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from read_images import ImageLoader

# Load dataset
dataset = pd.read_csv('mod_PH2_dataset.csv')
dataset['Name'] = dataset['Name'].str.strip() + '.bmp'
labels_dict = dict(zip(dataset['Name'], dataset['Label']))

# Load images
image_loader = ImageLoader('PH2Dataset/Custom Images/Normal')
labels = [labels_dict[image_loader.bmp_files[i]] for i in range(len(image_loader.bmp_files))]

# Feature Builders: Dinamically loading all files in FeatureBuilders folder
feature_builders = []
for file in os.listdir('FeatureBuilders'):
    if file.endswith('.py'):
        module_name = file[:-3]
        module = importlib.import_module(f'FeatureBuilders.{module_name}')
        if getattr(module, 'READY', False):
            feature_builders.append(module)

# Feature extraction
features = []
for image in image_loader.images_arrays:
    flattened_image = image.flatten()
    for builder in feature_builders:
        feature = builder.build(image)
        flattened_image = np.append(flattened_image, feature)
    features.append(flattened_image)

# Prepare data for ML model
X = np.array(features)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train and evaluate the classifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
mlp_classifier.fit(X_train, y_train)
predictions = mlp_classifier.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
