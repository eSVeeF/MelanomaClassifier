import numpy as np
import pandas as pd
import os
import importlib
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
from read_images import ImageLoader

# Load dataset
dataset = pd.read_csv('mod_PH2_dataset.csv')
dataset['Name'] = dataset['Name'].str.strip() + '.bmp'
labels_dict = dict(zip(dataset['Name'], dataset['Label']))

# Load images from both directories
relative_dir = 'PH2Dataset'
image_loader_normal = ImageLoader(relative_dir + '/Custom Images/Normal')
image_loader_lesion = ImageLoader(relative_dir + '/Custom Images/Lesion')
labels = [labels_dict[image_loader_normal.bmp_files[i]] for i in range(len(image_loader_normal.bmp_files))]

# Feature Builders loading
feature_builders = []
for file in os.listdir('FeatureBuilders'):
    if file.endswith('.py'):
        module_name = file[:-3]
        module = importlib.import_module(f'FeatureBuilders.{module_name}')
        if getattr(module, 'READY', False):
            print("Loaded " + getattr(module, 'FEATURE_NAME', False) + " builder")
            feature_builders.append(module)

print("Loaded " + str(len(feature_builders)) + " feature builders")
# Feature extraction
features = []
for normal_image, lesion_image in zip(image_loader_normal.images_arrays, image_loader_lesion.images_arrays):
    flattened_image = normal_image.flatten()
    reduced_image = []
    for builder in feature_builders:
        image_type = getattr(builder, 'IMAGE_TYPE', 'NORMAL')
        if image_type == 'NORMAL':
            feature = builder.build(image_normal=normal_image)
        elif image_type == 'LESION':
            feature = builder.build(image_lesion=lesion_image)
        elif image_type == 'BOTH':
            feature = builder.build(normal_image, lesion_image)
        reduced_image = np.append(reduced_image, feature)
    features.append(reduced_image)

# Prepare data for ML model
X = np.array(features)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train and evaluate the classifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(7,), max_iter=1000, random_state=42)
mlp_classifier.fit(X_train, y_train)
predictions = mlp_classifier.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# Create a range of values for the number of neurons (1 to 100)
neuron_range = list(range(1, 101))

# Create configurations for one hidden layer
hidden_layer_sizes = [(n,) for n in neuron_range]

# Define parameter grid
param_grid = {
    'hidden_layer_sizes': hidden_layer_sizes,
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate_init': [0.001, 0.01],
    'max_iter': [1000]
}

# Create MLPClassifier object
mlp = MLPClassifier()

# Create GridSearchCV object
grid_search = GridSearchCV(mlp, param_grid, n_jobs=-1, cv=5, scoring='accuracy')

# Fit grid_search to the data
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best Parameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)

# Evaluate the best model
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Test Accuracy: {accuracy}")