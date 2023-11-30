import numpy as np
import pandas as pd
import os
import importlib
from sklearn.model_selection import train_test_split
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
            feature_builders.append(module)

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

# ROC Curve
y_scores = mlp_classifier.predict_proba(X_test)[:, 1]  # Probability estimates of the positive class
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()