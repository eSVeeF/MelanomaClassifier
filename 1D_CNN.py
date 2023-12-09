import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.metrics import Recall
from keras.optimizers import Adam
from read_images import ImageLoader
from sklearn.model_selection import train_test_split
import os
import importlib

# Load dataset
dataset = pd.read_csv('data.csv')
labels_dict = dict(zip(dataset['Name'], dataset['Label']))

# Load images from both directories
relative_dir = 'PH2Dataset'
image_loader_normal = ImageLoader(relative_dir + '/Custom Images/Normal')
image_loader_lesion = ImageLoader(relative_dir + '/Custom Images/Lesion')
labels = [labels_dict[image_loader_normal.bmp_files[i]] for i in range(len(image_loader_normal.bmp_files))]

# use target_size=(761, 553) in read_images.py
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

original_features = features.copy()

# Convert all elements to np.float64
features = [arr.astype(np.float64) for arr in features]

#Normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
# Apply MinMaxScaler to the last three variables
scaler = MinMaxScaler()

features_last_3 = [arr[-3:] for arr in features]

scaled_last_3 = list(scaler.fit_transform(features_last_3))

aux = features.copy()

features = [np.concatenate([arr[:-3], scaled_arr]) for arr, scaled_arr in zip(aux, scaled_last_3)]

# Convert features and labels to NumPy arrays
X = np.array(features)
y = np.array(labels)

import random
alll = []
for i in range(5):
    # Prepare data for ML model
    rand = random.randint(0,100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand)

    # Ensure labels are the correct shape
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # Define the input shape based on the reshaped data
    n_features = X_train.shape[1]
    input_shape = (n_features, 1)

    # Create a CNN model
    cnn_model = Sequential([
        Conv1D(32, kernel_size= 3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=1), # this is no pooling basically
        Dropout(0.25),

        Conv1D(64, kernel_size= 3, activation='relu'),
        MaxPooling1D(pool_size=1),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Use 'softmax' if you have more than two classes
    ])

    learning_rate = 0.001
    optimizer = Adam(learning_rate=learning_rate)

    cnn_model.compile(loss='binary_crossentropy',  # Use 'categorical_crossentropy' for more than two classes
                      optimizer=optimizer,
                      metrics=['accuracy', Recall()])

    # Since CNNs require input as an array of images, ensure X_train and X_test are correctly shaped
    # They should have the shape (num_images, img_height, img_width, num_channels)
    # You might need to reshape them and also normalize the pixel values (e.g., divide by 255)

    # Fit the CNN model
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    cnn_model.fit(X_train_reshaped, y_train, batch_size=32, epochs=180, validation_data=(X_test_reshaped, y_test))

    # Evaluate the classifier
    score = cnn_model.evaluate(X_test_reshaped, y_test)
    alll.append(score)
alll = np.array(alll)
print('this is the loss, accuracy and recall of the 5 runs\n', alll)
print('average recall', np.sum(alll[:,2])/len(alll))

"""print(f"Test Loss: {score[0]}")
print(f"Test Accuracy: {score[1]}")
print(f"Test Recall: {score[2]}")

# Predictions
predictions_prob = cnn_model.predict(X_test_reshaped)
predictions = (predictions_prob > 0.5)  # Thresholding probabilities to get binary classification
predictions = predictions.astype(int) #convert to 0 1

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# Flatten y_test if it has a shape like (n_samples, 1)
y_test = y_test.flatten()

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Confusion Matrix:\n{conf_matrix}")"""