import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.metrics import Recall
from keras.optimizers import Adam
from read_images import ImageLoader
from sklearn.model_selection import train_test_split

# Load dataset
dataset = pd.read_csv('mod_PH2_dataset.csv')
dataset['Name'] = dataset['Name'].str.strip() + '.bmp'
labels_dict = dict(zip(dataset['Name'], dataset['Label']))

# Load images from both directories
relative_dir = 'PH2Dataset'
image_loader_normal = ImageLoader(relative_dir + '/Custom Images/Normal')
image_loader_lesion = ImageLoader(relative_dir + '/Custom Images/Lesion')
labels = [labels_dict[image_loader_normal.bmp_files[i]] for i in range(len(image_loader_normal.bmp_files))]

# No flattening should be done for CNNs as they need the 2D structure of the image
features = []
for normal_image, lesion_image in zip(image_loader_normal.images_arrays, image_loader_lesion.images_arrays):
    # Normalize pixel values to be between 0 and 1
    normalized_image = normal_image / 255.0
    features.append(normalized_image)

# Convert features and labels to NumPy arrays
X = np.array(features)
y = np.array(labels)

# Prepare data for ML model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure labels are the correct shape
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Define the input shape based on the reshaped data
input_shape = X_train.shape[1:]  # This will be (100, 100, 3) if your images are 100x100 RGB

# Create a CNN model
cnn_model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
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
cnn_model.fit(X_train, y_train, batch_size=32, epochs=90, validation_data=(X_test, y_test))

# Evaluate the classifier
score = cnn_model.evaluate(X_test, y_test)
print(f"Test Loss: {score[0]}")
print(f"Test Accuracy: {score[1]}")
print(f"Test Recall: {score[2]}")

# Predictions
predictions = cnn_model.predict(X_test)
predictions = (predictions > 0.5)  # Thresholding probabilities to get binary classification
