import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.metrics import Recall
from keras.optimizers.legacy import Adam
from read_images import ImageLoader
from sklearn.model_selection import train_test_split, KFold
from keras.callbacks import EarlyStopping

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

num_folds = 5
kfold = KFold(n_splits=num_folds, shuffle=True)

# List to store results from each fold
fold_results = []

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(X, y):

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

    # Compile the model
    cnn_model.compile(loss='binary_crossentropy',
                      optimizer=Adam(learning_rate=0.001),
                      metrics=['accuracy', Recall()])

    # Ensure labels are the correct shape for the fold
    y_train_fold = y[train].reshape(-1, 1)
    y_test_fold = y[test].reshape(-1, 1)

    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    early_stopping = EarlyStopping(monitor='loss', patience=10, mode='min')

    # Fit data to model
    history = cnn_model.fit(X[train], y[train].reshape(-1, 1),
                  batch_size=32,
                  epochs=100,
                  verbose=1,
                  callbacks=[early_stopping])

    # Evaluate the model on the test data
    scores = cnn_model.evaluate(X[test], y[test].reshape(-1, 1), verbose=0)

    # Append the scores to the results list
    fold_results.append(scores)

    print(f'Score for fold {fold_no}: {cnn_model.metrics_names[0]} of {scores[0]}; {cnn_model.metrics_names[1]} of {scores[1]*100}%; {cnn_model.metrics_names[2]} of {scores[2]*100}%')
    
    fold_no += 1

print(fold_results)

average_results = np.mean(fold_results, axis=0)
std_dev_results = np.std(fold_results, axis=0)
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
for i in range(len(cnn_model.metrics_names)):
    print(f'{cnn_model.metrics_names[i]}: {average_results[i]} (± {std_dev_results[i]})')