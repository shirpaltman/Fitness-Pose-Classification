import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Load datasets
features_distances = pd.read_csv('../data/3d_distances.csv')
features_angles = pd.read_csv('../data/angles.csv')
labels = pd.read_csv('../data/labels.csv')


# Load datasets
distances_3d = pd.read_csv('../data/3d_distances.csv')
angles = pd.read_csv('../data/angles.csv')
xyz_distances = pd.read_csv('../data/xyz_distances.csv')
landmarks = pd.read_csv('../data/landmarks.csv')
labels = pd.read_csv('../data/labels.csv')

# Merge features (drop `pose_id` to avoid duplications)
features = pd.concat([
    distances_3d.iloc[:, 1:],  # 3D distances
    angles.iloc[:, 1:],       # Angles
    xyz_distances.iloc[:, 1:],  # Per-axis distances
    landmarks.iloc[:, 1:]     # Raw coordinates
], axis=1)

# Encode the labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels['pose'])
one_hot_labels = to_categorical(encoded_labels)  # Convert to one-hot for softmax

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, one_hot_labels, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the softmax model
model = Sequential([
    Dense(units=128, input_dim=X_train.shape[1], activation='relu'),  # Increased units for richer data
    Dense(units=64, activation='relu'),
    Dense(units=10, activation='softmax')  # Adjust units to match the number of classes
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2, verbose=1)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.2f}")
print(f"Test Loss: {test_loss:.2f}")

# Plot training vs validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()

# Plot training vs validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()