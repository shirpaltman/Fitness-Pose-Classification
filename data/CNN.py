import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import matplotlib.pyplot as plt

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# Load datasets
file_paths = {
    "3d_distances": "3d_distances.csv",
    "angles": "angles.csv",
    "labels": "labels.csv",
    "landmarks": "landmarks.csv",
    "xyz_distances": "xyz_distances.csv",
}

datasets = {name: pd.read_csv(path) for name, path in file_paths.items()}

# Merge all datasets on 'pose_id'
merged_df = datasets["labels"]
for name, data in datasets.items():
    if name != "labels":
        merged_df = merged_df.merge(data, on="pose_id")

# Identify and remove non-numeric columns excluding 'pose'
non_numeric_columns = merged_df.select_dtypes(include=['object']).columns
merged_df = merged_df.drop(columns=non_numeric_columns.difference(['pose']))

# Prepare features and labels
X = merged_df.drop(columns=["pose", "pose_id"]).values  # Exclude label and ID columns
y = merged_df["pose"].values  # Target labels

# Encode target labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into train (70%), validation (15%), and test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.3, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Reshape data for CNN with additional dimension
X_train_cnn = X_train_scaled.reshape(-1, X_train_scaled.shape[1], 1)
X_val_cnn = X_val_scaled.reshape(-1, X_val_scaled.shape[1], 1)
X_test_cnn = X_test_scaled.reshape(-1, X_test_scaled.shape[1], 1)

# One-hot encode the target labels
y_train_one_hot = to_categorical(y_train, num_classes=len(label_encoder.classes_))
y_val_one_hot = to_categorical(y_val, num_classes=len(label_encoder.classes_))
y_test_one_hot = to_categorical(y_test, num_classes=len(label_encoder.classes_))

# Build the CNN model with Batch Normalization, Dropout, and Weight Decay (L2 Regularization)
model = Sequential([
    Input(shape=(X_train_cnn.shape[1], 1)),  # Input layer

    # First Convolutional Block
    Conv1D(32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2, padding='same'),
    Dropout(0.2),

    # Second Convolutional Block
    Conv1D(64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2, padding='same'),
    Dropout(0.3),

    # Third Convolutional Block
    Conv1D(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2, padding='same'),
    Dropout(0.4),

    # Flatten layer
    Flatten(),

    # First Dense Layer
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.6),

    # Second Dense Layer
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.6),

    # Output Layer
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print the model summary
model.summary()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Train the CNN model
history = model.fit(
    X_train_cnn, y_train_one_hot,
    validation_data=(X_val_cnn, y_val_one_hot),
    epochs=100, batch_size=16,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate the model on the test set
y_test_pred = np.argmax(model.predict(X_test_cnn), axis=1)
test_f1_score = f1_score(np.argmax(y_test_one_hot, axis=1), y_test_pred, average='weighted')
test_report = classification_report(np.argmax(y_test_one_hot, axis=1), y_test_pred, target_names=label_encoder.classes_)

# Print the test results
print(f"F1 Score (Test Set): {test_f1_score}")
print("\nClassification Report (Test Set):\n")
print(test_report)

# Plot training & validation accuracy and loss
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix")
    plt.show()

# Call the functions to display the plots
plot_training_history(history)
plot_confusion_matrix(np.argmax(y_test_one_hot, axis=1), y_test_pred, label_encoder.classes_)
