import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

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
X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Reshape data for CNN with additional dimension
X_train_cnn = X_train_scaled.reshape(-1, X_train_scaled.shape[1], 1)
X_val_cnn = X_val_scaled.reshape(-1, X_val_scaled.shape[1], 1)
X_test_cnn = X_test_scaled.reshape(-1, X_test_scaled.shape[1], 1)

# Build the CNN model
model = Sequential([
    Input(shape=(X_train_cnn.shape[1], 1)),  # Input layer
    Conv1D(32, kernel_size=3, activation='relu', padding='same'),
    MaxPooling1D(pool_size=2, padding='same'),
    Conv1D(64, kernel_size=3, activation='relu', padding='same'),
    MaxPooling1D(pool_size=2, padding='same'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')  # Output layer
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the CNN model
history = model.fit(
    X_train_cnn, y_train,
    validation_data=(X_val_cnn, y_val),
    epochs=20, batch_size=16,
    callbacks=[early_stopping]
)

# Evaluate the model on the test set
y_test_pred = np.argmax(model.predict(X_test_cnn), axis=1)
test_f1_score = f1_score(y_test, y_test_pred, average='weighted')
test_report = classification_report(y_test, y_test_pred, target_names=label_encoder.classes_)

# Print the test results
print(f"F1 Score (Test Set): {test_f1_score}")
print("\nClassification Report (Test Set):\n")
print(test_report)