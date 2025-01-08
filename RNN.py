import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten, Input
from tensorflow.keras.callbacks import EarlyStopping

# Load datasets
file_paths = {
    "xyz_distances": "xyz_distances.csv",
    "landmarks": "landmarks.csv",
    "angles": "angles.csv",
    "calculated_3d_distances": "calculated_3d_distances.csv",
    "labels": "labels.csv"
}

datasets = {name: pd.read_csv(path) for name, path in file_paths.items()}

# Merge datasets
angles_df = datasets["angles"]
labels_df = datasets["labels"]
xyz_distances_df = datasets["xyz_distances"]
calculated_3d_distances_df = datasets["calculated_3d_distances"]

merged_df = angles_df.merge(labels_df, on="vid_id")
merged_df = merged_df.merge(xyz_distances_df, on=["vid_id", "frame_order"])
merged_df = merged_df.merge(calculated_3d_distances_df, on=["vid_id", "frame_order"])

# Remove non-numeric columns
merged_df = merged_df.drop(columns=merged_df.select_dtypes(include=['object']).columns.difference(['class']))

# Prepare X and y
X = merged_df.drop(columns=["class"]).values  # Features
y = merged_df["class"].values  # Labels

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for sequence modeling
num_frames = 10  # Example sequence length
num_sequences = X_scaled.shape[0] // num_frames
X_reshaped = X_scaled[:num_sequences * num_frames].reshape(num_sequences, num_frames, -1)
y_reshaped = y[:num_sequences * num_frames:num_frames]

# KFold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_f1_scores = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X_reshaped)):
    print(f"Starting Fold {fold + 1}...")

    # Split data
    X_train, X_test = X_reshaped[train_idx], X_reshaped[test_idx]
    y_train, y_test = y_reshaped[train_idx], y_reshaped[test_idx]

    # Build CNN-LSTM model
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        Conv1D(64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        LSTM(64, return_sequences=True),
        Dropout(0.5),
        LSTM(64),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train model
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=32, callbacks=[early_stopping])

    # Evaluate model
    y_pred = np.argmax(model.predict(X_test), axis=1)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Fold {fold + 1} F1 Score: {f1}")
    fold_f1_scores.append(f1)

# Print overall results
print(f"Average F1 Score across folds: {np.mean(fold_f1_scores)}")
