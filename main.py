import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

# Define file paths (assuming the files are in the same directory as the script)
file_paths = {
    "xyz_distances": "xyz_distances.csv",
    "landmarks": "landmarks.csv",
    "angles": "angles.csv",
    "calculated_3d_distances": "calculated_3d_distances.csv",
    "labels": "labels.csv"
}

# Load datasets
datasets = {name: pd.read_csv(path) for name, path in file_paths.items()}

# Load angles.csv and labels.csv
angles_df = datasets["angles"]
labels_df = datasets["labels"]

# Merge angles with labels
merged_df = angles_df.merge(labels_df, on="vid_id")

# Prepare features and labels
X = merged_df.iloc[:, 2:-1].values  # All angle columns
y = merged_df["class"].values  # Target labels

# Encode target labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Build and train a Softmax (Logistic Regression) model
softmax_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500, random_state=42)
softmax_model.fit(X_train_scaled, y_train)

# Evaluate the model on the validation set
y_val_pred = softmax_model.predict(X_val_scaled)
val_f1_score = f1_score(y_val, y_val_pred, average='weighted')
val_report = classification_report(y_val, y_val_pred, target_names=label_encoder.classes_)

# Print the results
print(f"F1 Score (Validation): {val_f1_score}")
print("\nClassification Report (Validation):\n")
print(val_report)
