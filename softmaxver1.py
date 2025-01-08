import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch import nn, optim
from sklearn.metrics import f1_score, classification_report

# Define file paths (assuming files are in the same directory as the script)
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

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Define a neural network with explicit softmax
class PoseClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PoseClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=1)  # Explicit softmax layer
        )

    def forward(self, x):
        return self.fc(x)

# Initialize model, loss, and optimizer
input_dim = X_train_scaled.shape[1]
output_dim = len(label_encoder.classes_)
model = PoseClassifier(input_dim, output_dim)
criterion = nn.NLLLoss()  # Use negative log likelihood
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50
for epoch in range(epochs):
    # Training
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(torch.log(outputs), y_train_tensor)  # Apply log before NLLLoss
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(torch.log(val_outputs), y_val_tensor)
        val_preds = torch.argmax(val_outputs, dim=1)
        val_f1 = f1_score(y_val_tensor.numpy(), val_preds.numpy(), average='weighted')
        val_accuracy = (val_preds.numpy() == y_val_tensor.numpy()).mean()

    # Print epoch metrics
    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val F1 Score: {val_f1:.4f} | Val Accuracy: {val_accuracy:.4f}\n")

# Final evaluation on the test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_preds = torch.argmax(test_outputs, dim=1)
    test_f1 = f1_score(y_test_tensor.numpy(), test_preds.numpy(), average='weighted')
    test_report = classification_report(y_test_tensor.numpy(), test_preds.numpy(), target_names=label_encoder.classes_)

print("Final Evaluation on Test Set:")
print(f"F1 Score (Test): {test_f1}")
print("\nClassification Report (Test):\n")
print(test_report)
