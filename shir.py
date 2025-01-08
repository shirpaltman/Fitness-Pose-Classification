import random
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import label_indexer as li
from sklearn.metrics import f1_score

# Constants
num_of_classes = 10
num_epochs = 5000
trn_prt = 0.8  # Training portion (as percentage)
learning_rate = 0.0001
seed_number = 495
run_seeded = True
n_rounding = 5
train_test_loops = 1

# Determine the supported device
def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Convert a DataFrame to a PyTorch tensor
def df_to_tensor(df):
    device = get_device()
    return torch.from_numpy(df.values).float().to(device)

# Define the softmax classification model
class SoftmaxClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SoftmaxClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# Main function to run the model
def model_run(data_set, number_of_features):
    # Split dataset into training and testing
    dataset_length = len(data_set)
    train_length = int(trn_prt * dataset_length)
    test_length = dataset_length - train_length
    generator1 = torch.Generator()
    if run_seeded:
        generator1.manual_seed(seed_number)

    train_dataset, test_dataset = torch.utils.data.random_split(
        data_set, [train_length, test_length], generator=generator1
    )

    # Initialize model, loss function, and optimizer
    model = SoftmaxClassifier(number_of_features, num_of_classes).to(get_device())
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Prepare train data
    train_data, train_labels = map(torch.stack, zip(*train_dataset))
    train_data = train_data.to(get_device())
    train_labels = train_labels.to(get_device())

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        train_logits = model(train_data)
        curr_loss = loss_fn(train_logits, train_labels)
        curr_loss.backward()
        optimizer.step()

    # Evaluate model on test data
    with torch.no_grad():
        test_data, test_labels = map(torch.stack, zip(*test_dataset))
        test_data = test_data.to(get_device())
        test_labels = test_labels.to(get_device())

        test_logits = model(test_data)
        test_loss = loss_fn(test_logits, test_labels)

        predictions = torch.argmax(test_logits, dim=1).cpu().numpy()
        true_labels = test_labels.cpu().numpy()

        corrects = (predictions == true_labels)
        accuracy = corrects.sum().item() / len(corrects)

        # Compute F1 score
        f1 = f1_score(true_labels, predictions, average='weighted')

        print(f"Train Loss: {round(curr_loss.item(), n_rounding)}")
        print(f"Test Loss: {round(test_loss.item(), n_rounding)}")
        print(f"Accuracy: {round(accuracy, n_rounding)}")
        print(f"F1 Score: {round(f1, n_rounding)}")

    return test_loss.item(), accuracy, curr_loss.item(), f1

# Load data from CSV files
distances_3d = pd.read_csv('data/3d_distances.csv', skip_blank_lines=True)
angles = pd.read_csv('data/angles.csv', skip_blank_lines=True)
xyz_distances = pd.read_csv('data/xyz_distances.csv', skip_blank_lines=True)
landmarks = pd.read_csv('data/landmarks.csv', skip_blank_lines=True)
labels = pd.read_csv('data/labels.csv', skip_blank_lines=True)

# Debugging: Inspect columns
print(f"Columns in xyz_distances: {xyz_distances.columns}")
print(f"Shape of xyz_distances: {xyz_distances.shape}")

# Ensure 'pose_id' exists or adjust the column name
if 'pose_id' not in xyz_distances.columns:
    print("Warning: 'pose_id' not found in xyz_distances. Available columns:", xyz_distances.columns)
    # Adjust column name if needed
    # xyz_distances.rename(columns={'ActualColumnName': 'pose_id'}, inplace=True)

# Merge features into a single dataset
if 'pose_id' in xyz_distances.columns:
    data = angles
    data = data.join(xyz_distances.set_index('pose_id'), on='pose_id', how='left')
    data = data.join(landmarks.set_index('pose_id'), on='pose_id', how='left')
else:
    print("Error: 'pose_id' column is required for merging but was not found.")
    # Optionally skip merging or handle this case differently

# Prepare data for learning
if 'pose_id' in data.columns:
    data = data.drop('pose_id', axis=1)
labels = labels.drop('pose_id', axis=1)

# Create tensors
labels_tensor = li.create_label_tensor(num_of_classes, labels, 'pose')
data_tensor = df_to_tensor(data)

# Debugging: Check shapes
print(f"Data Tensor Shape: {data_tensor.shape}")
print(f"Labels Tensor Shape: {labels_tensor.shape}")

# Convert one-hot encoded labels to class indices
if labels_tensor.dim() > 1:  # Check if labels are one-hot encoded
    labels_tensor = torch.argmax(labels_tensor, dim=1)

# Create TensorDataset
dataset = torch.utils.data.TensorDataset(data_tensor, labels_tensor)

# Check number of features
num_of_features = data.shape[1]

# Run the model
model_run(dataset, num_of_features)
