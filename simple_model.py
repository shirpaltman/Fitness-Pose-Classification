import random

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import label_indexer as li


# defines and classes

# determine the supported device
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')  # don't have GPU
    return device


# convert a df to tensor to be used in pytorch
def df_to_tensor(df):
    device = get_device()
    return torch.from_numpy(df.values).float().to(device)


# Define the softmax classification model
class SoftmaxClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SoftmaxClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)  # Single linear layer
        self.softmax = nn.Softmax(dim=1)  # Apply softmax along class dimension

    def forward(self, x):
        logits = self.linear(x)
        my_probabilities = self.softmax(logits)  # Apply softmax to logits
        return my_probabilities


# end of defines and classes

# constants
num_of_classes = 10
num_epochs = 10
n = 500
# end of constants

# הבאת המידע מאקסל
distances_3d = pd.read_csv('data/3d_distances.csv', nrows=n, skip_blank_lines=True)
distances_3d = distances_3d.drop('pose_id', axis=1)
labels = pd.read_csv('data/labels.csv', nrows=n)
labels = labels.drop('pose_id', axis=1)
labels_indexed = li.create_label_tensor(num_of_classes, labels, 'pose')
# labels_indexed = li.convert_df_labels(labels, 'pose')
distances_3d_t = df_to_tensor(distances_3d)
# labels_t = df_to_tensor(labels_indexed)

num_of_features = distances_3d.columns.size

# יצירת משקלים ובאייס(bias)
W = torch.zeros((num_of_features, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

model = SoftmaxClassifier(num_of_features, num_of_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001)

for epoch in range(num_epochs):
    optimizer.zero_grad()
    probabilities = model(distances_3d_t)

    # nll = torch.sum(-labels_t * torch.log(probabilities))
    loss = torch.mean(-labels_indexed * torch.log(probabilities))
    loss.backward()

    optimizer.step()
