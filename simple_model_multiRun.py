import random

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import label_indexer as li

# defines and classes

# constants
num_of_classes = 10
num_epochs = 10000
train_split = 0.8
learning_rate = 0.0001
seed_number = 495
run_seeded = False
n_rounding = 5


# end of constants

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
        # my_probabilities = self.softmax(logits)  # Apply softmax to logits
        return logits


def model_run(data_set, number_of_features):
    generator1 = torch.Generator()
    if run_seeded:
        generator1.manual_seed(seed_number)
    train_dataset, test_dataset = torch.utils.data.random_split(data_set, [train_split, 1 - train_split], )

    model = SoftmaxClassifier(number_of_features, num_of_classes)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        logits = model(train_dataset[:][0])

        loss = loss_fn(logits, train_dataset[:][1])
        loss.backward()

        optimizer.step()
        # if epoch % 1000 == 0:
        #     print(f'Epoch {epoch}.')

    # getting results
    with torch.no_grad():
        targets = torch.argmax(test_dataset[:][1], dim=1)
        test_logits = model(test_dataset[:][0])
        loss = loss_fn(test_logits, test_dataset[:][1])
        ans = torch.argmax(test_logits, dim=1)
        corrects = (ans == targets)
        accuracy = corrects.sum().float() / float(targets.size(0))
        loss_num = loss.item()
        accuracy_num = accuracy.item()
        print(f"this is the loss of test: {round(loss.item(), n_rounding)}")
        print(f"this is the accuracy of test: {round(accuracy.item(), n_rounding)}")

    # data clearing
    del optimizer, loss_fn, loss, test_logits, test_dataset, train_dataset, model, targets
    del ans, corrects, accuracy

    return loss_num, accuracy_num


# end of defines and classes


if run_seeded:
    torch.manual_seed(seed_number)

# הבאת המידע מאקסל ואיחוד פיצ'רים
distances_3d = pd.read_csv('data/3d_distances.csv', skip_blank_lines=True)
angles = pd.read_csv('data/angles.csv', skip_blank_lines=True)
xyz_distances = pd.read_csv('data/xyz_distances.csv', skip_blank_lines=True)
labels = pd.read_csv('data/labels.csv', skip_blank_lines=True)

# איחוד מידע של הפיצ'רים לאוסף אחד
data = distances_3d.join(angles.set_index('pose_id'), on='pose_id', how='left')
data = data.join(xyz_distances.set_index('pose_id'), on='pose_id', how='left')

# עיבוד מידע להתאים ללמידה
data = data.drop('pose_id', axis=1)
labels = labels.drop('pose_id', axis=1)
labels_tensor = li.create_label_tensor(num_of_classes, labels, 'pose')
data_tensor = df_to_tensor(data)
dataset = torch.utils.data.TensorDataset(data_tensor, labels_tensor)

num_of_features = data.columns.size

all_loss = []
all_accuracy = []

for i in range(10):
    print(f"run number: {i}")
    loss, acc = model_run(dataset, num_of_features)
    all_loss.append(round(loss, n_rounding))
    all_accuracy.append(round(acc, n_rounding))

print('loss:')
print(*all_loss, sep='\n')
print('accuracy:')
print(*all_accuracy, sep='\n')
