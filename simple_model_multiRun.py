import random

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import label_indexer as li

# defines and classes

# constants
num_of_classes = 10
num_epochs = 100000
trn_prt = 0.8
learning_rate = 0.0001
seed_number = 495
run_seeded = True
n_rounding = 5
train_test_loops = 1



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
    else:
        generator1 = torch.default_generator

    train_dataset, test_dataset = torch.utils.data.random_split(data_set, [trn_prt, 1 - trn_prt], generator=generator1)

    model = SoftmaxClassifier(number_of_features, num_of_classes)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    curr_loss = 0
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        logits = model(train_dataset[:][0])

        curr_loss = loss_fn(logits, train_dataset[:][1])
        curr_loss.backward()

        optimizer.step()
        # if epoch % 1000 == 0:
        #     print(f'Epoch {epoch}.')

    # getting results
    with torch.no_grad():
        train_loss = curr_loss.item()

        targets = torch.argmax(test_dataset[:][1], dim=1)
        test_logits = model(test_dataset[:][0])
        test_loss = loss_fn(test_logits, test_dataset[:][1])

        ans = torch.argmax(test_logits, dim=1)
        corrects = (ans == targets)

        accuracy = corrects.sum().float() / float(targets.size(0))
        loss_num = test_loss.item()
        accuracy_num = accuracy.item()

        print(f"this is the training data loss of test: {round(train_loss, n_rounding)}")
        print(f"this is the test loss of test: {round(test_loss.item(), n_rounding)}")
        print(f"this is the accuracy of test: {round(accuracy.item(), n_rounding)}")
        print(f"{round(accuracy.item(), n_rounding)}\t{round(test_loss.item(), n_rounding)}\t", end="")
        print(f"{round(train_loss, n_rounding)}\t{learning_rate}\t{num_epochs}\t{number_of_features}")

    # data clearing
    # del optimizer, loss_fn, curr_loss, test_logits, test_dataset, train_dataset, model, targets
    # del ans, corrects, accuracy

    return loss_num, accuracy_num, train_loss


# end of defines and classes


if run_seeded:
    torch.manual_seed(seed_number)

# הבאת המידע מאקסל ואיחוד פיצ'רים
distances_3d = pd.read_csv('data/3d_distances.csv', skip_blank_lines=True)
angles = pd.read_csv('data/angles.csv', skip_blank_lines=True)
xyz_distances = pd.read_csv('data/xyz_distances.csv', skip_blank_lines=True)
landmarks = pd.read_csv('data/landmarks.csv', skip_blank_lines=True)
labels = pd.read_csv('data/labels.csv', skip_blank_lines=True)

# איחוד מידע של הפיצ'רים לאוסף אחד
# data = distances_3d
data = angles
# data = data.join(angles.set_index('pose_id'), on='pose_id', how='left')
data = data.join(xyz_distances.set_index('pose_id'), on='pose_id', how='left')
data = data.join(landmarks.set_index('pose_id'), on='pose_id', how='left')

# עיבוד מידע להתאים ללמידה
data = data.drop('pose_id', axis=1)
labels = labels.drop('pose_id', axis=1)
labels_tensor = li.create_label_tensor(num_of_classes, labels, 'pose')
data_tensor = df_to_tensor(data)
dataset = torch.utils.data.TensorDataset(data_tensor, labels_tensor)

num_of_features = data.columns.size

all_loss = []
all_accuracy = []
all_train_loss = []

model_run(dataset, num_of_features)

# for i in range(train_test_loops):
#     print(f"run number: {i+1}")
#     loss, acc, t_loss = model_run(dataset, num_of_features)
#     all_loss.append(round(loss, n_rounding))
#     all_accuracy.append(round(acc, n_rounding))
#     all_train_loss.append(round(t_loss, n_rounding))
