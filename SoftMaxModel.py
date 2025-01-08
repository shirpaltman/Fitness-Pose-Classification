import random

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torcheval.metrics.functional import multiclass_f1_score

# defines and classes

_labels = [
    ('jumping_jacks_down', 1),
    ('jumping_jacks_up', 2),
    ('pullups_down', 3),
    ('pullups_up', 4),
    ('pushups_down', 5),
    ('pushups_up', 6),
    ('situp_down', 7),
    ('situp_up', 8),
    ('squats_down', 9),
    ('squats_up', 10)
]


def create_label_tensor(num_of_labels, df: pd.DataFrame, df_column):
    result = None
    for lbl in _labels:
        df.loc[df[df_column] == lbl[0], df_column] = lbl[1]
    df[df_column] = pd.to_numeric(df[df_column])
    for indexed_label in df[df_column]:
        temp = torch.zeros(1, num_of_labels)
        temp[0][indexed_label - 1] = 1
        if result is None:
            result = temp
        else:
            result = torch.cat((result, temp), 0)
    return result


def get_label(label_id):
    for lbl in _labels:
        if lbl[0] == label_id:
            return lbl[1]
    # if not defined
    return 'Undefined'


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

# end of defines and classes


# constants- המשתנים אשר משנים את איך המודל ירוץ
num_of_classes = 10
num_epochs = 50000
train_part = 0.8
test_part = 0.2
learning_rate = 0.001
seed_number = 495
run_seeded = True
n_rounding = 5  # number rounding to the nth zero


# end of constants

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

# עיבוד מידע להתאים ללמידת סופטמקס
data = data.drop('pose_id', axis=1)
labels = labels.drop('pose_id', axis=1)
labels_tensor = create_label_tensor(num_of_classes, labels, 'pose')
data_tensor = df_to_tensor(data)
dataset = torch.utils.data.TensorDataset(data_tensor, labels_tensor)

# כמות הפיצ'רים
num_of_features = data.columns.size

# קביעת רנדומליות לעקביות
generator1 = torch.Generator()
if run_seeded:
    generator1.manual_seed(seed_number)
else:
    generator1 = torch.default_generator

# חלוקת המידע ל-train & test
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_part, test_part], generator=generator1)

model = SoftmaxClassifier(num_of_features, num_of_classes)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# אימון מודל
train_loss = torch.tensor(0.)
for epoch in range(num_epochs):
    optimizer.zero_grad()
    logits = model(train_dataset[:][0])

    train_loss = loss_fn(logits, train_dataset[:][1])
    train_loss.backward()

    optimizer.step()
    # if epoch % 1000 == 0:
    #     print(f'Epoch {epoch}.')

# בחינת המודל והדפסת תוצאות
with torch.no_grad():

    targets = torch.argmax(test_dataset[:][1], dim=1)
    test_logits = model(test_dataset[:][0])
    test_loss = loss_fn(test_logits, test_dataset[:][1])

    ans = torch.argmax(test_logits, dim=1)
    corrects = (ans == targets)
    f1_score = multiclass_f1_score(targets,ans,average='weighted',num_classes=10)

    accuracy = corrects.sum().float() / float(targets.size(0))
    train_loss_num = round(train_loss.item(), n_rounding)
    test_loss_num = round(test_loss.item(), n_rounding)
    accuracy_num = round(accuracy.item(), n_rounding)
    f1_score_num = round(f1_score.item(), n_rounding)

    # הדפסת תוצאות של מודל
    print(f"this is the training data loss of test: {train_loss_num}")
    print(f"this is the test loss of test: {test_loss_num}")
    print(f"this is the accuracy of test: {accuracy_num}")
    print(f"this is the f1_score of test: {f1_score_num}")

    # הדפסת מידע על ההרצה עצמה. נועד בשביל השוואת פרמטרים של המודל.
    print(f"{f1_score_num}\t{round(accuracy_num, n_rounding)}\t{test_loss_num}\t", end="")
    print(f"{train_loss_num}\t{learning_rate}\t{num_epochs}\t{num_of_features}")

# data clearing
# del optimizer, loss_fn, curr_loss, test_logits, test_dataset, train_dataset, model, targets
# del ans, corrects, accuracy
