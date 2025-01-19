import random
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torcheval.metrics.functional import multiclass_f1_score, multiclass_accuracy
import matplotlib.pyplot as plt

# defines and classes

_labels = [
    ('jumping_jacks_down', 0),
    ('jumping_jacks_up', 1),
    ('pullups_down', 2),
    ('pullups_up', 3),
    ('pushups_down', 4),
    ('pushups_up', 5),
    ('situp_down', 6),
    ('situp_up', 7),
    ('squats_down', 8),
    ('squats_up', 9)
]


def create_label_tensor(num_of_labels, df: pd.DataFrame, df_column):
    result = None
    for lbl in _labels:
        df.loc[df[df_column] == lbl[0], df_column] = lbl[1]
    df[df_column] = pd.to_numeric(df[df_column])
    for indexed_label in df[df_column]:
        temp = torch.zeros(1, num_of_labels)
        temp[0][indexed_label] = 1
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
        self.hidden1 = nn.Linear(input_dim, 100)
        self.hidden2 = nn.Linear(100, 50)
        self.linear = nn.Linear(50, output_dim)  # Single linear layer
        self.softmax = nn.Softmax(dim=1)  # Apply softmax along class dimension

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        logits = self.linear(x)
        # my_probabilities = self.softmax(logits)  # Apply softmax to logits
        return logits


# end of defines and classes

# constants- משתנים עקביים
num_of_classes = 10
n_rounding = 5  # number rounding to the nth zero
seed_number = 495
run_seeded = True

# model params- המשתנים אשר משנים את איך המודל ירוץ
num_epochs = 250
train_part = 0.7
val_part = 0.15
test_part = 0.15
learning_rate = 0.0025

# end of constants

if run_seeded:
    torch.manual_seed(seed_number)

# הבאת המידע מאקסל ואיחוד פיצ'רים
distances_3d = pd.read_csv('data/3d_distances.csv', skip_blank_lines=True)
angles = pd.read_csv('data/angles.csv', skip_blank_lines=True)
xyz_distances = pd.read_csv('data/xyz_distances.csv', skip_blank_lines=True)
landmarks = pd.read_csv('data/landmarks.csv', skip_blank_lines=True)
labels = pd.read_csv('data/labels.csv', skip_blank_lines=True)

features_entered = ",".join(["3d", "angle", "xyz", "landmarks"])

# איחוד מידע של הפיצ'רים לאוסף אחד
features = pd.concat([
    # distances_3d.iloc[:, 1:],  # 3D distances
    angles.iloc[:, 1:],  # Angles
    xyz_distances.iloc[:, 1:],  # Per-axis distances
    landmarks.iloc[:, 1:]  # Raw coordinates
], axis=1)

data = features

# עיבוד מידע להתאים ללמידת סופטמקס
# data = data.drop('pose_id', axis=1)
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
train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                                [train_part, val_part, test_part],
                                                                                generator=generator1)

model = SoftmaxClassifier(num_of_features, num_of_classes)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# אימון מודל
train_loss = torch.tensor(0.)
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    logits = model(train_dataset[:][0])

    train_loss = loss_fn(logits, train_dataset[:][1])
    train_loss.backward()

    optimizer.step()

    train_losses.append(train_loss.item())

    train_targets = torch.argmax(train_dataset[:][1], dim=1)
    train_accuracy = multiclass_accuracy(input=logits, target=train_targets)
    train_accuracies.append(train_accuracy.item())

    model.eval()
    with torch.no_grad():
        logits = model(validation_dataset[:][0])
        val_loss = loss_fn(logits, validation_dataset[:][1])
        val_losses.append(val_loss.item())

        val_targets = torch.argmax(validation_dataset[:][1], dim=1)
        val_accuracy = multiclass_accuracy(target=val_targets, input=logits)
        val_accuracies.append(val_accuracy.item())
    # if epoch % 1000 == 0:
    #     print(f'Epoch {epoch}.')

# בחינת המודל והדפסת תוצאות
with torch.no_grad():
    targets = torch.argmax(test_dataset[:][1], dim=1)
    test_logits = model(test_dataset[:][0])

    test_loss = loss_fn(test_logits, test_dataset[:][1])
    f1_score = multiclass_f1_score(input=test_logits, target=targets, average='weighted', num_classes=num_of_classes)
    accuracy = multiclass_accuracy(input=test_logits, target=targets, num_classes=num_of_classes)

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
    print(f"{train_loss_num}\t{learning_rate}\t{num_epochs}\t{num_of_features}\t{features_entered}")

plt.figure(figsize=(12, 6))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
test_results_text = f"Test Loss: {test_loss_num}\n"
test_results_text += f"Test Accuracy: {accuracy_num}\n"
test_results_text += f"Test F1 Score: {f1_score_num}"

# Add text box with test results
plt.text(1.05, 0.5, test_results_text, fontsize=12, transform=plt.gca().transAxes,
         verticalalignment='center', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.6))

plt.tight_layout()
plt.show()
# data clearing
# del optimizer, loss_fn, curr_loss, test_logits, test_dataset, train_dataset, model, targets
# del ans, corrects, accuracy
