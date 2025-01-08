import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Updated file path to labels.csv
# labels_file_path = r'C:\Users\shmue\Documents\GitHub\Fitness-Pose-Classification\data\labels.csv'
labels_file_path = 'data/labels.csv'

# Load the labels data
labels = pd.read_csv(labels_file_path)

# Check if the file loaded correctly
if labels.empty:
    print("The labels file is empty. Please check the file path.")
else:
    # Step 1: Find the most frequent class (majority class)
    most_frequent_pose = labels['pose'].value_counts().idxmax()

    # Step 2: Create baseline predictions (predict majority class for all samples)
    baseline_predictions = [most_frequent_pose] * len(labels)

    # Step 3: Extract actual labels
    actual_labels = labels['pose']

    # Step 4: Evaluate baseline model
    accuracy = accuracy_score(actual_labels, baseline_predictions)
    precision = precision_score(actual_labels, baseline_predictions, average='weighted', zero_division=0)
    recall = recall_score(actual_labels, baseline_predictions, average='weighted', zero_division=0)
    f1_score = f1_score(actual_labels, baseline_predictions, average='weighted', zero_division=0)

    # Step 5: Print results
    print(f"Baseline Model Results:")
    print(f"  Most Frequent Pose: {most_frequent_pose}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1_score:.4f}")

    # Step 6: Optional - Show class distribution
    print("\nClass Distribution:")
    print(labels['pose'].value_counts())
