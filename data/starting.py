import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
# ytuhjgfdsfghjklkjhgfdsfghjk
# Load datasets
features_distances = pd.read_csv('/mnt/data/3d_distances.csv')
features_angles = pd.read_csv('/mnt/data/angles.csv')
labels = pd.read_csv('/mnt/data/labels.csv')
