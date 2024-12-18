# making array of tuples (string desc,id)
import pandas as pd
import torch

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


def convert_df_labels(df, column):
    df_cpy = df.copy()
    for lbl in _labels:
        df_cpy.loc[df[column] == lbl[0], column] = lbl[1]
    df_cpy[column] = pd.to_numeric(df_cpy[column])
    return df_cpy


def create_label_tensor(num_of_labels,df:pd.DataFrame,df_column):
    result = None
    for lbl in _labels:
        df.loc[df[df_column] == lbl[0], df_column] = lbl[1]
    df[df_column] = pd.to_numeric(df[df_column])
    for indexed_label in df[df_column]:
        temp = torch.zeros(1,num_of_labels)
        temp[0][indexed_label-1] = 1
        if result is None:
            result = temp
        else:
            result = torch.cat((result, temp),0)
    return result

def get_label(label_id):
    for lbl in _labels:
        if lbl[0] == label_id:
            return lbl[1]
    # if not defined
    return 'Undefined'
