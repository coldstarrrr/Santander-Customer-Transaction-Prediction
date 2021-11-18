import pandas as pd
from utils.preprocessing import transform
from utils.fileutils import get_path
import os

def load_and_preprocess_data():
    train_data = pd.read_csv(get_path(r"data\train.csv"))
    test_data = pd.read_csv(get_path(r"data\test.csv"))

    train_X = train_data.drop(columns=['ID_code', 'target'])
    train_y = train_data['target']
    test_X = test_data.drop(columns=['ID_code'])

    return transform(train_X), train_y, transform(test_X), test_data['ID_code']
