import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from utils.dataset import load_and_preprocess_data
from utils.fileutils import get_path
import time
import sys

seed = 12

def train(k=5):
    # load training data
    train_X, train_y, test_X, test_id = load_and_preprocess_data()
    # Stratified CV
    k_fold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = k_fold.split(train_X, train_y)
    
    # naive bayes

    gnb_pred = np.zeros(len(test_X))
    for fold_n, (train_index, valid_index) in enumerate(k_fold.split(train_X, train_y)):
        print("fold {} starts at {}".format(fold_n, time.ctime()))
        X_train, X_valid = train_X.iloc[train_index], train_X.iloc[valid_index]
        y_train, y_valid = train_y.iloc[train_index], train_y.iloc[valid_index]

        gnb = GaussianNB()
        gnb.fit(X_train, y_train)

        gnb_pred += gnb.predict_proba(test_X)[:, 1] / 5

    gnb_submission = pd.DataFrame({'ID_code': test_id, 'target': gnb_pred})
    gnb_submission.to_csv(get_path(r'output/gnb_submission.csv'), index=False)

if __name__ == "__main__":
    k = int(sys.argv[1])
    print("Performing {} fold CV. Training starts at {}".format(k, time.ctime()))
    train(k)
    print("Training ends at {}".format(time.ctime()))
