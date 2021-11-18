import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import time
import optuna
from sklearn.metrics import roc_auc_score
from utils.dataset import load_and_preprocess_data
from utils.fileutils import get_path
import sys
import os
from sklearn.linear_model import LogisticRegression

seed = 12
path = get_path("dump")


def create_dump_directory():
    if not os.path.exists(path):
        os.makedirs(path)


def remove_dump_directory():
    if os.path.exists(path):
        # removing the file using the os.remove() method
        os.rmdir(path)


def logit_objective(trial, X, y, test_X, test_id, k_fold):
    print("Start trial {}".format(trial.number))
    weight = trial.suggest_int("class_weight", 1, 15)

    cv_scores = np.empty(k)
    logit_pred = np.zeros(len(test_X))
    for fold_n, (train_index, valid_index) in enumerate(k_fold.split(X, y)):
        print('Fold', fold_n, 'started at', time.ctime())
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        clf = LogisticRegression(penalty='l2', random_state=seed, solver='sag', class_weight={0: 1, 1: weight})
        clf.fit(X_train, y_train)
        y_preds = clf.predict_proba(X_valid)[:, 1]

        cv_scores[fold_n] = roc_auc_score(y_valid, y_preds)
        print("ROC AUC score of fold {} is {}".format(fold_n, cv_scores[fold_n]))
        logit_pred += clf.predict_proba(test_X)[:, 1] / k

    rf_prediction = pd.DataFrame({'ID_code': test_id, 'target': logit_pred})
    rf_prediction.to_pickle(get_path("dump/logit_{}.pickle").format(trial.number))

    return np.mean(cv_scores)


def train(k=5):
    create_dump_directory()
    # load training data
    train_X, train_y, test_X, test_id = load_and_preprocess_data()
    # Stratified CV
    k_fold = StratifiedKFold(n_splits=k, shuffle=True, random_state=12)
    folds = k_fold.split(train_X, train_y)

    # random forest
    # tuning the hyperparameter using optuna
    study = optuna.create_study(direction="maximize", study_name="logistic regression")
    func = lambda trial: logit_objective(trial, train_X, train_y, test_X, test_id, k_fold)
    study.optimize(func, n_trials=10)

    logit_submission = pd.read_pickle(get_path("dump/logit_{}.pickle").format(study.best_trial.number))
    logit_submission.to_csv(get_path(r'output/logit_submission.csv'), index=False)
    remove_dump_directory()


if __name__ == "__main__":
    k = int(sys.argv[1])
    print("Performing {} fold CV. Training starts at {}".format(k, time.ctime()))
    train(k)
    print("Training ends at {}".format(time.ctime()))