import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import time
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from utils.dataset import load_and_preprocess_data
from utils.fileutils import get_path
import sys
import os

seed = 12
path = get_path("dump")

def create_dump_directory():
    if not os.path.exists(path):
        os.makedirs(path)

def rf_objective(trial, X, y, test_X, test_id, k_fold):
    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 15, step=1),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["auto", "sqrt"])
        }

    cv_scores = np.empty(k)
    rf_pred = np.zeros(len(test_X))
    for fold_n, (train_index, valid_index) in enumerate(k_fold.split(X, y)):
        print('Fold', fold_n, 'started at', time.ctime())
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        rf = RandomForestClassifier(n_jobs=-1, random_state=seed, class_weight={0: 1, 1: 3}, **param_grid)
        rf.fit(X_train, y_train)
        y_preds = rf.predict_proba(X_valid)[:, 1]

        cv_scores[fold_n] = roc_auc_score(y_valid, y_preds)
        print("ROC AUC score of fold {} is {}".format(fold_n, cv_scores[fold_n]))
        rf_pred += rf.predict_proba(test_X)[:, 1] / k

    rf_prediction = pd.DataFrame({'ID_code': test_id, 'target': rf_pred})
    rf_prediction.to_pickle(get_path("dump/rf_{}.pickle").format(trial.number))

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
    study = optuna.create_study(direction="maximize", study_name="Random Forest")
    func = lambda trial: rf_objective(trial, train_X, train_y, test_X, test_id, k_fold)
    study.optimize(func, n_trials=30)

    rf_submission = pd.read_pickle(get_path("dump/rf_{}.pickle").format(study.best_trial.number))
    print("best trial is trial", study.best_trial.number)
    rf_submission.to_csv(get_path(r'output/rf_submission.csv'), index=False)


if __name__ == "__main__":
    k = int(sys.argv[1])
    print("Performing {} fold CV. Training starts at {}".format(k, time.ctime()))
    train(k)
    print("Training ends at {}".format(time.ctime()))