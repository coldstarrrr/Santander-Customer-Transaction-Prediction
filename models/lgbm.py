import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import time
import lightgbm as lgb
import sys
from utils.dataset import load_and_preprocess_data
from utils.fileutils import get_path

seed = 12

def train(k=5):
    # load training data
    train_X, train_y, test_X, test_id = load_and_preprocess_data()
    # Stratified CV
    k_fold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = k_fold.split(train_X, train_y)

    ## light gbm

    params = {'objective': "binary",
              'boost': "gbdt",
              'metric': "auc",
              'boost_from_average': "false",
              'num_threads': -1,
              'learning_rate': 0.01,
              'num_leaves': 13,
              'max_depth': -1,
              'tree_learner': "serial",
              'feature_fraction': 0.05,
              'bagging_freq': 5,
              'bagging_fraction': 0.4,
              'min_data_in_leaf': 80,
              'min_sum_hessian_in_leaf': 10.0,
              'verbosity': -100}

    lgb_pred = np.zeros(len(test_X))
    num_round = 1000000
    for fold_n, (train_index, valid_index) in enumerate(folds):
        print('Fold', fold_n, 'started at', time.ctime())
        X_train, X_valid = train_X.iloc[train_index], train_X.iloc[valid_index]
        y_train, y_valid = train_y.iloc[train_index], train_y.iloc[valid_index]

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid)

        lgb_model = lgb.train(params, train_data, num_round,
                              valid_sets=[train_data, valid_data], verbose_eval=1, early_stopping_rounds=3500)

        lgb_pred += lgb_model.predict(test_X, num_iteration=lgb_model.best_iteration) / k

    lgb_submission = pd.DataFrame({'ID_code': test_id, 'target': lgb_pred})
    lgb_submission.to_csv(get_path(r'output/lgbm_submission.csv'), index=False)


if __name__ == "__main__":
    print("Training starts at {}".format(time.cnow()))
    train(int(sys.argv[1]))
    print("Training ends at {}".format(time.cnow()))
