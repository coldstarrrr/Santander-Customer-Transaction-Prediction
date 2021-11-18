import pandas as pd
from utils.fileutils import get_path
import os

path = get_path("output")

if __name__ == "__main__":
    # neural network
    nn_submission = pd.read_csv(os.path.join(path, "nn_submission.csv"))
    # gaussian naive bayes
    gnb_submission = pd.read_csv(os.path.join(path, "gnb_submission.csv"))
    # light gbm
    lgbm_submission = pd.read_csv(os.path.join(path, "lgbm_submission.csv"))
    # logistic regression
    logit_submission = pd.read_csv(os.path.join(path, "logit_submission.csv"))
    # random forest
    rf_submission = pd.read_csv(os.path.join(path, "rf_submission.csv"))