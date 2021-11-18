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

    # blending 1: 2 LGBM + 1 NN
    blending_submission_1 = pd.DataFrame({"ID_code": nn_submission.ID_code, "target": (2 * lgbm_submission.target + nn_submission.target)/3})
    blending_submission_1.to_csv(os.path.join(path, "blending_submission_1.csv"), index=False)
    # blending 2: 3 LGBM + 2 NN + 1 Random Forest
    blending_submission_2 = pd.DataFrame({"ID_code": nn_submission.ID_code, "target": (3 * lgbm_submission.target + 2 * nn_submission.target + rf_submission.target) / 6})
    blending_submission_2.to_csv(os.path.join(path, "blending_submission_2.csv"), index=False)
    # blending 3: 3 LGBM + 2 NN + 1 Random Forest + 1 Gaussian Naive Bayes
    blending_submission_3 = pd.DataFrame({"ID_code": nn_submission.ID_code, "target": (3 * lgbm_submission.target + 2 * nn_submission.target + rf_submission.target + gnb_submission.target) / 7})
    blending_submission_3.to_csv(os.path.join(path, "blending_submission_3.csv"), index=False)
    # blending 4: 3 LGBM + 1 NN
    blending_submission_4 = pd.DataFrame({"ID_code": nn_submission.ID_code, "target": (3 * lgbm_submission.target + 1 * nn_submission.target)/4})
    blending_submission_4.to_csv(os.path.join(path, "blending_submission_4.csv"), index=False)