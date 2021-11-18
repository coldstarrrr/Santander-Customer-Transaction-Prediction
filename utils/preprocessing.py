import pandas as pd
from tqdm import tqdm
from scipy.special import erfinv
from sklearn.preprocessing import StandardScaler




def rank_gauss(df, cols=None):
    """
    performs gauss rank transformation: rank the data,
    then transform the rank to normal distribution using the inverse error function
    """
    ## rank -> (0,1)
    df_ = (df.rank() - 1) / len(df)
    ## (-1,1)
    df_ = df_ * 2 - 1
    df_ = df_.clip(-0.999, 0.999)
    return erfinv(df_)


def count_encoding(df):
    """
    count encoder
    """
    df_ = pd.DataFrame()
    feats = list(df.columns)
    for feat in tqdm(feats):
        val_cts = df[feat].value_counts().to_dict()
        # there's no NA data - don't do anything
        df_['{}_ct'.format(feat)] = df[feat].map(val_cts)
    return df_


def transform(df):
    """
    transform input data in the following way
    1. count encode the data, and gauss rank transform the count
    2. standard scale the original data
    3. horizontally concat the results from step 1 and 2
    """
    scaler = StandardScaler()
    df_count = rank_gauss(count_encoding(df))
    df_encoded = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
    return pd.concat([df_encoded, df_count], axis=1)
