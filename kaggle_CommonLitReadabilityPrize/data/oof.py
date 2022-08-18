import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


def create_folds(data, n_splits, random_state=None):
    data["fold"] = -1
    data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    num_bins = int(np.floor(1 + np.log2(len(data))))  # bins数量(Sturges规则)
    data.loc[:, "bins"] = pd.cut(data["target"], bins=num_bins, labels=False)  # 根据'target'列进行分箱
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)  # 分层k折
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, "fold"] = f

    data = data.drop("bins", axis=1)
    return data
