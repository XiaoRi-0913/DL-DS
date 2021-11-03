# main modules needed
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")


def load_data():
    X = pd.read_csv('DATA/wheat.X', header=None, sep='\s+')
    Y = pd.read_csv('DATA/wheat.Y', header=None, sep='\s+')
    itrait = 0
    X_train, X_test, y_train, y_test = train_test_split(X, Y[itrait], test_size=0.2, random_state=22)
    X = np.concatenate((X_train, X_test))  # 把训练集和测试集拼接
    pca = PCA(n_components=0.6)  # PCA 降维
    p = pca.fit(X).fit_transform(X)
    return X_train, X_test, y_train, y_test


def snp_preselection(X_train, X_test, y_train):
    pvals = []
    for i in range(X_train.shape[1]):
        b, intercept, r_value, p_value, std_err = stats.linregress(X_train[i], y_train)
        pvals.append(-np.log10(p_value))
    pvals = np.array(pvals)
    N_best = 100
    snp_list = pvals.argsort()[-N_best:]
    min_P_value = 1
    snp_list = np.nonzero(pvals > min_P_value)
    X_train = X_train[X_train.columns[snp_list]]
    X_test = X_test[X_test.columns[snp_list]]
    return X_train, X_test
