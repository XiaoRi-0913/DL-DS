import torch
from torch import nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.decomposition import PCA
import warnings
from day2.DS import utils
warnings.filterwarnings("ignore")

def load_data():
    X = pd.read_csv('../day2/DS/data/wheat.X', header=None, sep='\s+')
    Y = pd.read_csv('../day2/DS/data/wheat.Y', header=None, sep='\s+')
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


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(386, 1)

    def forward(self, x):
        return self.linear(x)

X_train, X_test, y_train, y_test = load_data()
X_train, X_test = snp_preselection(X_train, X_test, y_train)
X_train = utils.convertToTensor(X_train)
y_train = utils.convertToTensor(y_train)
X_test = utils.convertToTensor(X_test)
y_test = utils.convertToTensor(y_test)

model = LinearModel()
# size_average 损失是否求均值 reduce是否降维求和？
criterion = nn.MSELoss(size_average=False)
optimizer = torch.optim.ASGD(model.parameters(), lr=0.01)

for epoch in range(100):
    y_predict = model(X_train)
    loss = criterion(y_predict, y_train)
    if epoch % 10 == 0:
        print(epoch, loss)

    # 每一次训练 梯度都需要清零
    optimizer.zero_grad()
    loss.backward()
    # 更新
    optimizer.step()


y_test = model(X_test)
print("y_predict = ", y_test)
