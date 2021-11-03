from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

# keras items
from keras import regularizers
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers import Flatten, Conv1D, MaxPooling1D, LSTM  # CNNs
from keras.activations import relu, elu, linear, softmax
from keras.callbacks import EarlyStopping, Callback
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import adam, Nadam, sgd
from keras.losses import mean_squared_error, categorical_crossentropy, logcosh
from keras.utils.np_utils import to_categorical

import talos as ta
import wrangle as wr
from talos.metrics.keras_metrics import fmeasure_acc
from talos.model.layers import hidden_layers
from talos import live
from talos.model import lr_normalizer, early_stopper, hidden_layers
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import linear_model
import DataProcess as dp
import numpy as np


# 垃圾
def linear_lasso(X_train, X_test, y_train, y_test):
    lasso = linear_model.Lasso(alpha=0.01)
    lasso.fit(X_train, y_train)
    y_hat = lasso.predict(X_test)

    # mean squared error 均方误差
    mse = mean_squared_error(y_test, y_hat)
    # print('\nMSE in prediction =', mse)

    # 比较预测值和实际值，得到预测集和实际集的相关性
    corr = np.corrcoef(y_test, y_hat)[0, 1]
    print('\nCorr obs vs pred =', corr)


def mlp(X_train, X_test, y_train, y_test):
    # no. of SNPs in data
    nSNP = X_train.shape[1]

    # 初始化对象
    model = Sequential()

    # 隐层 1
    # Dense 维度
    model.add(Dense(64, input_dim=nSNP))
    # 激活函数
    model.add(Activation('relu'))
    # Add second layer
    model.add(Dense(32))
    model.add(Activation('softplus'))
    # Last, output layer
    model.add(Dense(1))

    # 均方误差，优化器：梯度下降
    model.compile(loss='mean_squared_error', optimizer='sgd')

    # list some properties
    model.summary()

    # training
    model.fit(X_train, y_train, epochs=100)

    # cross-validation: get predicted target values
    y_hat = model.predict(X_test, batch_size=128)

    mse_prediction = model.evaluate(X_test, y_test, batch_size=128)
    print('\nMSE in prediction =', mse_prediction)

    # correlation btw predicted and observed
    corr = np.corrcoef(y_test, y_hat[:, 0])[0, 1]
    print('\nCorr obs vs pred =', corr)
    # plot observed vs. predicted targets
    plt.title('MLP: Observed vs Predicted Y')
    plt.ylabel('Predicted')
    plt.xlabel('Observed')
    plt.scatter(y_test, y_hat, marker='o')
    plt.show()
