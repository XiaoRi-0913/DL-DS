import torch
import numpy as np

import DataProcess as dp
import torchLearning as tl
import utils

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = dp.load_data(3)
    X_train, X_test = dp.snp_preselection(X_train, X_test, y_train)
    print("X_train shape is {}, Y_train shape is {}, X_test shape is {}, y_test shape is {}"
          .format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
    X_train = utils.convertToTensor(X_train)
    y_train = utils.convertToTensor(y_train)
    X_test = utils.convertToTensor(X_test)
    tl.torch_linear(X_train, X_test, y_train, y_test)
    # dl.linear_lasso(X_train, X_test, y_train, y_test)

    # X = np.array([[1], [2] ,[3]])
    # Y = np.array([[1], [2], [3]])
    # X = X.flatten()
    # Y = Y.flatten()
    # cor = np.corrcoef(X,Y)[0,1]
    # print("==========y_test type is {}, y_hat type is {}==========".format(X.shape, Y.shape))
    # print("cor is {}".format(cor))
