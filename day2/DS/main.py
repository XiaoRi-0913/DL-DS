import torch

import DataProcess as dp
import DeepLearning as dl
import torchLearning as tl
import numpy as np
import utils


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = dp.load_data()
    X_train, X_test = dp.snp_preselection(X_train, X_test, y_train)
    # X_train = utils.convertToTensor(X_train)
    # y_train = utils.convertToTensor(y_train)
    # X_test = utils.convertToTensor(X_test)
    # y_test = utils.convertToTensor(y_test)
    # print(X_train)
    # tl.torch_linear(X_train, X_test, y_train, y_test)
    dl.linear_lasso(X_train, X_test, y_train, y_test)
