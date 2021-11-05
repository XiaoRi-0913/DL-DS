import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        # self.linear = nn.Linear(141, 1)
        # self.linear = nn.Linear(220, 1)
        # self.linear = nn.Linear(217, 1)
        self.linear = nn.Linear(296, 1)

    def forward(self, x):
        return self.linear(x)


def torch_linear(X_train, x_test, Y_train, y_test):
    model = LinearModel()
    # size_average 损失是否求均值 reduce是否降维求和？
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    for epoch in range(1001):
        # 梯度清零
        optimizer.zero_grad()
        # 前向传播
        outputs = model(X_train)
        # 计算损失
        loss = criterion(outputs, Y_train)
        # 反向传播
        loss.backward()
        # 更新权重参数
        optimizer.step()
        if epoch % 50 == 0:
            print("epoch{}, loss{}".format(epoch, loss.item()))
    y_valid = model(x_test).data.numpy()
    y_valid = y_valid.flatten()
    # y_valid = torch.from_numpy(y_valid)
    # plot observed vs. predicted targets
    corr = np.corrcoef(y_test, y_valid)[0, 1]
    print("corr is {}".format(corr))
    plt.title('torch : Observed vs Predicted Y')
    plt.ylabel('Predicted')
    plt.xlabel('Observed')
    plt.scatter(y_test, y_valid, marker='o')
    plt.show()


