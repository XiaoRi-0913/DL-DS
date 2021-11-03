import numpy as np
import torch
from torch import nn


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(386, 1)

    def forward(self, x):
        return self.linear(x)


def torch_linear(X_train, x_test, Y_train, y_test):
    model = LinearModel()
    # size_average 损失是否求均值 reduce是否降维求和？
    criterion = nn.MSELoss(size_average=False)
    optimizer = torch.optim.ASGD(model.parameters(), lr=0.1)
    print("X_train is =====",X_train[0, 100])
    print("y_train is =====", Y_train)
    for epoch in range(100):
        y_predict = model(X_train)
        loss = criterion(y_predict, Y_train)
        # 每一次训练 梯度都需要清零
        optimizer.zero_grad()
        loss.backward()
        # 更新
        optimizer.step()
        if epoch % 10 == 0:
            print(epoch, loss)
    y_valid = model(torch.from_numpy(np.array(x_test)).requires_grad_()).data.numpy()
    print(y_valid[10])
