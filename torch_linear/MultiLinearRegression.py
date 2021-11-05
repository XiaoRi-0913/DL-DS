# 拟合y=0.9+0.5*x+3*x^2
#  y = b + w1 *x + w2 *x**2

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# f(x)=5x1+4x2+3x3+3
x_train = np.array(
    [[1, 3, 4], [2, 4, 2], [7, 5, 9], [2, 5, 6], [6, 4, 2], [8, 2, 7], [9, 3, 6], [1, 6, 8], [5, 3, 6], [3, 7, 3]],
    dtype=np.float32)
y_train = x_train[:, 0] * 5 + x_train[:, 1] * 4 + 3 * x_train[:, 2] + 3
y_train = y_train.reshape((10, 1))

print(x_train.shape)
print(y_train.shape)

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)


class MultiLinearRegression(nn.Module):
    def __init__(self):
        super(MultiLinearRegression, self).__init__()
        self.linear = nn.Linear(3, 1)  # 因为3个变量映射1个输出

    def forward(self, x):
        out = self.linear(x)
        return out


model = MultiLinearRegression()

if torch.cuda.is_available():
    model = model.cuda()
    x_train = x_train.cuda()
    y_train = y_train.cuda()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

for epoch in range(1000):
    output = model(x_train)  # 前向传播
    loss = criterion(output, y_train)  # 损失计算
    loss_value = loss.data.cpu().numpy()  # 获取损失值
    optimizer.zero_grad()  # 梯度置零
    loss.backward()  # 反向传播
    optimizer.step()  # 更新梯度

    epoch += 1
    if epoch % 100 == 0:  # 每100步打印一次损失
        print('Epoch:{}, loss:{:.6f}'.format(epoch, loss_value))
    if loss_value <= 1e-3:
        break

w = model.linear.weight.data.cpu().numpy()
b = model.linear.bias.data.cpu().numpy()
print('w:{},b:{}'.format(w, b))

# 结果为
