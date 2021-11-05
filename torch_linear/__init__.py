import torch
from torch import nn

x_data = torch.Tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])
x_test = torch.Tensor([[4.0, 4.0]])


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


model = LinearModel()
criterion = nn.MSELoss(size_average=False)
optimizer = torch.optim.ASGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    # 前向传播
    outputs = model(x_data)
    # 计算损失
    loss = criterion(outputs, y_data)
    # 反向传播
    loss.backward()
    # 更新权重参数
    optimizer.step()
    if epoch % 100 == 0:
        print("epoch{}, loss{}".format(epoch, loss.item()))

print("w = ", model.linear.weight.item())
print("b = ", model.linear.bias.item())
predicted = model(x_test).data.numpy()
print("y_predict = ", predicted)
