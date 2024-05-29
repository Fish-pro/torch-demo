import torch
import math

# 设置数据类型和设备
dtype = torch.float
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

# 生成数据
x = torch.linspace(-math.pi, math.pi, 2000, dtype=dtype, device=device)
y = torch.sin(x)

# 初始化多项式系数，并设置 requires_grad=True
a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

# 设置学习率和训练
learning_rate = 1e-6
for t in range(2000):
    # 前向传播计算预测值
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # 计算损失
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # 反向传播计算梯度
    loss.backward()

    # 使用梯度更新参数
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        # 清零梯度
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

# 输出结果
print(f"Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3")
