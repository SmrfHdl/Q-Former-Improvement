import torch

# a = torch.arange(1, 4, 1, torch.float32)
# print(a)
# out = torch.empty(5)
# torch.arange(0, 10, 2, out)
# print(out)
x = torch.arange(6)
print(x)
y = x.view(2, 3)
print(y)
y[0, 0] = 99
print(y)
print(x)
