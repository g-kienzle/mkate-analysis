from mkate_create import Net
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import random


my_nn = Net()
optimizer = optim.SGD(my_nn.parameters(), lr=0.01)

checkpoint = torch.load("my_nn.tar")
my_nn.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
loss = checkpoint['loss']

my_nn.eval()

file_name = "mkate_data.xlsx"
data = pd.read_excel(file_name, 0, header=[0,1])

X = torch.Tensor([[bool(int(x)) for x in y.replace("'","")] for y in data.iloc[0:8192,0].values])
Y = torch.Tensor(data.iloc[0:8192,[6,7]].values)

print(my_nn(torch.zeros(13)))
print(Y)

print(my_nn)

dist, real_r, real_b = [torch.sum(l).item() for l in X], [l.tolist()[0] for l in Y], [l.tolist()[1] for l in Y]

estim_data = my_nn(X)

estim_r, estim_b = [l.tolist()[0] for l in estim_data], [l.tolist()[1] for l in estim_data]

fig, plots = plt.subplots(2,2)

real_red = plots[0,0]
real_blue = plots[1,0]
estim_red = plots[0,1]
estim_blue = plots[1,1]

real_red.scatter(dist, real_r, c="red")
real_red.set_title('Distance/Real Red')
real_red.set_xlabel('Distance from gene 1')
real_red.set_ylabel('Red appearance')

real_blue.scatter(dist, real_b, c="blue")
real_blue.set_title('Distance/Real Blue')
real_blue.set_xlabel('Distance from gene 1')
real_blue.set_ylabel('Blue appearance')

estim_red.scatter(dist, estim_r, c="red")
estim_red.set_title('Distance/Estimated Red')
estim_red.set_xlabel('Distance from gene 1')
estim_red.set_ylabel('Red appearance')

estim_blue.scatter(dist, estim_b, c="blue")
estim_blue.set_title('Distance/Estimated Blue')
estim_blue.set_xlabel('Distance from gene 1')
estim_blue.set_ylabel('Blue appearance')

plt.show()