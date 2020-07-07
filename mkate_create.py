import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import random


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(13, 12)
        self.fc2 = nn.Linear(12, 2)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__== "__main__":
    file_name = "mkate_data.xlsx"
    data = pd.read_excel(file_name, 0, header=[0,1])
    print(data)
    print(data.iloc[0:8192,0].values)
    X = torch.Tensor([[bool(int(x)) for x in y.replace("'","")] for y in data.iloc[0:8192,0].values])
    Y = torch.Tensor(data.iloc[0:8192,[6,7]].values)
    print(X)
    print(Y)

    my_nn = Net()
    print(my_nn)

    optimizer = optim.SGD(my_nn.parameters(), lr=0.01)

    #for epoch in range(50):
    good = False
    epoch = 0
    while not good:
        x = random.sample(list(zip(X,Y)),1)
        s, e = x[0]
        print(f"\nStart: {s}\nEnd (True): {e}\nEnd (Estimate): {my_nn(s)}")
        running_loss = 0.0
        for i, (start, end) in enumerate(random.sample(list(zip(X,Y)),2048)):
            optimizer.zero_grad()
            output = my_nn(start)
            criterion = nn.MSELoss()
            loss = criterion(output, end)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999: 
                print(f'[{epoch+1}, {i+1}] loss: {running_loss/1000}')
                if running_loss/1000 < 0.005:
                    good = True
                running_loss = 0.0
        epoch = epoch+1


    torch.save({
                'model_state_dict': my_nn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, "my_nn.tar")
