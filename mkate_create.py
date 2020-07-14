import mkate_load
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import random

#Neural network structure: 13 input bits, 12-node hidden layer, 2 output floats
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(13, 12)
        self.fc2 = nn.Linear(12, 2)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_data_by_depth(file_name, depth, side):
    #Data import
    data = pd.read_excel(file_name, 0, header=[0,1])
    
    #Data parsing
    inputs = list([[bool(int(x)) for x in y.replace("'","")] for y in data.iloc[0:8192,0].values])
    outputs = list(data.iloc[0:8192,[6,7]].values)

    count = 0
    if side == "b":
        for i in range(8192):
            if sum(inputs[i-count]) > depth:
                del inputs[i-count]
                del outputs[i-count]
                count = count+1
    if side == "r":
        for i in range(8192):
            if sum(inputs[i-count]) < depth:
                del inputs[i-count]
                del outputs[i-count]
                count = count+1

    return torch.Tensor(inputs), torch.Tensor(outputs)

def create(lrate, batch_size, good_loss, X, Y):
    #Initialize NN, define batch size and optimizer
    my_nn = Net()
    optimizer = optim.SGD(my_nn.parameters(), lr=lrate)

    #Initialize iterator values
    good = False
    epoch = 0

    #Train until MSE less than good_loss
    while not good:
        running_loss = 0.0
        l = list(zip(X,Y))
        random.shuffle(l)
        for j in range(len(l)//batch_size):
            for i, (start, end) in enumerate(l[batch_size*j:batch_size*(j+1)]):
                optimizer.zero_grad()
                output = my_nn(start)
                criterion = nn.MSELoss()
                loss = criterion(output, end)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

        epoch = epoch+1

        print(f'[Epoch: {epoch}] loss: {running_loss/len(l)}')
        if running_loss/len(l) < good_loss:
            good = True
    
    return my_nn, optimizer, loss

def main():
    side = ""
    while side != "b" and side != "r":
        side = input("Choose to enter depth from protein 0000000000000 (b) or protein 1111111111111 (r): ")
    depth = 0
    while depth < 1 or depth > 13:
        try:
            depth = int(input("Enter depth: "))
        except:
            pass
    good_loss = 0
    while good_loss <= 0 or good_loss >= 1:
        try:
            good_loss = int(input("Enter MSE loss to train until (press 'Enter' for default): "))
        except:
            good_loss = 0.0055

    X, Y = get_data_by_depth("mkate_data.xlsx", depth, side)

    my_nn, optimizer, loss = create(0.01, 32, good_loss, X, Y)

    #Save to my_nn.tar
    torch.save({
                'model_state_dict': my_nn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, "my_nn.tar")

    #Display graphs from my_nn.tar
    mkate_load.graph_from_tar("my_nn.tar")

if __name__== "__main__":
    main()