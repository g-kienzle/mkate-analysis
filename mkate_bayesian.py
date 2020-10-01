import bayesian_load
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math


class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0, 1)

    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))

    def sample(self):
        epsilon = self.normal.sample(self.rho.size())
        return self.mu + self.sigma * epsilon

    def log_prob(self, input):
        return (
            -math.log(math.sqrt(2 * math.pi))
            - torch.log(self.sigma)
            - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)
        ).sum()


class ScaleMixtureGaussian(object):
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0, sigma1)
        self.gaussian2 = torch.distributions.Normal(0, sigma2)

    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1 - self.pi) * prob2)).sum()


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Weight parameters
        self.weight_mu = nn.Parameter(
            torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2)
        )
        self.weight_rho = nn.Parameter(
            torch.Tensor(out_features, in_features).uniform_(-5, -4)
        )
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
        # Prior distributions
        PI = 0.5
        SIGMA_1 = torch.FloatTensor([math.exp(-0)])
        SIGMA_2 = torch.FloatTensor([math.exp(-6)])
        self.weight_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.bias_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, input, sample=False, calculate_log_probs=False):
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(
                weight
            ) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(
                weight
            ) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        return F.linear(input, weight, bias)


# Neural network structure: 13 input bits, 8-node hidden layer, 2 output floats
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = BayesianLinear(13, 8)
        self.fc2 = BayesianLinear(8, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def log_prior(self):
        return self.fc1.log_prior + self.fc2.log_prior

    def log_variational_posterior(self):
        return self.fc1.log_variational_posterior + self.fc2.log_variational_posterior

    def loss(self, inputs, target):
        outputs = self(inputs)
        log_prior = self.log_prior()
        variance_loss = F.mse_loss(outputs, target)
        loss = variance_loss - log_prior * 1.0 / NUM_BATCHES
        return loss


def get_data_by_depth(file_name, depth, side):
    # Data import
    data = pd.read_excel(file_name, 0, header=[0, 1])

    # Data parsing
    inputs = list(
        [
            [bool(int(x)) for x in y.replace("'", "")]
            for y in data.iloc[0:8192, 0].values
        ]
    )
    outputs = list(data.iloc[0:8192, [6, 7]].values)

    count = 0
    if side == "b":
        for i in range(8192):
            if sum(inputs[i - count]) > depth:
                del inputs[i - count]
                del outputs[i - count]
                count = count + 1
    if side == "r":
        for i in range(8192):
            if sum(inputs[i - count]) < depth:
                del inputs[i - count]
                del outputs[i - count]
                count = count + 1

    return torch.Tensor(inputs), torch.Tensor(outputs)


NUM_BATCHES = 0


def create(lrate, batch_size, epochs, X, Y):
    # Initialize NN, define batch size and optimizer
    global NUM_BATCHES
    NUM_BATCHES = len(X) // batch_size
    my_nn = Net()
    optimizer = optim.SGD(my_nn.parameters(), lr=lrate)
    my_nn.train()
    # Train for "epochs"
    loss = 0
    for epoch in range(epochs):
        running_loss = 0.0
        l = list(zip(X, Y))
        random.shuffle(l)
        for j in range(len(l) // batch_size):
            for i, (start, end) in enumerate(l[batch_size * j : batch_size * (j + 1)]):
                my_nn.zero_grad()
                loss = my_nn.loss(start, end)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

        print(f"[Epoch: {epoch+1}] loss: {running_loss/len(l)}")

    my_nn.train(False)
    return my_nn, optimizer, loss


def main():
    side = ""
    while side != "b" and side != "r":
        side = input(
            "Choose to enter depth from protein 0000000000000 (b) or protein 1111111111111 (r): "
        )
    depth = 0
    while depth < 1 or depth > 13:
        try:
            depth = int(input("Enter depth: "))
        except:
            pass
    epochs = 0
    while epochs <= 0 or epochs >= 10000:
        try:
            epochs = int(input("Enter epoch count (press 'Enter' for default): "))
        except:
            epochs = 50

    X, Y = get_data_by_depth("mkate_data.xlsx", depth, side)

    my_nn, optimizer, loss = create(0.01, 32, epochs, X, Y)

    # Save to my_nn_b.tar
    torch.save(
        {
            "model_state_dict": my_nn.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        "my_nn_b.tar",
    )

    # Display graphs from my_nn_b.tar
    bayesian_load.graph_from_tar("my_nn_b.tar").show()


if __name__ == "__main__":
    main()