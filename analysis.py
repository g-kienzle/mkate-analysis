import mkate_mse
import mse_load
import mkate_bayesian
import bayesian_load
import torch
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np

# Gets graphs for all depth/side combinations
def bplots_from_tar(tar, type):
    plt.close()
    plt.rcParams.update({"figure.max_open_warning": 0})
    # Initialize NN and optimizer
    my_nn = mkate_bayesian.Net() if type == "b" else mkate_mse.Net()
    optimizer = optim.SGD(my_nn.parameters(), lr=0.01)

    # Import my_nn.tar
    checkpoint = torch.load(tar)
    my_nn.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    loss = checkpoint["loss"]

    my_nn.eval()

    data = pd.read_excel("mkate_data.xlsx", 0, header=[0, 1])

    X = torch.Tensor(
        [
            [bool(int(x)) for x in y.replace("'", "")]
            for y in data.iloc[0:8192, 0].values
        ]
    )
    Y = torch.Tensor(data.iloc[0:8192, [6, 7]].values)

    # Parse graphed data
    dist, real_r, real_b = (
        [torch.sum(l).item() for l in X],
        [l.tolist()[0] for l in Y],
        [l.tolist()[1] for l in Y],
    )

    estim_data = my_nn(X)

    estim_r, estim_b = [l.tolist()[0] for l in estim_data], [
        l.tolist()[1] for l in estim_data
    ]

    real_t, estim_t = [r + b for r, b in zip(real_r, real_b)], [
        r + b for r, b in zip(estim_r, estim_b)
    ]

    diff_r, diff_b, diff_t = (
        [r - e for r, e in zip(real_r, estim_r)],
        [r - e for r, e in zip(real_b, estim_b)],
        [r - e for r, e in zip(real_t, estim_t)],
    )
    a1 = []
    a2 = []
    a3 = []
    a4 = []
    a5 = []
    a6 = []
    a7 = []
    a8 = []
    a9 = []
    a10 = []
    a11 = []
    a12 = []
    a13 = []

    vals = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13]
    for n, v in zip(dist, diff_t):
        vals[int(n - 1)].append(v)

    fig, plots = plt.subplots()
    plt.boxplot(vals)
    # Size = fig.get_size_inches()
    # fig.set_size_inches(Size[0] * 2, Size[1] * 2, forward=True)

    return plt


def error_plot(depth, side, type):
    vals = [[], []]
    plt.close()

    for d in range(1, 14):
        # Save graphs from tars

        my_nn = mkate_bayesian.Net() if type == "b" else mkate_mse.Net()
        optimizer = optim.SGD(my_nn.parameters(), lr=0.01)

        # Import my_nn.tar
        checkpoint = (
            torch.load("Tars/Bayesian_" + str(d) + side + ".tar")
            if type == "b"
            else torch.load("Tars/MSE_" + str(d) + side + ".tar")
        )
        my_nn.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        loss = checkpoint["loss"]

        my_nn.eval()

        data = pd.read_excel("mkate_data.xlsx", 0, header=[0, 1])
        X = torch.Tensor(
            [
                [bool(int(x)) for x in y.replace("'", "")]
                for y in data.iloc[0:8192, 0].values
            ]
        )
        Y = torch.Tensor(data.iloc[0:8192, [6, 7]].values)
        dist, real_r, real_b = (
            [torch.sum(l).item() for l in X],
            [l.tolist()[0] for l in Y],
            [l.tolist()[1] for l in Y],
        )

        estim_data = my_nn(X)

        estim_r, estim_b = [l.tolist()[0] for l in estim_data], [
            l.tolist()[1] for l in estim_data
        ]

        trimmed_real = []
        trimmed_estim = []

        arrayzip = (
            zip(dist, real_r, estim_r) if side == "r" else zip(dist, real_b, estim_b)
        )

        for dis, r, e in arrayzip:
            if dis == depth:
                trimmed_real.append(r)
                trimmed_estim.append(e)

        vals[0].append(d)
        vals[1].append(
            F.mse_loss(torch.Tensor(trimmed_estim), torch.Tensor(trimmed_real)).item()
        )

    plt.scatter(vals[0],vals[1])
    plt.table(cellText=[[round(x,3) for x in vals[1]]], loc = 'top')
    return plt


def main():
    for depth in range(1, 14):
        for side in ["b", "r"]:
            # Save graphs from tars
            bplots_from_tar("Tars/MSE_" + str(depth) + side + ".tar", "m").savefig(
                "Analysis/MSE_box_" + str(depth) + side
            )

            bplots_from_tar("Tars/Bayesian_" + str(depth) + side + ".tar", "b").savefig(
                "Analysis/Bayesian_box_" + str(depth) + side
            )
            
            error_plot(depth, side, "m").savefig(
                "Analysis/MSE_MSE_" + str(depth) + side
            )
            error_plot(depth, side, "b").savefig(
                "Analysis/Bayesian_MSE_" + str(depth) + side
            )


if __name__ == "__main__":
    main()