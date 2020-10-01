import mkate_bayesian
import pandas as pd
import torch
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt


def graph_from_tar(tar):
    plt.rcParams.update({"figure.max_open_warning": 0})
    # Initialize NN and optimizer
    my_nn = mkate_bayesian.Net()
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
    # Initialize and display plots
    fig, plots = plt.subplots(3, 3)

    real_red = plots[0, 0]
    real_blue = plots[1, 0]
    real_tot = plots[2, 0]
    estim_red = plots[0, 1]
    estim_blue = plots[1, 1]
    estim_tot = plots[2, 1]
    diff_red = plots[0, 2]
    diff_blue = plots[1, 2]
    diff_tot = plots[2, 2]

    images = []

    h = real_red.hist2d(
        dist, real_r, bins=[13, 32], cmap="pink"
    )  # , norm=matplotlib.colors.LogNorm())
    images.append(h)
    fig.colorbar(h[3], ax=real_red)
    real_red.set_title("Measured Red Brightness")
    real_red.set_xlabel("Depth from mKate2")
    real_red.set_ylabel("Red brightness")

    h = real_blue.hist2d(
        dist, real_b, bins=[13, 32], cmap="pink"
    )  # , norm=matplotlib.colors.LogNorm())
    images.append(h)
    fig.colorbar(h[3], ax=real_blue)
    real_blue.set_title("Measured Blue Brightness")
    real_blue.set_xlabel("Depth from mKate2")
    real_blue.set_ylabel("Blue brightness")

    h = real_tot.hist2d(
        dist, real_t, bins=[13, 32], cmap="pink"
    )  # , norm=matplotlib.colors.LogNorm())
    images.append(h)
    fig.colorbar(h[3], ax=real_tot)
    real_tot.set_title("Measured Total Brightness")
    real_tot.set_xlabel("Depth from mKate2")
    real_tot.set_ylabel("Total brightness")

    h = estim_red.hist2d(
        dist, estim_r, bins=[13, 32], cmap="pink"
    )  # , norm=matplotlib.colors.LogNorm())
    images.append(h)
    fig.colorbar(h[3], ax=estim_red)
    estim_red.set_title("Estimated Red Brightness")
    estim_red.set_xlabel("Depth from mKate2")
    estim_red.set_ylabel("Red brightness")

    h = estim_blue.hist2d(
        dist, estim_b, bins=[13, 32], cmap="pink"
    )  # , norm=matplotlib.colors.LogNorm())
    images.append(h)
    fig.colorbar(h[3], ax=estim_blue)
    estim_blue.set_title("Estimated Blue Brightness")
    estim_blue.set_xlabel("Depth from mKate2")
    estim_blue.set_ylabel("Blue brightness")

    h = estim_tot.hist2d(
        dist, estim_t, bins=[13, 32], cmap="pink"
    )  # , norm=matplotlib.colors.LogNorm())
    images.append(h)
    fig.colorbar(h[3], ax=estim_tot)
    estim_tot.set_title("Estimated Total Brightness")
    estim_tot.set_xlabel("Depth from mKate2")
    estim_tot.set_ylabel("Total brightness")

    h = diff_red.hist2d(dist, diff_r, bins=[13, 32], cmap="pink")
    images.append(h)
    diff_red.set_title("Difference in Red")
    diff_red.set_xlabel("Depth from mKate2")
    diff_red.set_ylabel("Red brightness")

    h = diff_blue.hist2d(dist, diff_b, bins=[13, 32], cmap="pink")
    images.append(h)
    diff_blue.set_title("Difference in Blue")
    diff_blue.set_xlabel("Depth from mKate2")
    diff_blue.set_ylabel("Blue brightness")

    h = diff_tot.hist2d(dist, diff_t, bins=[13, 32], cmap="pink")
    images.append(h)
    diff_tot.set_title("Difference in Total")
    diff_tot.set_xlabel("Depth from mKate2")
    diff_tot.set_ylabel("Total brightness")

    vmin = min(image[3].get_array().min() for image in images)
    vmax = max(image[3].get_array().max() for image in images)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im[3].set_norm(norm)

    plt.tight_layout()
    Size = fig.get_size_inches()
    fig.set_size_inches(Size[0] * 2, Size[1] * 2, forward=True)
    # plt.show()
    return plt


def main():
    graph_from_tar("my_nn_b.tar").show()


if __name__ == "__main__":
    main()