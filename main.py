import mkate_mse
import mse_load
import mkate_bayesian
import bayesian_load
import torch


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

    X, Y = mkate_mse.get_data_by_depth("mkate_data.xlsx", depth, side)

    my_nn_m, optimizer_m, loss_m = mkate_mse.create(0.01, 32, epochs, X, Y)

    # Save to my_nn_m.tar
    torch.save(
        {
            "model_state_dict": my_nn_m.state_dict(),
            "optimizer_state_dict": optimizer_m.state_dict(),
            "loss": loss_m,
        },
        "my_nn_m.tar",
    )

    my_nn_b, optimizer_b, loss_b = mkate_bayesian.create(0.01, 32, epochs, X, Y)

    # Save to my_nn_b.tar
    torch.save(
        {
            "model_state_dict": my_nn_b.state_dict(),
            "optimizer_state_dict": optimizer_b.state_dict(),
            "loss": loss_b,
        },
        "my_nn_b.tar",
    )

    # Display graphs from my_nn_m.tar
    mse_load.graph_from_tar("my_nn_m.tar").show()

    # Display graphs from my_nn_b.tar
    bayesian_load.graph_from_tar("my_nn_b.tar").show()


if __name__ == "__main__":
    main()