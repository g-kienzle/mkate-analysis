import mkate_mse
import mse_load
import mkate_bayesian
import bayesian_load
import torch

#Gets graphs for all depth/side combinations
def main():
    for depth in range(1,14):
        for side in ['b','r']:
            X, Y = mkate_mse.get_data_by_depth("mkate_data.xlsx", depth, side)

            my_nn_m, optimizer_m, loss_m = mkate_mse.create(0.01, 32, 50, X, Y)

            #Save to my_nn_m.tar
            torch.save({
                'model_state_dict': my_nn_m.state_dict(),
                'optimizer_state_dict': optimizer_m.state_dict(),
                'loss': loss_m,
                }, "Tars/MSE_"+str(depth)+side+".tar")

            my_nn_b, optimizer_b, loss_b = mkate_bayesian.create(0.01, 32, 50, X, Y)

            #Save to my_nn_b.tar
            torch.save({
                'model_state_dict': my_nn_b.state_dict(),
                'optimizer_state_dict': optimizer_b.state_dict(),
                'loss': loss_b,
                }, "Tars/Bayesian_"+str(depth)+side+".tar")

            #Save graphs from tars
            mse_load.graph_from_tar("Tars/MSE_"+str(depth)+side+".tar").savefig("Data/MSE_"+str(depth)+side)

            bayesian_load.graph_from_tar("Tars/Bayesian_"+str(depth)+side+".tar").savefig("Data/Bayesian_"+str(depth)+side)

if __name__ == "__main__":
    main()