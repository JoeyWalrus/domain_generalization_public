import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
import os
from ast import literal_eval

print("imports done")
np.set_printoptions(threshold=np.inf)


def eval_part(
    test_loader,
    test_loader_nr,
    model_names,
    path_to_models,
    split_nr,
    device="mps",
    criterion=nn.MSELoss(),
    debug=False,
):
    models = os.listdir(path_to_models)
    split_res = []
    for n, model_name in enumerate(model_names):
        if debug:
            print(n, model_name)
        model = [
            i
            for i in models
            if model_name in i and i.endswith(f"{split_nr}_{test_loader_nr}.pt")
        ][0]

        model = torch.load(f"models/{model}")
        model.to(device)
        model.eval()
        result = validate_parallel(test_loader, model, criterion, device)
        split_res.append(np.round(np.mean(result), 4))
    return split_res


def load_losses(path_to_loss_file):
    with open(path_to_loss_file) as f:
        liste = list(literal_eval(f[0]))
    print(liste)


if __name__ == "__main__":
    # load_losses(
    #     "code/real_code/losses/parkinson_all_losses_model_baseline_parkinson_model_baseline_parkinson-1-100-2_0.txt"
    # )
    print(os.listdir("code/real_code/models"))
