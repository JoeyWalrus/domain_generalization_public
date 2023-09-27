import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from utils import *

print("imports done")
np.set_printoptions(threshold=np.inf)


def training_part(
    train_loader,
    test1_loader,
    test2_loader,
    model_list,
    split_nr,
    device="mps",
    lr=0.0001,
    max_epochs=1000,
    early_stopping=50,
    criterion=nn.MSELoss(),
    debug=False,
):
    device = device
    lr = lr

    all_losses = [[] for _ in range(3)]

    for n, m in enumerate(model_list):
        if debug:
            print(f"Model nr {n}: {m.name}")
        model = m
        model.to(device)
        criterion = criterion
        optimizer = optim.Adam(model.parameters(), lr=lr)

        nr_epochs = max_epochs
        best_loss_1 = 9999999999999999
        best_model_1_best_epoch = -1
        best_loss_2 = 9999999999999999
        best_model_2_best_epoch = -1

        for e in range(nr_epochs):
            t_l = epoch_parallel(
                train_loader,
                model,
                criterion,
                optimizer,
                device=device,
            )
            r1 = validate_parallel(test1_loader, model, criterion, device)
            r2 = validate_parallel(test2_loader, model, criterion, device)
            print(
                f"Split nr {split_nr}: {model.name}: Epoch {e}: Mean Train loss: {np.round(np.mean(t_l),4)}"
            )

            all_losses[0].append(t_l)
            all_losses[1].append(r1)
            all_losses[2].append(r2)

            # the naming of the saved models is the following: experiment name, model name, split number, test number
            # test number is important since this indicates whether the model was validated against test set 1 or 2
            # and should therefore be evaluated agains the other test set respectively in the evaluation part
            if np.mean(r1) < best_loss_1:
                best_loss_1 = np.mean(r1)
                best_model_1_best_epoch = e
                torch.save(
                    model,
                    f"models/{model.name}_{split_nr}_0.pt",
                )
                print(
                    f"Split nr {split_nr}: {model.name}: Epoch {e}: Mean Test 1 loss: {np.round(np.mean(r1),4)} - best so far"
                )
            else:
                print(
                    f"Split nr {split_nr}: {model.name}: Epoch {e}: Mean Test 1 loss: {np.round(np.mean(r1),4)}"
                )

            if np.mean(r2) < best_loss_2:
                best_loss_2 = np.mean(r2)
                best_model_2_best_epoch = e
                torch.save(
                    model,
                    f"models/{model.name}_{split_nr}_1.pt",
                )
                print(
                    f"Split nr {split_nr}: {model.name}: Epoch {e}: Mean Test 2 loss: {np.round(np.mean(r2),4)} - best so far"
                )
            else:
                print(
                    f"Split nr {split_nr}: {model.name}: Epoch {e}: Mean Test 2 loss: {np.round(np.mean(r2),4)}"
                )

            if (
                e - best_model_1_best_epoch > early_stopping
                and e - best_model_2_best_epoch > early_stopping
            ):
                print(f"Early stopping for model {model.__class__.__name__ }")
                break

        # save the nested list all_losses to a file as a string
        # this is done to avoid problems with the numpy save function
        # since the nested list is not a numpy array
        with open(
            f"losses/{model.name}_{split_nr}.txt",
            "w+",
        ) as f:
            f.write(str(all_losses))
