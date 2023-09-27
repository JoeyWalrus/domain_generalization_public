import torch
import torch.nn as nn
import os
import numpy as np


def epoch_parallel(dataloader, model, criterion, optimizer, device):
    model.train()

    train_loss = []
    count = 1
    for t in dataloader:
        X1 = t[0].to(device)
        X2 = t[1].to(device)
        y = t[2].to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X1, X2)

        # Compute the loss
        loss = criterion(outputs, y)

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

        train_loss.append(loss.item())
        count += 1

    return train_loss


def validate_parallel(dataloader, model, criterion, device):
    model.eval()

    test_loss = []
    for t in dataloader:
        X1 = t[0].to(device)
        X2 = t[1].to(device)
        y = t[2].to(device)

        # Forward pass
        outputs = model(X1, X2)

        # Compute the loss
        loss = criterion(outputs, y)

        test_loss.append(loss.item())
    return test_loss


def calc_f1_scores(res, target):
    f1_scores = []
    for i in np.arange(0, 1, 0.01):
        pred = res > i

        tp = torch.sum((pred == 1) & (target == 1))
        tn = torch.sum((pred == 0) & (target == 0))
        fp = torch.sum((pred == 1) & (target == 0))
        fn = torch.sum((pred == 0) & (target == 1))

        tp = tp.item()
        tn = tn.item()
        fp = fp.item()
        fn = fn.item()

        # calculate f1 score
        if tp + fn == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)
        if tp + fp == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)
        if recall == 0 or precision == 0:
            f1 = 0
        else:
            f1 = 2 * recall * precision / (recall + precision)
        f1_scores.append(f1)
    return f1_scores


def eval_f1_score(
    test_loader,
    test_loader_nr,
    model_names,
    path_to_models,
    split_nr,
    device="mps",
):
    models = os.listdir(path_to_models)
    split_res_f1 = []
    s = nn.Softmax(dim=1)
    for model_name in model_names:
        print(model_name)
        model = [
            i
            for i in models
            if model_name in i and i.endswith(f"{split_nr}_{test_loader_nr}.pt")
        ][0]

        model = torch.load(f"models/{model}")
        model.to(device)
        model.eval()

        res = []
        true = []
        for x in test_loader:
            x = [i.to(device) for i in x]
            ouput = model(x[0], x[1])
            ouput = s(ouput)[:, 1]
            target = torch.max(x[2], dim=1).indices
            res.append(ouput.detach().cpu())
            true.append(target.detach().cpu())
        res = torch.cat((res))
        true = torch.cat((true))
        f1_scores = calc_f1_scores(res, true)
        split_res_f1.append(f1_scores)
        del model
    return split_res_f1
