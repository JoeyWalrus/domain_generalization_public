import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# version without different vowels
def load_parkinson_data(path="data/telemonitoring_parkinsons_updrs.data.csv"):
    data = pd.read_csv(path)
    # transform data to numpy array - split into ids, X and y
    np_data = data.to_numpy()
    data_ids = np_data[:, 0]
    data_y = np_data[:, [5]]
    data_X = np_data[:, [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]
    # normalize data
    minimum_values = np.min(data_X, axis=0, keepdims=True)
    maximum_values = np.max(data_X, axis=0, keepdims=True)
    data_X_normalized = (data_X - minimum_values) / (maximum_values - minimum_values)

    # split into domains and subdomains
    domains_xs = [[[] for j in range(1)] for _ in range(len(data["subject#"].unique()))]
    domains_ys = [[[] for j in range(1)] for _ in range(len(data["subject#"].unique()))]
    vowel = 0
    old_x = 0
    old_id = 0
    for n, i in enumerate(data_ids):
        idx = int(np.round(i - 1))
        if data_X[n][2] < old_x:
            vowel += 1
        if idx != old_id:
            vowel = 0
            old_id = idx
        old_x = data_X[n][2]
        domains_xs[idx][0].append(data_X_normalized[n])
        domains_ys[idx][0].append((data_y[n]))

    for i in range(len(domains_xs)):
        for j in range(len(domains_xs[i])):
            domains_xs[i][j] = np.array(domains_xs[i][j])
            domains_ys[i][j] = np.array(domains_ys[i][j])

    for i in range(len(domains_xs)):
        for j in range(len(domains_xs[i])):
            assert len(domains_xs[i][j]) == len(domains_ys[i][j])
            assert np.min(domains_xs[i][j]) >= 0
            assert np.max(domains_xs[i][j]) <= 1
    return domains_xs, domains_ys


def load_parkinson_data_normalized_data(
    path="data/telemonitoring_parkinsons_updrs.data.csv",
    nr_outputs=2,
):
    data = pd.read_csv(path)

    # transform data to numpy array - split into ids, X and y
    np_data = data.to_numpy()
    data_ids = np_data[:, 0]
    if nr_outputs == 2:
        data_y = np_data[:, [4, 5]]
    elif nr_outputs == 1:
        data_y = np_data[:, [5]]
    data_X = np_data[:, [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]

    # try to make space more eventyl distributed
    # calc mean per feature
    mean_per_feature = np.mean(data_X, axis=0)
    changed_entries = 0
    for i in range(len(data_X[0])):
        changed_entries += len(data_X[:, i][data_X[:, i] > mean_per_feature[i] * 2])
        data_X[:, i][data_X[:, i] > mean_per_feature[i] * 2] = mean_per_feature[i] * 2
    print("changed entries: ", changed_entries)

    # normalize data
    minimum_values = np.min(data_X, axis=0, keepdims=True)
    maximum_values = np.max(data_X, axis=0, keepdims=True)
    data_X_normalized = (data_X - minimum_values) / (maximum_values - minimum_values)

    domains_xs = [[[] for j in range(1)] for _ in range(len(data["subject#"].unique()))]
    domains_ys = [[[] for j in range(1)] for _ in range(len(data["subject#"].unique()))]

    vowel = 0
    old_x = 0
    old_id = 0
    for n, i in enumerate(data_ids):
        idx = int(np.round(i - 1))
        if data_X[n][2] < old_x:
            vowel += 1
        if idx != old_id:
            vowel = 0
            old_id = idx
        old_x = data_X[n][2]
        domains_xs[idx][0].append(data_X_normalized[n])
        domains_ys[idx][0].append((data_y[n]))

    for i in range(len(domains_xs)):
        for j in range(len(domains_xs[i])):
            domains_xs[i][j] = np.array(domains_xs[i][j])
            domains_ys[i][j] = np.array(domains_ys[i][j])

    for i in range(len(domains_xs)):
        for j in range(len(domains_xs[i])):
            assert len(domains_xs[i][j]) == len(domains_ys[i][j])
            assert np.min(domains_xs[i][j]) >= 0
            assert np.max(domains_xs[i][j]) <= 1

    return domains_xs, domains_ys


if __name__ == "__main__":
    load_parkinson_data()
    load_parkinson_data_normalized_data()
