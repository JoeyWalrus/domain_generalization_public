import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_parkinson_data_normalized_augmented(
    path="data/telemonitoring_parkinsons_updrs_augmented.csv",
    nr_outputs=2,
):
    data = pd.read_csv(path, sep=";", header=None)

    # transform data to numpy array - split into ids, X and y
    np_data = data.to_numpy()
    data_ids = np_data[:, [0, 22]]
    if nr_outputs == 2:
        data_y = np_data[:, [4, 5]]
    elif nr_outputs == 1:
        data_y = np_data[:, [5]]
    data_X = np_data[:, [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]

    # normalize data
    minimum_values = np.min(data_X, axis=0, keepdims=True)
    maximum_values = np.max(data_X, axis=0, keepdims=True)
    data_X_normalized = (data_X - minimum_values) / (maximum_values - minimum_values)

    domains_xs = [[[] for j in range(6)] for _ in range(len(data[0].unique()))]
    domains_ys = [[[] for j in range(6)] for _ in range(len(data[0].unique()))]

    for n, i in enumerate(data_ids):
        idx = int(np.round(i[0] - 1))
        vowel = int(np.round(i[1] - 1))
        domains_xs[idx][vowel].append(data_X_normalized[n])
        domains_ys[idx][vowel].append((data_y[n]))

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
    load_parkinson_data_normalized_augmented()
