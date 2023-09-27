import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_sepsis_data(
    path="data/physionet_training_setA_interpolated_merged.csv",
):
    data = pd.read_csv(path)
    # transform data to numpy array - split into ids, X and y
    np_data = data.to_numpy()
    data_ids = np_data[:, 32]
    data_y = np_data[:, 31]
    data_y = np.eye(2)[data_y.astype(int)]
    data_X = np_data[:, :31]

    # normalize data
    minimum_values = np.min(data_X, axis=0, keepdims=True)
    maximum_values = np.max(data_X, axis=0, keepdims=True)
    data_X_normalized = (data_X - minimum_values) / (maximum_values - minimum_values)

    domains_xs = [
        [[] for j in range(1)] for _ in range(len(data["patient_nr"].unique()))
    ]
    domains_ys = [
        [[] for j in range(1)] for _ in range(len(data["patient_nr"].unique()))
    ]

    for n, i in enumerate(data_ids):
        idx = int(np.round(i))
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
    load_sepsis_data()
