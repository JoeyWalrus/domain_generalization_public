import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class ParallelDomainGenDataset(Dataset):
    def __init__(self, X1, X2, Y):
        self.X1 = X1
        self.X2 = X2
        self.Y = Y

    def __len__(self):
        return len(self.X1)

    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.Y[idx]


def flatten_list(l):
    l = [item for sublist in l for item in sublist]
    l = [item for sublist in l for item in sublist]
    return l


def add_data_to_array(emb_array, x_array, y_array, indices):
    tmp_emb_array = []
    tmp_x_array = []
    tmp_y_array = []
    for i in indices:
        tmp_emb_array.append(emb_array[i])
        tmp_x_array.append(x_array[i])
        tmp_y_array.append(y_array[i])

    tmp_emb_array = torch.from_numpy(np.array(flatten_list(tmp_emb_array))).float()
    tmp_x_array = torch.from_numpy(np.array(flatten_list(tmp_x_array))).float()
    tmp_y_array = torch.from_numpy(np.array(flatten_list(tmp_y_array))).float()

    return tmp_emb_array, tmp_x_array, tmp_y_array


def training_preparation_part(
    emb_array, x_array, y_array, split_indices, batch_size=128
):
    train_emb_tensor, train_x_tensor, train_y_tensor = add_data_to_array(
        emb_array, x_array, y_array, split_indices[0]
    )
    test1_emb_tensor, test1_x_tensor, test1_y_tensor = add_data_to_array(
        emb_array, x_array, y_array, split_indices[1]
    )
    test2_emb_tensor, test2_x_tensor, test2_y_tensor = add_data_to_array(
        emb_array, x_array, y_array, split_indices[2]
    )

    train_dataset = ParallelDomainGenDataset(
        train_emb_tensor, train_x_tensor, train_y_tensor
    )
    test1_dataset = ParallelDomainGenDataset(
        test1_emb_tensor, test1_x_tensor, test1_y_tensor
    )
    test2_dataset = ParallelDomainGenDataset(
        test2_emb_tensor, test2_x_tensor, test2_y_tensor
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test1_loader = DataLoader(test1_dataset, batch_size=batch_size, shuffle=True)
    test2_loader = DataLoader(test2_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test1_loader, test2_loader


if __name__ == "__main__":
    np.random.seed(0)
    emb_data = [
        [
            np.random.rand(np.random.randint(50, 150), 10, 100)
            for _ in range(np.random.randint(5, 10))
        ]
        for _ in range(5)
    ]
    value_data = []
    y_data = []

    for e in emb_data:
        value_data.append([])
        y_data.append([])
        for v in e:
            y_data[-1].append(np.ones((len(v))))
            value_data[-1].append(np.random.rand(len(v), 10))

    training_preparation_part(
        emb_data, value_data, y_data, ([0, 1, 2], [3], [4]), batch_size=128
    )
