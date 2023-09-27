import torch
import torch.nn as nn


class head_empty(nn.Module):
    def __init__(
        self,
        feature_size,
        embedding_size,
        nr_layers=1,
        nr_neurons=100,
        output_size=5,
        dropout=0.3,
    ):
        super(head_empty, self).__init__()
        self.name = f"head_empty-{feature_size}-{embedding_size}-{nr_layers}-{nr_neurons}-{output_size}-{dropout}"
        self.head = nn.Sequential()

    def forward(self, x):
        device = x.device
        return torch.empty(x.shape[0], 0).to(device)


class head_linear(nn.Module):
    """Simple linear head"""

    def __init__(
        self,
        feature_size,
        embedding_size,
        nr_layers=1,
        nr_neurons=100,
        output_size=5,
        dropout=0.3,
    ):
        super(head_linear, self).__init__()
        self.name = f"head_linear-{feature_size}-{embedding_size}-{nr_layers}-{nr_neurons}-{output_size}-{dropout}"
        self.head = nn.Sequential(
            nn.Linear(feature_size, nr_neurons),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Sequential(
                *[
                    nn.Sequential(
                        nn.Linear(nr_neurons, nr_neurons),
                        nn.Dropout(dropout),
                        nn.ReLU(),
                    )
                    for _ in range(nr_layers)
                ]
            ),
            nn.Flatten(),
            nn.Linear(nr_neurons * embedding_size, output_size),
        )

    def forward(self, x):
        return self.head(x)


class head_cnn_weighted_sum(nn.Module):
    """Head that computes weighted sum"""

    def __init__(
        self,
        feature_size,
        embedding_size,
        nr_layers=1,
        nr_neurons=100,
        output_size=5,
        dropout=0.3,
    ):
        super(head_cnn_weighted_sum, self).__init__()
        self.name = f"head_cnn_weighted_sum-{feature_size}-{embedding_size}-{nr_layers}-{nr_neurons}-{output_size}-{dropout}"
        self.head = nn.Sequential(
            nn.Conv2d(1, output_size, (feature_size, 1), padding=0),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Flatten(),
            head_linear(
                feature_size=embedding_size * output_size,
                embedding_size=1,
                nr_layers=nr_layers,
                nr_neurons=nr_neurons,
                output_size=output_size,
                dropout=dropout,
            ),
        )

    def forward(self, x):
        return self.head(x.unsqueeze(1))


class head_cnn_pure(nn.Module):
    def __init__(
        self,
        feature_size,
        embedding_size,
        nr_layers=1,
        nr_neurons=100,
        output_size=5,
        dropout=0.3,
    ):
        super(head_cnn_pure, self).__init__()
        self.name = f"head_cnn_pure-{feature_size}-{embedding_size}-{nr_layers}-{nr_neurons}-{output_size}-{dropout}"
        self.head = nn.Sequential(
            nn.Conv2d(1, nr_neurons, 3, padding=0),
            nn.Dropout(dropout),
            nn.ReLU(),
            *[
                nn.Sequential(
                    nn.Conv2d(nr_neurons, nr_neurons, 3, padding=0),
                    nn.Dropout(dropout),
                    nn.ReLU(),
                )
                for _ in range(nr_layers)
            ],
            nn.Flatten(),
            nn.Linear(
                nr_neurons
                * (feature_size - 2 - 2 * nr_layers)
                * (embedding_size - 2 - 2 * nr_layers),
                output_size,
            ),
        )

    def forward(self, x):
        return self.head(x.unsqueeze(1))


################################################################################################


class model_embedding_and_input(nn.Module):
    def __init__(
        self,
        experiment_name,
        model_name,
        feature_size,
        embedding_size,
        embedding_nr_layers=1,
        embedding_neurons=100,
        embedding_head_class=head_linear,
        hidden_size=5,
        evaluation_input_size=10,
        evaluation_nr_layers=1,
        evaluation_neurons=100,
        evaluation_head_class=head_linear,
        output_size=1,
    ):
        super(model_embedding_and_input, self).__init__()
        self.embedding_head = embedding_head_class(
            feature_size=feature_size,
            embedding_size=embedding_size,
            nr_layers=embedding_nr_layers,
            nr_neurons=embedding_neurons,
            output_size=hidden_size,
        )
        self.evaluation_head = evaluation_head_class(
            feature_size=evaluation_input_size,
            embedding_size=1,
            nr_layers=evaluation_nr_layers,
            nr_neurons=evaluation_neurons,
            output_size=output_size,
        )
        self.name = f"{experiment_name}-{model_name}|{self.embedding_head.name}|{self.evaluation_head.name}"

    def forward(self, x1, x2):
        x1 = self.embedding_head(x1)
        x3 = torch.cat((x1, x2), dim=1)
        y = self.evaluation_head(x3)
        y = y.squeeze()
        if len(y.shape) == 1:
            y = y.unsqueeze(1)
        return y


if __name__ == "__main__":
    device = "mps"
    # test inputs
    t1 = torch.rand(20, 10, 30).to(device)
    t2 = torch.rand(20, 10).to(device)
    # test the heads
    m = head_empty(
        feature_size=10,
        embedding_size=30,
        nr_layers=1,
        nr_neurons=100,
        output_size=5,
        dropout=0.3,
    ).to(device)
    assert m(t1).shape == torch.Size([20, 0])
    m = head_linear(
        feature_size=30,
        embedding_size=10,
        nr_layers=1,
        nr_neurons=100,
        output_size=5,
        dropout=0.3,
    ).to(device)
    assert m(t1).shape == torch.Size([20, 5])
    m = head_linear(
        feature_size=10,
        embedding_size=1,
        nr_layers=1,
        nr_neurons=100,
        output_size=5,
        dropout=0.3,
    ).to(device)
    assert m(t2).shape == torch.Size([20, 5])

    m = head_cnn_weighted_sum(
        feature_size=10,
        embedding_size=30,
        nr_layers=1,
        nr_neurons=100,
        output_size=5,
        dropout=0.3,
    ).to(device)
    assert m(t1).shape == torch.Size([20, 5])
    m = head_cnn_pure(
        feature_size=10,
        embedding_size=30,
        nr_layers=1,
        nr_neurons=100,
        output_size=5,
        dropout=0.3,
    ).to(device)
    assert m(t1).shape == torch.Size([20, 5])

    # this would be the baseline model
    m = model_embedding_and_input(
        experiment_name="test",
        model_name="baseline",
        feature_size=0,
        embedding_size=0,
        embedding_nr_layers=0,
        embedding_neurons=1,
        embedding_head_class=head_empty,
        hidden_size=10,
        evaluation_input_size=10,
        evaluation_nr_layers=1,
        evaluation_neurons=100,
        evaluation_head_class=head_linear,
        output_size=5,
    ).to(device)
    assert m(t1, t2).shape == torch.Size([20, 5])

    # this would be a model with a linear head
    m = model_embedding_and_input(
        experiment_name="test",
        model_name="linear",
        feature_size=30,
        embedding_size=10,
        embedding_nr_layers=1,
        embedding_neurons=100,
        embedding_head_class=head_linear,
        hidden_size=10,
        evaluation_input_size=20,
        evaluation_nr_layers=1,
        evaluation_neurons=100,
        evaluation_head_class=head_linear,
        output_size=5,
    ).to(device)
    assert m(t1, t2).shape == torch.Size([20, 5])

    # this would be a model with a cnn summarization head
    m = model_embedding_and_input(
        experiment_name="test",
        model_name="cnn_sum",
        feature_size=10,
        embedding_size=30,
        embedding_nr_layers=1,
        embedding_neurons=100,
        embedding_head_class=head_cnn_weighted_sum,
        hidden_size=10,
        evaluation_input_size=20,
        evaluation_nr_layers=1,
        evaluation_neurons=100,
        evaluation_head_class=head_linear,
        output_size=5,
    ).to(device)
    assert m(t1, t2).shape == torch.Size([20, 5])

    # this would be a model with a cnn pure head
    m = model_embedding_and_input(
        experiment_name="test",
        model_name="cnn_pure",
        feature_size=10,
        embedding_size=30,
        embedding_nr_layers=1,
        embedding_neurons=100,
        embedding_head_class=head_cnn_pure,
        hidden_size=10,
        evaluation_input_size=20,
        evaluation_nr_layers=1,
        evaluation_neurons=100,
        evaluation_head_class=head_linear,
        output_size=5,
    ).to(device)
    assert m(t1, t2).shape == torch.Size([20, 5])
    print("done")
