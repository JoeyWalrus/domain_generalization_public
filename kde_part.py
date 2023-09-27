from sklearn.neighbors import KernelDensity
import numpy as np


def calc_kde_uniform_sampling(
    data_to_be_fitted,
    points_to_be_evalualted,
    kernel=KernelDensity(kernel="gaussian", bandwidth=0.03),
):
    kde = kernel.fit(data_to_be_fitted)
    res = np.exp(kde.score_samples(points_to_be_evalualted))
    return res


def normalize_embedding(embedding):
    e_area = np.sum((embedding[:, :-1] + embedding[:, 1:]) / 2) / len(embedding)
    embedding /= e_area
    return embedding


def kde_part(data, kernel="gaussian", bandwidth=0.03, points=100):
    # calculates kde per dimension of subdomain, returns array of
    # nr_subdomain samples x nr_dimensions x points

    # note: no normalization of embeddings right now!
    mesh = np.linspace(0, 1, points).reshape(-1, 1)
    emb_data = []
    # now add the embedding to the data
    for n, domain in enumerate(data):
        emb_data.append([])
        for m, subdomain in enumerate(domain):
            emb_data[n].append([])
            for dimension in range(len(subdomain[0])):
                embeddings = calc_kde_uniform_sampling(
                    subdomain[:, dimension].reshape(-1, 1),
                    mesh,
                    kernel=KernelDensity(kernel=kernel, bandwidth=bandwidth),
                ).reshape(1, -1)

                embeddings = normalize_embedding(embeddings)

                emb_data[n][m].append(embeddings)
            emb_data[n][m] = np.concatenate(emb_data[n][m], axis=0)
            emb_data[n][m] = np.repeat(
                np.expand_dims(emb_data[n][m], 0), len(subdomain), axis=0
            )
        if n % 100 == 0:
            print(n)

    assert len(emb_data) == len(data)
    for i in range(len(data)):
        assert len(emb_data[i]) == len(data[i])
        for j in range(len(data[i])):
            assert len(emb_data[i][j]) == len(data[i][j])
    return emb_data


if __name__ == "__main__":
    np.random.seed(0)
    data = [
        [
            np.random.rand(np.random.randint(50, 150), 10)
            for _ in range(np.random.randint(5, 10))
        ]
        for _ in range(5)
    ]
    for d in data:
        print(len(d))

    res = kde_part(data)
    print(res[0][0].shape)
