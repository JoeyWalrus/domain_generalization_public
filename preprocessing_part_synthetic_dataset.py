import numpy as np
from scipy.stats import truncnorm


print("imports done")


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def create_synthetic_dataset(nr_domains=100, sample_size=200):
    # creation of synthetic dataset
    N = nr_domains
    # calculates number of training samples
    # draws N means from "meta" normal distribution
    domain_means = np.repeat(
        np.random.uniform(0.3, 0.7, N).reshape(-1, 1), sample_size, axis=1
    )
    # draws N noise vectors from normal distribution
    y_noise = np.random.normal(0, 0.02, (N, sample_size))

    # draws N domain latents from uniform distribution - domain latents are the variances for the data
    # data is not directly drawn from a normal distribution with a concrete standard deviation
    # but with zero mean and fixed standard deviation and then scaled (or added with means)
    # the reason: the same generator can be used for all distributions
    domain_latents = np.repeat(
        np.random.uniform(2, 10, N).reshape(-1, 1), sample_size, axis=1
    )
    # here the generator is created for all distributions
    X = get_truncated_normal(mean=0, sd=1 / 8, low=-0.3, upp=0.3)
    x = X.rvs((N, sample_size))
    # now the domain latents are used to scale the data (and the domain means to shift it) to the desired distribution
    for i in range(N):
        x[i, :] *= domain_latents[i] / 10
        x[i, :] += domain_means[i]
    y = (
        np.sin(x * 300 / (domain_means * 10) ** 2) / 10
        + 0.9
        - ((x - 0.5) * 1.7) ** 2
        + y_noise
    )

    x = np.expand_dims(x, 1)
    x = np.expand_dims(x, 3)
    y = np.expand_dims(y, 1)
    y = np.expand_dims(y, 3)

    return x, y


if __name__ == "__main__":
    create_synthetic_dataset(100, 200)
