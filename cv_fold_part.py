import numpy as np
import itertools


def cv_fold_part(nr_domains, nr_folds):
    domain_indices = np.arange(nr_domains)
    np.random.shuffle(domain_indices)
    fold_indices = np.array_split(domain_indices, nr_folds)
    combinations = list(itertools.combinations(np.arange(nr_folds), 2))

    splits = []
    for c in combinations:
        train_idx = np.concatenate(
            [fold_indices[i] for i in np.setdiff1d(np.arange(nr_folds), np.array(c))]
        )
        test_idx_1 = fold_indices[c[0]]
        test_idx_2 = fold_indices[c[1]]
        splits.append((train_idx, test_idx_1, test_idx_2))

    for s in splits:
        assert np.concatenate(s).shape[0] == nr_domains
        assert np.intersect1d(s[0], s[1]).shape[0] == 0
        assert np.intersect1d(s[0], s[2]).shape[0] == 0
        assert np.intersect1d(s[1], s[2]).shape[0] == 0

    return splits


if __name__ == "__main__":
    res = cv_fold_part(10, 4)
    print(res)
