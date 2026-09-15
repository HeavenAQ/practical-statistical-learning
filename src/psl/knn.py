import numpy as np
from scipy.stats import norm


def bayes_error_example():
    # bayes error
    mu1, mu0 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    d = np.linalg.norm(mu1 - mu0)
    bayes_error = np.round(norm.cdf((-d / 2)), 5)
    print("Bayes Error:", bayes_error)

    # empirical value
    N = 200_000
    rng = np.random.default_rng(580)
    X = np.vstack(
        [
            rng.standard_normal((N, 2)) + mu1,
            rng.standard_normal((N, 2)) + mu0,
        ]
    )
