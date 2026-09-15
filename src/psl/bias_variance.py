import numpy as np


def bias_var(deg, reps=6000, n=30, sig=0.3, x0=0.2, seed=0):
    """
    Given a f = sin(2x), refit it 6000 times with 30 samples and x0 = 0.2, and samples in the [-1, 3] interval
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(-1, 3, n)
    fx0 = np.sin(2 * x0)
    preds = np.array(
        [
            np.polyval(
                np.polyfit(
                    x,
                    # add some noises to the original function
                    np.sin(2 * x) + rng.normal(0, sig, n),
                    deg,
                ),
                x0,
            )
            for _ in range(reps)
        ]
    )

    mean = preds.mean()
    bias_2 = (mean - fx0) ** 2
    var = preds.var()
    return bias_2, var, sig**2, bias_2 + var + sig**2


for d in range(10):
    print(d, [round(v, 5) for v in bias_var(d)])
