import numpy as np

rng = np.random.default_rng(2024)
n, p = 50, 4
X = np.c_[np.ones(n), rng.standard_normal((n, p))]
y = X @ np.array([1.0, 2.0, -1.0, 0.0, 5.0]) + rng.normal(0, 1.5, n)


def fit(X, y):
    b = np.linalg.solve(X.T @ X, X.T @ y)
    return b, np.sum((y - X @ b) ** 2)


b, rss_full = fit(X, y)
df = n - X.shape[1]
s2 = rss_full / df  # random noise
se = np.sqrt(s2 * np.diag(np.linalg.inv(X.T @ X)))
t = b / se

_, rss_red = fit(np.delete(X, 3, axis=1), y)
# df_red - df_full = 1
F = (rss_red - rss_full) / s2
print(t[3] ** 2, F)
