from pathlib import Path

import matplotlib.pyplot as plt
import torch

OUTPUT = Path("/Users/chenyimin/Desktop/zenn-docs/images")

# Assume that
# - our formula is y = 3 + 0.8 * x1 +  0.5 * x2 + noise
# - we have 30 data points

# Create random data
torch.manual_seed(42)
n = 30

x1 = torch.rand(n) * 10
x2 = torch.randn(n) * 10
noise = torch.randn(n) * 3
y = 3 + 0.8 * x1 + 0.5 * x2 + noise

# Create design matrix
X = torch.stack(
    [
        torch.ones(n),
        x1,
        x2,
    ],
    dim=1,
)

# estimation
beta = torch.linalg.lstsq(X, y).solution
beta0, beta1, beta2 = beta
y_hat = X @ beta

# calculate RSS
residuals = y - y_hat

# visualization
x1_grid = torch.linspace(
    x1.min(),
    x1.max(),
    25,
)

x2_grid = torch.linspace(
    x2.min(),
    x2.max(),
    25,
)
X1_grid, X2_grid = torch.meshgrid(
    x1_grid,
    x2_grid,
    indexing="ij",
)
Y_grid = beta0 + beta1 * X1_grid + beta2 * X2_grid

# plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(
    111,
    projection="3d",
)
ax.plot_surface(
    X1_grid,
    X2_grid,
    Y_grid,
    alpha=0.25,
    color="slateblue",
    edgecolor="darkslateblue",
    linewidth=0.4,
    antialiased=True,
)
ax.scatter(
    x1,
    x2,
    y,
    s=55,
    color="orange",
)

for i in range(n):
    ax.plot(
        [x1[i], x1[i]],
        [x2[i], x2[i]],
        [y_hat[i], y[i]],
        linestyle=":",
        color="gray",
    )

ax.view_init(elev=25, azim=-60)


plt.savefig(OUTPUT / "multiple-linear-regression.png")
