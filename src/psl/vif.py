import numpy as np


# variance inflation factor
# Regress predictor j on all the others and call the result R^2.
def vif(Z):
    out = []
    for j in range(Z.shape[1]):
        W = np.c_[np.ones(len(Z)), np.delete(Z, j, axis=1)]
        zj = Z[:, j]
        yh = W @ np.linalg.lstsq(W, zj, rcond=None)[0]
        r2 = 1 - ((zj - yh) ** 2).sum() / ((zj - zj.mean()) ** 2).sum()
        out.append(1 / (1 - r2))
    return np.array(out)
