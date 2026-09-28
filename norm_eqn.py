import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# weight toggle
WEIGHTED = "YES"

csv_path = sys.argv[1] if len(sys.argv) > 1 else "Astrostatistics_hubble.csv"
df = pd.read_csv(csv_path)
dist = df.iloc[:, 0].to_numpy()
dist_err = df.iloc[:, 1].to_numpy()
vel = df.iloc[:, 2].to_numpy()
n = len(dist)

# making design matrix
X = np.column_stack([np.ones(n), vel])
y = dist

if WEIGHTED == "YES":
    w = 1 / dist_err**2
else:
    w = np.ones(n)
W = np.diag(w)  # weight matrix

XtX = X.T @ W @ X
Xty = X.T @ W @ y
beta_hat = np.linalg.solve(XtX, Xty)
c, beta = beta_hat  # parameters beta =! beta_hat

y_hat = X @ beta_hat
r = y - y_hat  # residuals

rss = np.sum(w * r**2)  # residual sum of squares
y_bar = np.sum(w * y) / np.sum(w)

dof = n - X.shape[1]
sigma2 = rss / dof
cov_beta = sigma2 * np.linalg.inv(XtX)  # covariance matrix
se_c, se_beta = np.sqrt(np.diag(cov_beta))

fit_label = "weighted" if WEIGHTED == "YES" else "unweighted"
print(f"({fit_label} fit)")
print(rf"Distance = ({beta:.10f} +/- {se_beta:.10f}) x Velocity + ({c:.10f} +/- {se_c:.3f})")
print(cov_beta)