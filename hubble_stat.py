import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

# making it nice
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times'
plt.rcParams['font.size'] = 15

# weight toggle
WEIGHTED = "YES"

# perturbation toggle
PERTURB = "NO"
N_TRIALS = 1000

# removal togle (max 35)
REMOVE = "NO"
N_REMOVE = 35

# sequence removal togle, dont pair with removal (max 35)
SEQ_REMOVE = "YES"
N_SEQ_TRIALS = 35  

csv_path = "Astrostatistics_hubble.csv"
df = pd.read_csv(csv_path)

if REMOVE == "YES":
    keep = np.sort(np.random.choice(len(df), len(df) - N_REMOVE, replace=False))
    df = df.iloc[keep]
    print(f"Removed {N_REMOVE} random points, {len(df)} remaining")

dist = df.iloc[:, 0]
dist_err = df.iloc[:, 1]
vel = df.iloc[:, 2]

if WEIGHTED == "YES":
    weights = 1 / dist_err
    (beta, c), cov = np.polyfit(vel, dist, 1, w=weights, cov=True)
else:
    (beta, c), cov = np.polyfit(vel, dist, 1, cov=True)

print(cov)

p = beta * vel + c  # fit value
r = dist - p  # residuals

for v, d, pi, ri in zip(vel, dist, p, r):
    print(f"velocity={v:.3f}, distance={d:.3f}, predicted={pi:.3f}, residual={ri:.3f}")

fit_label = "weighted fit" if WEIGHTED == "YES" else "unweighted fit"
xfit = np.linspace(vel.min(), vel.max(), 50)
yfit = beta * xfit + c

fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(
    vel,
    dist,
    yerr=dist_err,
    fmt="o",
    ecolor="lightgray",
    elinewidth=1,
    capsize=3,
    markersize=5,
    markerfacecolor="cornflowerblue",
    markeredgecolor="dodgerblue",
    zorder=3,
)

beta_trials = None

if PERTURB == "YES":
    beta_trials = np.zeros(N_TRIALS)

    for i in range(N_TRIALS):
        dist_perturbed = dist + np.random.normal(0, dist_err)

        if WEIGHTED == "YES":
            (beta_i, c_i), cov_i = np.polyfit(vel, dist_perturbed, 1, w=weights, cov=True)
        else:
            (beta_i, c_i), cov_i = np.polyfit(vel, dist_perturbed, 1, cov=True)

        beta_trials[i] = beta_i

        yfit_i = beta_i * xfit + c_i
        ax.plot(
            xfit,
            yfit_i,
            color="cornflowerblue",
            linewidth=0.5,
            alpha=0.05,
            zorder=1,
        )

    print(f"Perturbed gradient: mean={beta_trials.mean():.6f}, std={beta_trials.std():.6f}")

ax.plot(
    xfit,
    yfit,
    color='darkorange',
    linewidth=1,
    linestyle='--',
    label=rf"Distance = {beta:.6f} $\times$ Velocity + {c:.6f} ({fit_label})",
    zorder=4,
)
ax.set_ylabel("Distance (Mpc)")
ax.set_xlabel("Velocity (km/s)")
ax.set_title("Velocity vs. Distance (1a SN)", fontsize=18)
ax.legend()

if PERTURB == "YES":
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.hist(
        beta_trials,
        bins=30,
        color="cornflowerblue",
        edgecolor="dodgerblue",
    )
    ax2.xaxis.set_major_locator(MultipleLocator(0.0005))
    ax2.axvline(
        beta,
        color="darkorange",
        linewidth=2,
        linestyle="--",
        label=f"original gradient = {beta:.6f}",
    )
    ax2.set_xlabel("Best-fit gradient (beta)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Distribution of Perturbed Gradients ({N_TRIALS} trials)", fontsize=18)
    ax2.legend()

if SEQ_REMOVE == "YES":
    vel_arr = vel.to_numpy()
    dist_arr = dist.to_numpy()
    dist_err_arr = dist_err.to_numpy()

    n_points = len(vel_arr)
    max_removed = n_points - 4  # np.polyfit with cov=True needs more than 3 points
    n_removed = np.arange(max_removed + 1)

    seq_sigma = np.zeros((N_SEQ_TRIALS, max_removed + 1))  # std error on the gradient
    seq_dbeta = np.zeros((N_SEQ_TRIALS, max_removed + 1))  # |gradient - gradient with no removal|

    for t in range(N_SEQ_TRIALS):
        order = np.random.permutation(n_points)  # random order of removal

        for k in n_removed:
            idx = order[k:]  # drop the first k points in the random order

            if WEIGHTED == "YES":
                (beta_k, c_k), cov_k = np.polyfit(vel_arr[idx], dist_arr[idx], 1, w=1 / dist_err_arr[idx], cov=True)
            else:
                (beta_k, c_k), cov_k = np.polyfit(vel_arr[idx], dist_arr[idx], 1, cov=True)

            seq_sigma[t, k] = np.sqrt(cov_k[0, 0])
            seq_dbeta[t, k] = abs(beta_k - beta)

    fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))

    for ax_i, data, ylabel, title in zip(
        (ax3, ax4),
        (seq_sigma, seq_dbeta),
        ("Gradient standard error", r"|$\beta_{removed} - \beta_{original}$|"),
        ("Uncertainty vs. Points Removed", "Gradient Shift vs. Points Removed"),
    ):
        ax_i.plot(
            n_removed,
            data.T,
            color="cornflowerblue",
            linewidth=0.5,
            alpha=0.3,
            zorder=1,
        )
        ax_i.plot(
            n_removed,
            data.mean(axis=0),
            color="darkorange",
            linewidth=2,
            linestyle="--",
            label=f"mean of {N_SEQ_TRIALS} random orders",
            zorder=4,
        )
        ax_i.set_xlabel("Number of points removed")
        ax_i.set_ylabel(ylabel)
        ax_i.set_title(title, fontsize=18)
        ax_i.legend()

    fig3.tight_layout()

plt.show()