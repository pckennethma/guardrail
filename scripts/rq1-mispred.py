import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Data for the correlation matrix
"""
adult:
                  prediction_error  sanitizer_alert  noise_injection
prediction_error          1.000000         0.236935         0.186071
sanitizer_alert           0.236935         1.000000         0.622922
noise_injection           0.186071         0.622922         1.000000

lc:
                  prediction_error  sanitizer_alert  noise_injection
prediction_error          1.000000         0.127746         0.137964
sanitizer_alert           0.127746         1.000000         0.838457
noise_injection           0.137964         0.838457         1.000000

bs:
                  prediction_error  sanitizer_alert  noise_injection
prediction_error          1.000000         0.197226         0.118705
sanitizer_alert           0.197226         1.000000         0.476944
noise_injection           0.118705         0.476944         1.000000

ins:
                  prediction_error  sanitizer_alert  noise_injection
prediction_error          1.000000         0.031057         0.161334
sanitizer_alert           0.031057         1.000000         0.257222
noise_injection           0.161334         0.257222         1.000000
"""
data = {
    "INS": np.array(
        [
            [0.0, 0.03, 0.16],
            [0.03, 0.0, 0.26],
            [0.16, 0.26, 0.0],
        ]
    ),
    "ADULT": np.array(
        [
            [0.0, 0.24, 0.19],
            [0.24, 0.0, 0.62],
            [0.19, 0.62, 0.0],
        ]
    ),
    "LC": np.array(
        [
            [0.0, 0.13, 0.14],
            [0.13, 0.0, 0.84],
            [0.14, 0.84, 0.0],
        ]
    ),
    "BS": np.array(
        [
            [0.0, 0.20, 0.12],
            [0.20, 0.0, 0.48],
            [0.12, 0.48, 0.0],
        ]
    ),
}

dataset_order = ["ADULT", "LC", "INS", "BS"]

fig, stacked_axs = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(7.5, 7.5),
    gridspec_kw={"hspace": 0.3, "wspace": 0.3},
    sharex="col",
    sharey="row",
)
sns.set(font_scale=1.2)
axs = [
    stacked_axs[0, 0],
    stacked_axs[0, 1],
    stacked_axs[1, 0],
    stacked_axs[1, 1],
]
diag_mask = np.zeros_like(data["INS"], dtype=bool)
np.fill_diagonal(diag_mask, True)

for i, dataset in enumerate(dataset_order):
    ax = axs[i]
    sns.heatmap(
        data[dataset],
        annot=True,
        cmap="Blues",
        fmt=".2f",
        ax=ax,
        square=True,
        cbar=False,
        mask=diag_mask,
        vmin=0,
        vmax=1,
    )
    ax.set_title(dataset)
    ax.set_xlabel("")
    ax.set_xticklabels(["Mis-pred.", "Detected", "Injected"])
    ax.set_yticklabels(["Mis-pred.", "Detected", "Injected"])
    ax.set_ylabel("")
fig.tight_layout()
fig.savefig("statistics/rq1-mispred.pdf", bbox_inches="tight")
