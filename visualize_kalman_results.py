import pandas as pd
import common as cmn
import os
import matplotlib.pyplot as plt

import pull_misspec_analysis_kalman

cmn.latex_look(plt)

# Pull folder
folder = cmn.pull_folder
prefixes = ["pull_process_kalman", "pull_frame_kalman", "pull_noise_kalman_25", "pull_noise_kalman_50",
            "pull_misspec_kalman_50"]
suffixes = ["_avg", "_99", "_999"]
suf_ylabels = ["avg. MSE", "MSE 99-th", "MSE 999-th"]
num_results = len(suffixes)

## Process
fig, axes = plt.subplots(ncols=1, nrows=num_results, figsize=(4, 15))
for s, suffix in enumerate(suffixes):
    filename = os.path.join(folder, prefixes[0] + suffix + '.csv')
    # read data
    results = pd.read_csv(filename).iloc[:, :].to_numpy().T

    axes[s].plot(results[0], results[1:].T, label=cmn.pull_scheduler_names)
    axes[s].set_ylabel(suf_ylabels[s])
    axes[s].set_xlabel(r"$\sigma_w$")

for i in range(num_results):
    axes[i].grid(True)
    axes[i].legend()
fig.tight_layout()
fig.show()

## Frame
fig, axes = plt.subplots(ncols=1, nrows=num_results, figsize=(4, 15))
for s, suffix in enumerate(suffixes):
    filename = os.path.join(folder, prefixes[1] + suffix + '.csv')
    # read data
    results = pd.read_csv(filename).iloc[:, :].to_numpy().T

    width = 0.25
    for i in range(3):
        offset = i * width
        axes[s].bar(results[0] + offset, results[i+1], width, label=cmn.pull_scheduler_names[i])
    # axes[i].set_title(cmn.results_label[i])
    axes[s].set_ylabel(suf_ylabels[s])
    axes[s].set_xlabel(r"$Q$")

for i in range(num_results):
    axes[i].grid(True)
    axes[i].legend()
fig.tight_layout()
fig.show()

for prefix in prefixes[2:4]:
    # Noise
    ## Process
    fig, axes = plt.subplots(ncols=1, nrows=num_results, figsize=(4, 15))
    for s, suffix in enumerate(suffixes):
        filename = os.path.join(folder, prefix + suffix + '.csv')
        # read data
        results = pd.read_csv(filename).iloc[:, :].to_numpy().T

        axes[s].plot(results[0], results[1:].T, label=cmn.pull_scheduler_names)
        axes[s].set_ylabel(suf_ylabels[s])
        axes[s].set_xlabel(r"$\hat{\sigma}_w$")

    for i in range(num_results):
        axes[i].grid(True)
        axes[i].legend()
    fig.tight_layout()
    fig.show()

# Mispecified
## Process
fig, axes = plt.subplots(ncols=1, nrows=num_results, figsize=(4, 15))
for s, suffix in enumerate(suffixes):
    filename = os.path.join(folder, prefixes[4] + suffix + '.csv')
    # read data
    results = pd.read_csv(filename).iloc[:, :].to_numpy().T

    axes[s].plot(results[0], results[1:].T, label=cmn.pull_scheduler_names)
    axes[s].set_ylabel(suf_ylabels[s])
    axes[s].set_xlabel(r"$\delta_F$")

for i in range(num_results):
    axes[i].grid(True)
    axes[i].legend()
fig.tight_layout()
fig.show()