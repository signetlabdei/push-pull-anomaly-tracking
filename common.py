import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from tqdm import tqdm
import sys


# folders
data_folder = os.path.join(os.getcwd(), 'data')
pull_folder = os.path.join(data_folder, 'pull_only')
push_folder = os.path.join(data_folder, 'push_only')
coexistence_folder = os.path.join(data_folder, 'coexistence')

# Main system parameters
M = 100             # Num bins histogram
bins = 100000       # Histogram bins (Kalman)
maxval = 100        # Histogram max value (Kalman)
N = 100             # Num nodes N_a
C = 4               # cluster size
D = 20              # Num clusters
T = int(1e3)        # Num frames
E = 100             # num episodes

# Resource grid
frame_duration = 10e-3      # 10 ms
R = 20

# Anomaly parameters
max_age = 100       # Max age to save
SIGMA = 0.2         # PPS collision threshold

### DRIFT PARAMETERS ###
p11 = 0.9
# Load parameters for quasi-heterogeneous clusters
het_p01 = np.array([0.002332, 0.01642, 0.01691, 0.01749])
het_multipliers = np.array([1., 1.5984, 2.1425, 2.663, 3.17])
# Load parameters for heterogeneous clusters
hom_p01 = 0.007 * np.ones(C)
hom_multipliers = np.array([1.82, 2.909, 3.899, 4.847, 5.771])
# Load for quasi-heterogeneous with total load 0.25
p01_25 = np.array([0.00385, 0.031, 0.032, 0.033])
# Algorithm parameters
dt_realign_thr = 0.95
ETA = 0.005

### KALMAN DRIFT PARAMETERS ###
cent = 1
side = - 1 / 9

F = np.zeros((C, C))
for i in range(C):
    F[i, i] = cent
    F[i, i - 1] = side
    F[i, np.mod(i + 1, C)] = side

# F = np.asarray([[0.8, 0.1, 0, -0.2], [0.2, 0.8, -0.1, 0], [0, 0.2, 8/9, -0.1], [0, 0.2, 0.3, 8/9]])

sigma_w = 0.25
sigma_v = 0.2
sigma_w_hat = 0.25
sigma_v_hat = 0.2
H = np.eye(C)

# Schedulers
# Schemes
push_scheduler_names = ['MAF', 'FSA', 'AFSA', 'PPS']
pull_scheduler_names = ['PPS', 'CRA', 'MAF']

def std_bar(total_iteration):
    return tqdm(total_iteration, file=sys.stdout, leave=False, ncols=60, ascii=True)

def common_parser():
    parser = ArgumentParser(description="Common parser to run simulations.")

    parser.add_argument('--parallel', action='store_true', help="Enable parallel processing")
    parser.add_argument('--savedir', type=str, default=None, help="Optional path to save results in")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite results eve if already present in the savedir")

    args = parser.parse_args()
    return args.parallel, args.savedir, args.debug, args.overwrite

def check_data(data: tuple, prefix: str, folder: str, overwrite_flag: bool = False):
    """Check if a csv with results exists. If no overwrite flag is given, it tries to add points given by data_shape
    to the new results"""
    # Generate filename
    filename = os.path.join(folder, prefix + '.csv')
    # Take size of the data
    cols = data[0]
    rows = data[1]
    data_shape = (len(cols), len(rows))
    # Generate empty result matrix
    results = np.full(data_shape, np.nan)

    # Check existence
    if os.path.exists(filename) and not overwrite_flag:
        df = pd.read_csv(filename)
        existing_x = df.iloc[:, 0].values  # Existing x values (first column)
        existing_y = df.iloc[:, 1:].values  # Existing y values (other columns)

        # Find the indices in `rows` where the x values exist in `existing_x`
        mask = np.isin(rows, existing_x)

        # For the matching rows, find their positions in existing_x
        # Use np.searchsorted or a dictionary for mapping
        x_to_idx = {x: i for i, x in enumerate(existing_x)}
        row_indices = np.array([x_to_idx[x] for x in rows[mask]])

        # Assign the values from existing_y to results
        results[:, mask] = existing_y[row_indices, :].T  # Transpose to align dimensions
    return results, filename

def latex_look(plt):
    # Set LaTeX as the default text interpreter
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman']  # LaTeX default font
