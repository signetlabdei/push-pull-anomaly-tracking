import os
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm
import sys


# folders
data_folder = os.path.join(os.getcwd(), 'data')
pull_folder = os.path.join(data_folder, 'pull_only')
push_folder = os.path.join(data_folder, 'push_only')
coexistence_folder = os.path.join(data_folder, 'coexistence')

# Main system parameters
M = 100             # Num bins histogram
N = 100             # Num nodes N_a
C = 4               # cluster size
D = 20              # Num clusters
T = int(1e3)        # Num frames
E = 100             # num episodes
RNG = np.random.default_rng(0)  # Random Number Generator

# Resource grid
frame_duration = 10e-3      # 10 ms
R = 20

# Anomaly parameters
max_age = 100       # Max age to save
SIGMA = 0.2         # PPS collision threshole

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

