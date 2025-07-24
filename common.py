import os

import numpy as np
from tqdm import tqdm
import sys


# folders
data_folder = os.path.join(os.getcwd(), 'data')
pull_folder = os.path.join(data_folder, 'pull_only')
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
R_step = 1
Q_vec = np.arange(5, R + R_step, R_step, dtype=int)
P_vec = np.arange(5, R + R_step, R_step, dtype=int)

# Anomaly parameters
max_age = 100
SIGMA = 0.2

### MISALIGNMENT PARAMETERS ###
p11 = 0.9

# Load parameters for quasi-heterogeneous clusters
het_p01 = np.array([0.002332, 0.01642, 0.01691, 0.01749])
het_multipliers = np.array([1., 1.5984, 2.1425, 2.663, 3.17])

# Load parameters for heterogeneous clusters
hom_p01 = 0.007 * np.ones(C)
hom_multipliers = np.array([1.82, 2.909, 3.899, 4.847, 5.771])

p01_25 = np.array([0.00385, 0.031, 0.032, 0.033])

# Algorithm parameters
dt_detection_thr = 0.95
ETA = 0.005

# Schedulers
# Schemes
dist_schedulers = ['PPS', 'CRA', 'MAF']

def std_bar(total_iteration):
    return tqdm(total_iteration, file=sys.stdout, leave=False, ncols=60, ascii=True)

