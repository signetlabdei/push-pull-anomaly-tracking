import os

import numpy as np
from tqdm import tqdm
import sys


# folders
data_folder = os.path.join(os.getcwd(), 'data')
pull_folder = os.path.join(data_folder, 'pull_only')

# Main system parameters
N = 100
C = 4
D = 20

# Frame
max_num_frame = int(1e5)
frame_duration = 10e-3      # 10 ms
S = 20
S_step = 1
Q_vec = np.arange(5, S + S_step, S_step, dtype=int)
P_vec = np.arange(5, S + S_step, S_step, dtype=int)

### MISALIGNMENT PARAMETERS ###
p_11 = 0.9

# Basic load
abs_rate_int = np.array([1., 2., 3., 4., 5.])    # anomalies/node/s related to the multiplier

# Load parameters for quasi-heterogeneous clusters
qhet_p_01 = np.array([0.002332, 0.01642, 0.01691, 0.01749])
qhet_multipliers = np.array([1., 1.5984, 2.1425, 2.663, 3.17])

# Load parameters for heterogeneous clusters
het_p_01 = np.array([0.001, 0.005, 0.009, 0.013])
het_multipliers = np.array([1.974, 3.162, 4.245, 5.282, 6.292])

# Load parameters for heterogeneous clusters
hom_p_01 = het_p_01.mean() * np.ones(len(het_p_01))
hom_multipliers = np.array([1.82, 2.909, 3.899, 4.847, 5.771])

# Algorithm parameters
distributed_detection = 0.95
risk_step = 0.5
risk_thr_vec = np.arange(0., 1. + risk_step, risk_step, dtype=float)

# Schedulers
# Schemes
dist_schedulers = ['PPS', 'PPS-cluster', 'PPS-nodes', 'RR', 'random']

def std_bar(total_iteration):
    return tqdm(total_iteration, file=sys.stdout, leave=False, ncols=60, ascii=True)


# NumPy
def compute_ecdf(metric):
    sorted_vals = np.sort(metric)
    cdf_vals = np.linspace(0, 1, metric.shape[0])
    return sorted_vals, cdf_vals


