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

# Anomaly parameters
# Distributed
p_11 = 0.9
# p_01_base = np.array([0.001, 0.00704, 0.00725, 0.00750])
# multiplier = np.array([1., 1.508, 1.939, 2.332])
# absorption_rate = np.array([0.250, 0.500, 0.750, 1.000])    # anomalies/node/s related to the multiplier

p_01_qhet = np.array([0.002332, 0.01642, 0.01691, 0.01749])
multiplier = np.array([1., 1.5984, 2.1425, 2.663, 3.17])
absorption_rate = np.array([1., 2., 3., 4., 5.])    # anomalies/node/s related to the multiplier

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


