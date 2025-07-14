import numpy as np
from tqdm import tqdm
import sys

# Main system parameters
N = 100
C = 4
D = 20

# Frame
max_num_frame = int(1e5)
frame_duration = 10e-3      # 10 ms
S = 25
S_step = 5
Q_vec = np.arange(5, S + S_step, S_step, dtype=int)
P_vec = np.arange(5, S + S_step, S_step, dtype=int)

# Anomaly parameters
p_01 = 0.01
p_11 = 0.9

# Algorithm parameters
distributed_detection = 0.95
risk_step = 0.2
risk_thr_vec = np.arange(0., 1. + risk_step, risk_step, dtype=float)

def std_bar(total_iteration):
    return tqdm(total_iteration, file=sys.stdout, leave=False, ncols=60, ascii=True)


# NumPy
def compute_ecdf(metric):
    sorted_vals = np.sort(metric)
    cdf_vals = np.linspace(0, 1, metric.shape[0])
    return sorted_vals, cdf_vals


