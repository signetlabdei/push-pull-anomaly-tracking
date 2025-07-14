import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from common import Q_vec, risk_thr_vec, max_num_frame, D, compute_ecdf, frame_duration

data_folder = os.path.join(os.getcwd(), 'data')

for risk_thr in [0., 0.2]:

    distributed_aoii = np.zeros((len(Q_vec), max_num_frame, D))

    for q, Q in enumerate(Q_vec):
        filename = os.path.join(data_folder, f'pull_results_riskth{risk_thr:0.1f}_Q{Q:02d}.npz')

        if os.path.exists(filename):
            print(f"Reading risk_thr={risk_thr:1.1f}, Q={Q:02d}")
            files = np.load(filename)
            distributed_aoii[q] = files['arr_1']

    # conversion
    distributed_aoii_ms = distributed_aoii * frame_duration / 1e-3

    # Avg
    avg_aoii = np.mean(distributed_aoii_ms, axis=(1, 2))
    df_avg = pd.DataFrame({'Q': Q_vec, 'p1': avg_aoii})
    df_avg.to_csv(os.path.join(data_folder, "pull_frame_avg.csv"), index=False)

    # Flatting over the clusters
    flat_aoii_ms = distributed_aoii_ms.reshape((len(Q_vec), -1))
    # percentile99
    percentile_99 = np.percentile(flat_aoii_ms, 99, axis=1)
    df_p99 = pd.DataFrame({'Q': Q_vec, 'p1': percentile_99})
    df_p99.to_csv(os.path.join(data_folder, "pull_frame_99.csv"), index=False)

    # percentule999
    percentile_999 = np.percentile(flat_aoii_ms, 99.9, axis=1)
    df_p999 = pd.DataFrame({'Q': Q_vec, 'p1': percentile_999})
    df_p999.to_csv(os.path.join(data_folder, "pull_frame_999.csv"), index=False)

    q_indices = Q_vec

    plt.figure(figsize=(10, 5))
    width = 0.25

    # Figure just because
    plt.bar(q_indices - width, avg_aoii, width=width, label='Average')
    plt.bar(q_indices, percentile_99, width=width, label='99th Percentile')
    plt.bar(q_indices + width, percentile_999, width=width, label='99.9th Percentile')

    plt.xlabel(r'number of pull REs $Q$')
    plt.ylabel('AoII [ms]')
    plt.title('Average and Percentiles vs Q')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # CDF
    plt.figure(figsize=(10, 6))
    for q, Q in enumerate(Q_vec):
        x, y = compute_ecdf(flat_aoii_ms[q])
        plt.plot(x, y, label=f'Q={Q}')
        df_ecdf = pd.DataFrame({
            'aoii': x,  # x-axis: sorted values
            'p1': y  # y-axis: ECDF values
        })
        df_ecdf.to_csv(os.path.join(data_folder,f"pull_cdf_Q{q:02d}.csv"), index=False)
    plt.xlabel('AoII [ms]')
    plt.ylabel('ECDF')
    plt.title('Empirical CDF per Q')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Save cdf




