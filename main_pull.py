import numpy as np
import matplotlib.pyplot as plt
import os
from distributed_scheduler import DistributedAnomalyScheduler
from common import max_num_frame, Q_vec, risk_thr_vec, C, D, p_01, p_11, distributed_detection, std_bar


# Simulation variables
rng = np.random.default_rng(0)
debug_mode = False
overwrite = False
data_folder = os.path.join(os.getcwd(), 'data')
# Population
cluster_size = C
num_cluster = D
num_clustered_nodes = num_cluster * cluster_size   # The first clustered nodes have distributed anomalies

for risk_thr in risk_thr_vec:
    for Q in Q_vec:
        ### Logging ###
        print(f"risk_thr={risk_thr:1.1f}, Q={Q:02d} status:")
        filename = f'pull_results_riskth{risk_thr:0.1f}_Q{Q:02d}.npz'

        # Check if results file exists
        if not overwrite and os.path.exists(os.path.join(data_folder, filename)):
            print("\tResults already exists, skipping.")
            continue

        # Utility variables
        distributed_state = np.zeros(num_clustered_nodes)
        distributed_anomaly = np.zeros((max_num_frame, num_cluster), dtype=int)  # z^{(i)}(k) in the paper
        distributed_aoii = np.zeros((max_num_frame, num_cluster))

        # Instantiate scheduler
        dist_sched = DistributedAnomalyScheduler(num_clustered_nodes, cluster_size, p_01, p_11, debug_mode)

        for k in std_bar(range(max_num_frame)):
            ### ANOMALY GENERATION ###
            new_state = np.zeros(num_clustered_nodes)
            for i in range(num_clustered_nodes):
                cluster_id = int(np.floor(i / cluster_size))
                p = p_01
                if distributed_state[i] == 1:
                    anomaly = np.sum(distributed_state[cluster_size * (cluster_id - 1): cluster_size * cluster_id]) >= cluster_size / 2
                    if anomaly:
                        p = 1
                    else:
                        p = p_11
                new_state[i] = rng.random() < p
            distributed_state = new_state

            # Compute z^{(i)}(k)
            for cluster in range(num_cluster):
                distributed_anomaly[k, cluster] = np.sum(distributed_state[dist_sched.cluster_map == cluster]) >= cluster_size / 2

            # Compute AoII
            if k > 0:
                distributed_aoii[k, :] = distributed_aoii[k - 1, :] + distributed_anomaly[k, :]
            else:
                distributed_aoii[k, :] = distributed_anomaly[k, :]

            ### PULL-BASED SUBFRAME ###
            # Get pull schedule
            scheduled = dist_sched.schedule(Q, risk_thr)

            ### POST-FRAME UPDATE ###
            # Distributed anomaly belief update
            successful = scheduled # Ignoring push
            cluster_in_anomaly = dist_sched.update_posterior_pmf(successful, distributed_state[successful], distributed_detection)
            # Reset state, anomaly and aoii for cluster where an anomaly was found
            for cluster in cluster_in_anomaly:
                distributed_state[cluster_size * (cluster - 1): cluster_size * cluster] = 0
                distributed_anomaly[k, cluster] = 0
                distributed_aoii[k, cluster] = 0

            ### DEBUG ###
            if debug_mode:
                print('y', distributed_state)
                print('z', distributed_anomaly[k, :])
                print('a', distributed_aoii[k, :])
                input("Press Enter to continue...")

        np.savez(os.path.join(data_folder, filename), distributed_anomaly, distributed_aoii)
