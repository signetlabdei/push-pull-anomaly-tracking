import numpy as np
import matplotlib.pyplot as plt
import os
from distributed_scheduler import DistributedAnomalyScheduler
from common import Q_vec, risk_thr_vec, C, D, p_01, p_11, distributed_detection, std_bar


def run_episode(num_bins, cluster_size, num_cluster, max_num_frame, pull_res, p_vec, risk_thr, detection_thr, debug_mode):
    """ Generate a single episode

    :param cluster_size:
    :param num_cluster:
    :param max_num_frame:
    :param pull_res:
    :param p_vec: vector containing [p_01, p_11]
    :param risk_thr:
    :param detection_thr:
    :param debug_mode:
    :return:
    """
    # Instatiate scheduler
    num_clustered_nodes = num_cluster * cluster_size   # The first clustered nodes have distributed anomalies
    dist_sched = DistributedAnomalyScheduler(num_clustered_nodes, cluster_size, p_vec[0], p_vec[1], debug_mode)

    # Utility variables
    distributed_state = np.zeros(num_clustered_nodes, dtype=int)    # y(k) in the paper
    distributed_anomaly = np.zeros(num_cluster, dtype=int)          # z^{(i)}(k) in the paper
    distributed_aoii = np.zeros((max_num_frame, num_cluster))

    # Start frames
    for k in std_bar(range(max_num_frame)):

        ### ANOMALY GENERATION ###
        new_state = np.zeros(num_clustered_nodes)
        for node in range(num_clustered_nodes):
            cluster = dist_sched.cluster_map[node]
            p = p_vec[0]
            if distributed_state[node] == 1:
                # Check if the anomaly is present
                if np.sum(distributed_state[dist_sched.cluster_map == cluster]) >= cluster_size / 2:
                    p = 1
                else:
                    p = p_vec[1]
            new_state[node] = rng.random() < p
        distributed_state = new_state

        # Compute distributed state z^{(i)}(k)
        for cluster in range(num_cluster):
            distributed_anomaly[cluster] = np.sum(distributed_state[dist_sched.cluster_map == cluster]) >= cluster_size / 2

        # Compute AoII
        if k > 0:
            distributed_aoii[k, :] = distributed_aoii[k - 1, :] + distributed_anomaly
        else:
            distributed_aoii[k, :] = distributed_anomaly

        ### PULL-BASED SUBFRAME ###
        # Get pull schedule
        scheduled = dist_sched.schedule(pull_res, risk_thr)

        ### POST-FRAME UPDATE ###
        # Distributed anomaly belief update
        successful = scheduled  # Ignoring push
        cluster_in_anomaly = dist_sched.update_state_pmf(successful,
                                                         distributed_state[successful],
                                                         detection_thr)
        # Reset state, anomaly and aoii for cluster where an anomaly was found
        for cluster in cluster_in_anomaly:
            distributed_state[dist_sched.cluster_map == cluster] = 0
            distributed_anomaly[cluster] = 0
            distributed_aoii[k, cluster] = 0

        ### DEBUG ###
        if debug_mode:
            print('y', distributed_state)
            print('z', distributed_anomaly)
            print('a', distributed_aoii[k, :])
            input("Press Enter to continue...")

    distributed_aoii_tot = np.reshape(distributed_aoii, max_num_frame * num_cluster)
    return np.histogram(distributed_aoii_tot, bins=num_bins + 1, range=(-0.5, num_bins + 0.5), density=True)



if __name__ == '__main__':
    # Save values
    data_folder = os.path.join(os.getcwd(), 'data')

    # Simulation variables
    M = 100
    T = int(1e3)
    episodes = 100
    rng = np.random.default_rng(0)

    debug_mode = True
    overwrite = False

    # Population
    cluster_size = C
    num_cluster = D

    risk_thr_vec = [0.4]

    prob_avg = np.zeros((len(Q_vec), len(risk_thr_vec)))
    prob_99 = np.zeros((len(Q_vec), len(risk_thr_vec)))
    prob_999 = np.zeros((len(Q_vec), len(risk_thr_vec)))

    for r, risk_thr in enumerate(risk_thr_vec):
        for q, Q in enumerate(Q_vec):
            ### Logging ###
            print(f"risk_thr={risk_thr:1.1f}, Q={Q:02d} status:")

            dist_aoii_hist = np.zeros(M + 1)
            for ep in range(episodes):
                print(f'\tEpisode: {ep:02d}/{episodes-1:02d}')
                dist_aoii_hist += run_episode(M, C, D, T, Q, [p_01, p_11], risk_thr, distributed_detection, debug_mode)[0] / episodes
            dist_aoii_cdf = np.cumsum(dist_aoii_hist)
            prob_99[q, r] = np.where(dist_aoii_cdf > 0.99)[0][0]
            prob_999[q, r] = np.where(dist_aoii_cdf > 0.999)[0][0]
            prob_avg[q, r] = np.dot(dist_aoii_hist, np.arange(0, M + 1, 1))


            np.savetxt(os.path.join(data_folder, "pull_frame_avg.csv"), prob_avg, delimiter=",")
            np.savetxt(os.path.join(data_folder, "pull_frame_99.csv"), prob_99, delimiter=",")
            np.savetxt(os.path.join(data_folder, "pull_frame_999.csv"), prob_999, delimiter=",")
