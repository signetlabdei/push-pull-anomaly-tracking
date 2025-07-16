import numpy as np
import matplotlib.pyplot as plt
import os
from distributed_scheduler import DistributedAnomalyScheduler
from common import risk_thr_vec, C, D, p_01_base, multiplier, p_11, distributed_detection, std_bar, absorption_rate


def run_episode(num_bins: int, cluster_size: int, num_cluster: int, max_num_frame: int,
                pull_res: int, p_01_vec: np.array or list, p_11: float,
                risk_thr: float, detection_thr: float, debug_mode: bool):
    """ Generate a single episode

    :param cluster_size:
    :param num_cluster:
    :param max_num_frame:
    :param pull_res:
    :param p_01_vec: vector containing [p_01, p_11] # TODO: it should contain the p_01 per node!
    :param p_11: float
    :param risk_thr:
    :param detection_thr:
    :param debug_mode:
    :return:
    """
    # Instantiate scheduler
    assert len(p_01_vec) == cluster_size, "The probability of detecting an anomaly has to be the size of the cluster"

    num_clustered_nodes = num_cluster * cluster_size   # The first clustered nodes have distributed anomalies
    dist_sched = DistributedAnomalyScheduler(num_clustered_nodes, cluster_size, p_01_vec, p_11, debug_mode)

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
            # Rearrange the 01 transition probability on a cluster basis
            p = p_01_vec[int(np.mod(node,cluster_size))]
            if distributed_state[node] == 1:
                # Check if the anomaly is present
                if np.sum(distributed_state[dist_sched.cluster_map == cluster]) >= cluster_size / 2:
                    p = 1.
                else:
                    p = p_11
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
        cluster_in_anomaly = dist_sched.update_posterior_pmf(successful,
                                                             distributed_state[successful],
                                                             detection_thr)

        ### DEBUG for visualization ###
        if debug_mode:
            risk = dist_sched.get_cluster_risk
            print('s', scheduled)
            print('o', distributed_state[successful])
            print('y', distributed_state)
            print('r', risk)
            print('z', distributed_anomaly)
            print('a', distributed_aoii[k, :])
            input("Press Enter to continue...")

        # Reset state, anomaly and aoii for cluster where an anomaly was found
        for cluster in cluster_in_anomaly:
            distributed_state[dist_sched.cluster_map == cluster] = 0
            distributed_anomaly[cluster] = 0
            distributed_aoii[k, cluster] = 0

    distributed_aoii_tot = np.reshape(distributed_aoii, max_num_frame * num_cluster)
    return np.histogram(distributed_aoii_tot, bins=num_bins + 1, range=(-0.5, num_bins + 0.5), density=True)


if __name__ == '__main__':
    # Save values
    data_folder = os.path.join(os.getcwd(), 'data', 'pull_only')

    # Simulation variables
    M = 100
    T = int(1e4)
    episodes = 10
    rng = np.random.default_rng(0)
    p_01 = p_01_base * multiplier[1]

    debug_mode = False
    overwrite = False

    # Population
    cluster_size = C
    num_cluster = D
    Q_vec = [10, 20]
    risk_thr = 0.5

    prob_avg = np.zeros((len(Q_vec), len(risk_thr_vec)))
    prob_99 = np.zeros((len(Q_vec), len(risk_thr_vec)))
    prob_999 = np.zeros((len(Q_vec), len(risk_thr_vec)))

    for m, mult in enumerate(multiplier):
        for q, Q in enumerate(Q_vec):
            p_01 = p_01_base * mult
            ### Logging ###
            print(f"load={absorption_rate[m]:1.3f}, Q={Q:02d} status:")

            dist_aoii_hist = np.zeros(M + 1)
            for ep in range(episodes):
                print(f'\tEpisode: {ep:02d}/{episodes-1:02d}')
                dist_aoii_hist += run_episode(M, cluster_size, num_cluster, T, Q, p_01, p_11, risk_thr, distributed_detection, debug_mode)[0] / episodes
            dist_aoii_cdf = np.cumsum(dist_aoii_hist)
            prob_99[q, m] = np.where(dist_aoii_cdf > 0.99)[0][0]
            prob_999[q, m] = np.where(dist_aoii_cdf > 0.999)[0][0]
            prob_avg[q, m] = np.dot(dist_aoii_hist, np.arange(0, M + 1, 1))


            np.savetxt(os.path.join(data_folder, "pull_load_avg.csv"), prob_avg, delimiter=",")
            np.savetxt(os.path.join(data_folder, "pull_load_99.csv"), prob_99, delimiter=",")
            np.savetxt(os.path.join(data_folder, "pull_load_999.csv"), prob_999, delimiter=",")
