import numpy as np
import pandas as pd
import os
from distributed_scheduler import DistributedAnomalyScheduler, DistributedRoundRobin, DistributedRandom
from common import C, D, p_01_qhet, multiplier, p_11, distributed_detection, std_bar, absorption_rate, dist_schedulers, pull_folder



def run_episode(sched_type: int,
                num_bins: int, cluster_size: int, num_cluster: int, max_num_frame: int,
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
    if sched_type == 0:
        dist_sched = DistributedAnomalyScheduler(num_clustered_nodes, cluster_size, p_01_vec, p_11, debug_mode)
    elif sched_type == 1:
        dist_sched = DistributedAnomalyScheduler(num_clustered_nodes, cluster_size, p_01_vec, p_11, debug_mode)
        risk_thr = 0.
    elif sched_type == 2:
        dist_sched = DistributedAnomalyScheduler(num_clustered_nodes, cluster_size, p_01_vec, p_11, debug_mode)
        risk_thr = 1.
    elif sched_type == 3:
        dist_sched = DistributedRoundRobin(num_clustered_nodes, cluster_size, p_01_vec, p_11, debug_mode)
    elif sched_type == 4:
        dist_sched = DistributedRandom(num_clustered_nodes, cluster_size, p_01_vec, p_11, debug_mode)

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
        if sched_type != 3:
            scheduled = dist_sched.schedule(pull_res, risk_thr)
        else:
            scheduled = dist_sched.schedule(pull_res, k)

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
    # Simulation variables
    dec = 6
    M = 200
    T = int(1e4)
    episodes = 10
    rng = np.random.default_rng(0)
    schedulers = dist_schedulers

    debug_mode = False
    overwrite = True

    # Population
    cluster_size = C
    num_cluster = D

    # Parameters
    Q = 10
    risk_thr = 0.5

    # Check if files exist and load it if there
    prefix = 'pull_load'
    filename_avg = os.path.join(pull_folder, prefix + '_avg.csv')
    filename_99 = os.path.join(pull_folder, prefix + '_99.csv')
    filename_999 = os.path.join(pull_folder, prefix + '_999.csv')
    filename_cdf = os.path.join(pull_folder, prefix + '_cdf.csv')

    if os.path.exists(filename_avg) and not overwrite:
        prob_avg = pd.read_csv(filename_avg).iloc[:, 1:].to_numpy()
    else:
        prob_avg = np.full((len(schedulers), len(multiplier)), np.nan)
    if os.path.exists(filename_99) and not overwrite:
        prob_99 = pd.read_csv(filename_99).iloc[:, 1:].to_numpy()
    else:
        prob_99 = np.full((len(schedulers), len(multiplier)), np.nan)
    if os.path.exists(filename_999) and not overwrite:
        prob_999 = pd.read_csv(filename_999).iloc[:, 1:].to_numpy()
    else:
        prob_999 = np.full((len(schedulers), len(multiplier)), np.nan)
    if os.path.exists(filename_cdf) and not overwrite:
        cdf = pd.read_csv(filename_cdf).iloc[:, 1:].to_numpy()
    else:
        cdf = np.full((len(multiplier), M + 1), np.nan)

    for s, _ in enumerate(schedulers):
        for m, mult in enumerate(multiplier):
            p_01 = p_01_qhet * mult
            ### Logging ###
            print(f"load={absorption_rate[m]:1.3f}, sched={schedulers[s]}. Status:")

            # Check if data is there
            if overwrite or np.isnan(prob_avg[m, s]):
                dist_aoii_hist = np.zeros(M + 1)
                for ep in range(episodes):
                    print(f'\tEpisode: {ep:02d}/{episodes-1:02d}')
                    dist_aoii_hist += run_episode(s, M, cluster_size, num_cluster, T, Q, p_01, p_11, risk_thr, distributed_detection, debug_mode)[0] / episodes
                dist_aoii_cdf = np.cumsum(dist_aoii_hist)
                prob_99[s, m] = np.where(dist_aoii_cdf > 0.99)[0][0]
                prob_999[s, m] = np.where(dist_aoii_cdf > 0.999)[0][0]
                prob_avg[s, m] = np.dot(dist_aoii_hist, np.arange(0, M + 1, 1))
                # Save CDF for the PPS-0.5 only
                if s == 0:
                    cdf[m] = dist_aoii_cdf

                # Generate data frame and save it (redundant but to avoid to lose data for any reason)
                # Generate data frame and save it (redundant but to avoid to lose data for any reason)
                for res, file in [(prob_avg, filename_avg), (prob_99, filename_99), (prob_999, filename_999)]:
                    df = pd.DataFrame(res.round(dec), columns=schedulers)
                    df.insert(0, 'abs_rate', absorption_rate)
                    df.to_csv(file, index=False)

                if s == 0:
                    cdf_df = pd.DataFrame(cdf.T, columns=absorption_rate)
                    cdf_df.insert(0, 'Psi', np.arange(M + 1))
                    cdf_df.to_csv(filename_cdf, index=False)

            else:
                print("\t...already done!")
                continue
