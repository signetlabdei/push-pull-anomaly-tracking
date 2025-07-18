import numpy as np
import pandas as pd
import os
from local_scheduler import generate_local_anomalies, LocalAnomalyScheduler
from distributed_scheduler import generate_distributed_anomalies, DistributedAnomalyScheduler
from push_pull_manager import ResourceManager
from common import N, R, C, D, max_age, qhet_p_01, qhet_multipliers, p_11, dt_detection_thr, std_bar, coexistence_folder


def run_episode(num_bins: int, max_num_frame: int, resources: int,
                num_nodes: int, max_age: int, anomaly_rate: float, p_c: float, aoii_thr: int,
                cluster_size: int, num_cluster: int, risk_thr: float,  p_01_vec: np.array or list, p_11: float, detection_thr: float,
                manager_type: int,
                rng: np.random.Generator,
                debug_mode: bool = False):
    # Instantiate scheduler
    assert len(p_01_vec) == cluster_size, "The probability of detecting an anomaly has to be the size of the cluster"

    num_clustered_nodes = num_cluster * cluster_size  # The first clustered nodes have distributed anomalies

    # Instantiate schedulers
    local_sched = LocalAnomalyScheduler(num_nodes, max_age, anomaly_rate, 1, debug_mode)
    dist_sched = DistributedAnomalyScheduler(num_clustered_nodes, cluster_size, p_01_vec, p_11, debug_mode)
    manager = ResourceManager(manager_type, resources)
    # Set P beforehand
    manager.set_min_threshold(5)
    if manager_type == 2:
        manager.set_hysteresis(0.005)

    # Utility variables
    local_state = np.zeros(num_nodes)
    distributed_state = np.zeros(num_clustered_nodes, dtype=int)  # y(k) in the paper
    distributed_anomaly = np.zeros(num_cluster, dtype=int)  # z^{(i)}(k) in the paper
    local_aoii = np.zeros((max_num_frame, num_nodes))
    distributed_aoii = np.zeros((max_num_frame, num_cluster))

    for k in std_bar(range(max_num_frame)):
        ### ANOMALY GENERATION ###
        # Local
        local_state = generate_local_anomalies(anomaly_rate, local_state, rng)
        # Distributed
        distributed_state = generate_distributed_anomalies(p_01_vec, p_11, distributed_state, rng)

        # Compute distributed anomaly z^{(i)}(k)
        distributed_anomaly = np.asarray(
            np.sum(distributed_state.reshape(num_clustered_nodes // cluster_size, cluster_size), axis=1)
            >= cluster_size / 2, dtype=int)

        ### COMPUTE AOII ###
        local_aoii[k, :] = local_aoii[k - 1, :] + local_state if k > 0 else local_state
        distributed_aoii[k, :] = distributed_aoii[k - 1, :] + distributed_anomaly if k > 0 else distributed_anomaly

        ### UPDATE SCHEDULER PRIORS ###
        local_sched.update_prior()
        dist_sched.update_prior()

        ### SUBFRAME ALLOCATION ###
        local_risk = local_sched.get_risk(aoii_thr)
        dist_risk = dist_sched.get_average_risk
        P, Q = manager.allocate_resources(local_risk, dist_risk)  # Allocate resources
        if debug_mode:
            print('local_risk', local_risk, 'dist_risk', dist_risk, 'ratio', local_risk / dist_risk)
            print('P', P, 'Q', Q)

        ### PULL-BASED SUBFRAME ###
        # Get pull schedule
        scheduled = dist_sched.schedule(Q, risk_thr)
        # Fix local anomalies in scheduled slots
        local_aoii[k, scheduled] = 0
        local_state[scheduled] = 0

        ### PUSH-BASED SUBFRAME ###
        # Get local anomaly threshold
        threshold = local_sched.schedule(P, p_c, scheduled)
        # Select random slots for active nodes
        choices = rng.integers(1, P + 1, num_nodes) * np.asarray(local_aoii[k, :] > threshold)
        outcome = np.zeros(P, dtype=int)
        successful_push = []
        for p in range(1, P + 1):
            chosen = np.where(choices == p)[0]
            if chosen.size != 0:
                if chosen.size == 1:
                    if chosen[0] < num_clustered_nodes:
                        successful_push.append(chosen[0])
                    outcome[p - 1] = chosen[0] + 1
                    local_state[chosen[0]] = 0
                    local_aoii[k, chosen[0]] = 0
                else:
                    outcome[p - 1] = -1

        ### POST-FRAME UPDATE ###
        # Local and distributed anomaly belief update
        local_sched.update_psi(threshold, outcome)
        successful = np.append(scheduled, np.asarray(successful_push, dtype=int))
        cluster_in_anomaly = dist_sched.update_posterior_pmf(successful, distributed_state[successful],
                                                             dt_detection_thr)

        ### LOGGING ###
        if debug_mode:
            print('s', local_state)
            print('t', threshold)
            print('c', choices)
            print('out', outcome)
            print('la', local_aoii[k, :])
            print('sch', scheduled)
            print('o', distributed_state[successful])
            print('y', distributed_state)
            print('dr', dist_risk)
            print('z', distributed_anomaly)
            print('da', distributed_aoii[k, :])
            input("Press Enter to continue...")

        # Reset state, anomaly and aoii for cluster where an anomaly was found
        for cluster in cluster_in_anomaly:
            distributed_state[dist_sched.cluster_map == cluster] = 0
            distributed_anomaly[cluster] = 0
            distributed_aoii[k, cluster] = 0

    local_aoii_tot = np.reshape(local_aoii, max_num_frame * num_nodes)
    distributed_aoii_tot = np.reshape(distributed_aoii, max_num_frame * num_cluster)
    return (np.histogram(local_aoii_tot, bins=num_bins + 1, range=(-0.5, num_bins + 0.5), density=True),
            np.histogram(distributed_aoii_tot, bins=num_bins + 1, range=(-0.5, num_bins + 0.5), density=True))


if __name__ == '__main__':
    # Simulation variables
    dec = 6
    M = 200
    T = int(1e4)
    episodes = 10
    managers = np.array([1, 2])

    # Local
    local_anomaly_rate = 0.03
    p_c = 0.2

    # Distributed
    p_01 = qhet_p_01 * qhet_multipliers[2]
    dist_risk_thr = 0.5

    debug = False
    overwrite = False

    # Check if files exist and load it if there
    prefix = 'coexistence_manager'
    filename_avg = os.path.join(coexistence_folder, prefix + '_avg.csv')
    filename_99 = os.path.join(coexistence_folder, prefix + '_99.csv')
    filename_999 = os.path.join(coexistence_folder, prefix + '_999.csv')

    if os.path.exists(filename_avg) and not overwrite:
        prob_avg = pd.read_csv(filename_avg).iloc[:, 1:].to_numpy()
    else:
        prob_avg = np.full((len(managers), 2), np.nan)
    if os.path.exists(filename_99) and not overwrite:
        prob_99 = pd.read_csv(filename_99).iloc[1:, 1:].to_numpy()
    else:
        prob_99 = np.full((len(managers), 2), np.nan)
    if os.path.exists(filename_999) and not overwrite:
        prob_999 = pd.read_csv(filename_999).iloc[1:, 1:].to_numpy()
    else:
        prob_999 = np.full((len(managers), 2), np.nan)


    for m, manager in enumerate(managers):
        ### Logging ###
        print(f"Manager={manager:02d}. Status:")

        # Check if data is there
        if overwrite or np.all(np.isnan(prob_avg[m])):
            loca_aoii_hist = np.zeros(M + 1)
            dist_aoii_hist = np.zeros(M + 1)
            for ep in range(episodes):
                print(f'\tEpisode: {ep:02d}/{episodes-1:02d}')
                loc_tmp, dist_tmp = run_episode(M, T, R,
                                                N, max_age, local_anomaly_rate, p_c, 3,
                                                C, D, dist_risk_thr, p_01, p_11, dt_detection_thr,
                                                manager,
                                                np.random.default_rng(0), debug)
                loca_aoii_hist += loc_tmp[0] / episodes
                dist_aoii_hist += dist_tmp[0] / episodes

            # Local
            loca_aoii_cdf = np.cumsum(loca_aoii_hist)
            prob_99[m, 0] = np.where(loca_aoii_cdf > 0.99)[0][0]
            prob_999[m, 0] = np.where(loca_aoii_cdf > 0.999)[0][0]
            prob_avg[m, 0] = np.dot(loca_aoii_hist, np.arange(0, M + 1, 1))

            # Dist
            dist_aoii_cdf = np.cumsum(dist_aoii_hist)
            prob_99[m, 1] = np.where(dist_aoii_cdf > 0.99)[0][0]
            prob_999[m, 1] = np.where(dist_aoii_cdf > 0.999)[0][0]
            prob_avg[m, 1] = np.dot(dist_aoii_hist, np.arange(0, M + 1, 1))

            # Generate data frame and save it (redundant but to avoid to lose data for any reason)
            for res, file in [(prob_avg, filename_avg), (prob_99, filename_99), (prob_999, filename_999)]:
                df = pd.DataFrame(res.round(dec), columns=['Theta', 'Psi'])
                df.insert(0, 'managers', managers)
                df.to_csv(file, index=False)
        else:
            print("\t...already done!")
            continue
