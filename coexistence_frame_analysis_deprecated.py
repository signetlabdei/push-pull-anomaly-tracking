import numpy as np
import pandas as pd
import os
from push_scheduler import generate_anomalies, PushScheduler
from pull_scheduler import generate_drifts, PullScheduler
from push_pull_manager import ResourceManager
from common import N, R, C, D, T, E, M, max_age, SIGMA, ETA, het_p01, het_multipliers, p01_25, p11, dt_realign_thr, std_bar, coexistence_folder


def run_episode(num_bins: int, max_num_frame: int, resources: int,
                num_nodes: int, max_age: int, anomaly_rate: float, sigma: float, aoii_thr: int,
                cluster_size: int, num_cluster: int, p_01_vec: np.array or list, p_11: float, detection_thr: float,
                manager_type: int, push_resources: int = 2, hysteresis: float = 0.005,
                rng: np.random.Generator = np.random.default_rng(),
                debug_mode: bool = False):
    # Instantiate scheduler
    assert len(p_01_vec) == cluster_size, "The probability of detecting an anomaly has to be the size of the cluster"

    num_clustered_nodes = num_cluster * cluster_size  # The first clustered nodes have distributed anomalies

    # Instantiate schedulers
    local_sched = PushScheduler(num_nodes, max_age, anomaly_rate, 1, debug_mode)
    dist_sched = PullScheduler(num_clustered_nodes, cluster_size, p_01_vec, p_11, rng, debug_mode)
    manager = ResourceManager(manager_type, resources)
    if manager_type == 0:   # Set P beforehand
        manager.set_push_resources(push_resources)
    elif manager_type in [1, 2]:
        manager.set_min_threshold(push_resources)
        if manager_type == 2:
            manager.set_hysteresis(hysteresis)

    # Utility variables
    local_state = np.zeros(num_nodes)
    distributed_state = np.zeros(num_clustered_nodes, dtype=int)  # y(k) in the paper
    local_aoii = np.zeros((max_num_frame, num_nodes))
    distributed_aoii = np.zeros((max_num_frame, num_cluster))

    for k in std_bar(range(max_num_frame)):
        ### ANOMALY GENERATION ###
        # Local
        local_state = generate_anomalies(anomaly_rate, local_state, rng)
        # Distributed
        distributed_state = generate_drifts(p_01_vec, p_11, distributed_state, rng)

        # Compute distributed anomaly z^{(i)}(k)
        distributed_anomaly = np.asarray(np.sum(distributed_state.reshape(num_clustered_nodes // cluster_size, cluster_size), axis= 1)
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
        scheduled = dist_sched.schedule_pps(Q)
        # Fix local anomalies in scheduled slots
        local_aoii[k, scheduled] = 0
        local_state[scheduled] = 0

        ### PUSH-BASED SUBFRAME ###
        # Get local anomaly threshold
        threshold = local_sched.schedule(P, sigma, scheduled)
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
                                                             detection_thr)

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
    debug = False
    overwrite = True

    # Parameters
    P_vec = np.arange(2, 19)
    aoii_thr = 2


    # Order of saving data
    column_titles = ['ThetaAvg', 'Theta99', 'Theta999', 'PsiAvg', 'Psi99', 'Psi999']

    # Start cases
    for load in ['het']: # ['hom', 'het']:
        # Check if files exist and load it if there
        prefix = 'coexistence_frame_' + load
        filename = os.path.join(coexistence_folder, prefix + '.csv')

        if os.path.exists(filename):
            aoii = pd.read_csv(filename).iloc[:, 1:].to_numpy()
        else:
            aoii = np.full((len(P_vec), 6), np.nan)

        # Get load
        anomaly_rate = 0.03 if load == 'hom' else 0.035
        p_01 = het_p01 * het_multipliers[2] if load == 'hom' else p01_25

        # Start iterations
        for p, P in enumerate(P_vec):
            ### Logging ###
            print(f"Load {load}; P={P:02d}. Status:")

            # Check if data is there
            if overwrite or np.all(np.isnan(aoii[p])):
                if P != 2:
                    continue
                loca_aoii_hist = np.zeros(M + 1)
                dist_aoii_hist = np.zeros(M + 1)
                for ep in range(E):
                    print(f'\tEpisode: {ep:02d}/{E-1:02d}')
                    loc_tmp, dist_tmp = run_episode(M, T, R,
                                                    N, max_age, anomaly_rate, SIGMA, aoii_thr,
                                                    C, D, p_01, p11, dt_realign_thr,
                                                    0, P, ETA,
                                                    np.random.default_rng(ep), debug)
                    loca_aoii_hist += loc_tmp[0] / E
                    dist_aoii_hist += dist_tmp[0] / E

                # Local
                loca_aoii_cdf = np.cumsum(loca_aoii_hist)
                aoii[p, 0] = np.dot(loca_aoii_hist, np.arange(0, M + 1, 1))
                aoii[p, 1] = np.where(loca_aoii_cdf > 0.99)[0][0]
                aoii[p, 2] = np.where(loca_aoii_cdf > 0.999)[0][0]

                # Dist
                dist_aoii_cdf = np.cumsum(dist_aoii_hist)
                aoii[p, 3] = np.dot(dist_aoii_hist, np.arange(0, M + 1, 1))
                aoii[p, 4] = np.where(dist_aoii_cdf > 0.99)[0][0]
                aoii[p, 5] = np.where(dist_aoii_cdf > 0.999)[0][0]

                # Generate data frame and save it (redundant but to avoid to lose data for any reason)
                df = pd.DataFrame(aoii.round(dec), columns=column_titles)
                df.insert(0, 'P', P_vec)
                df.to_csv(filename, index=False)

            else:
                print("\t...already done!")
                continue
