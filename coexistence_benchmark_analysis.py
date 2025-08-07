import os
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

from push_scheduler import generate_anomalies, PushScheduler, PushMAFScheduler, PushAlohaScheduler
from pull_scheduler import generate_drifts, PullScheduler
from push_pull_manager import ResourceManager
from common import N, R, C, D, T, E, M, RNG, max_age, SIGMA, ETA, het_p01, het_multipliers, p01_25, p11, dt_realign_thr, std_bar, coexistence_folder
import common as cmn


def run_episode(episode_idx: int,
                push_sched_type: int, pull_sched_type: int,
                num_bins: int, max_num_frame: int, resources: int,
                num_nodes: int, max_age: int, anomaly_rate: float, collision_thr: float, aoii_thr: int,
                cluster_size: int, num_cluster: int, p_01_vec: np.ndarray, p_11: float, realign_thr: float,
                manager_type: int, push_resources: int = 2, hysteresis: float = 0.005,
                debug_mode: bool = False):
    r"""Run a single episode of a push-pull scenario. Parallelization allowed.

        :param episode_idx: The index of the episode to run.
        :param push_sched_type: The type of push scheduler to use.
        :param pull_sched_type: The type of pull scheduler to use.
        :param num_bins: The number of bins to use for the output histogram.
        :param max_num_frame: The maximum number of frames to simulate.
        :param resources: The amount of resources available :math:`R`.
        :param num_nodes: The number of nodes :math:`N`.
        :param max_age: The maximum age that can be saved for anomalies.
        :param anomaly_rate: The anomaly rate :math:`\rho_a`.
        :param collision_thr: The collision threshold :math:`\sigma`.
        :param aoii_thr: The AoII risk threshold :math:`\hat{\theta}`.
        :param cluster_size: The size of the clusters :math:`C`.
        :param num_cluster: The number of clusters :math:`D`.
        :param p_01_vec: The probability of detecting a drift (0 to 1) for each node in a cluster.
        :param p_11: The probability of remaining in the drift state (equal for each node in the cluster).
        :param realign_thr: The DT re-alignment threshold :math:`\nu_{reset}`.
        :param manager_type: The type of manager to use (0: fixed resources, 1: RSM, 2:SSM).
        :param push_resources: The amount of resources available for push if manager_type = 0 :math:`P` or the minimum resources :math:`R_{min}` otherwise.
        :param hysteresis: The SSM hysteresis threshold :math:`\eta`. Used only if manager_type = 2.
        :param debug_mode: If true, run in debug mode.
        :returns Tuple containing: (histogram of anomaly AoII, histogram of DT drift AoII)
        """
    # Instantiate scheduler
    assert len(p_01_vec) == cluster_size, "The probability of detecting an anomaly has to be the size of the cluster"

    rng = np.random.default_rng(episode_idx)

    # Instantiate manager
    manager = ResourceManager(manager_type, resources)
    if manager_type == 0:  # Set P beforehand
        manager.set_push_resources(push_resources)
    elif manager_type in [1, 2]:
        manager.set_min_threshold(push_resources)
        if manager_type == 2:
            manager.set_hysteresis(hysteresis)

    # Instantiate schedulers
    if push_sched_type == 0:
        push_scheduler = PushMAFScheduler(num_nodes)
    elif push_sched_type == 1:
        tx_rate = 0.9 / (num_nodes * anomaly_rate / push_resources)
    elif push_sched_type == 2:
        push_scheduler = PushAlohaScheduler(num_nodes, anomaly_rate, push_resources)
    else: # push_sched_type == 3:
        push_scheduler = PushScheduler(num_nodes, max_age, anomaly_rate, 1, debug_mode)

    num_clustered_nodes = num_cluster * cluster_size  # The first clustered nodes have distributed anomalies
    pull_scheduler = PullScheduler(num_clustered_nodes, cluster_size, p_01_vec, p_11, rng, debug_mode)


    # Utility variables
    anomaly_state = np.zeros(num_nodes)
    drift_state = np.zeros(num_clustered_nodes, dtype=int)  # y(k) in the paper
    anomaly_aoii = np.zeros((max_num_frame, num_nodes))
    drift_aoii = np.zeros((max_num_frame, num_cluster))

    for k in std_bar(range(max_num_frame)):
        ### ANOMALY GENERATION ###
        # Local
        anomaly_state = generate_anomalies(anomaly_rate, anomaly_state, rng)
        # Distributed
        drift_state = generate_drifts(p_01_vec, p_11, drift_state, rng)

        # Compute distributed anomaly z^{(i)}(k)
        drift_detected = np.asarray(np.sum(drift_state.reshape(num_clustered_nodes // cluster_size, cluster_size), axis= 1)
                                         >= cluster_size / 2, dtype=int)

        ### COMPUTE AOII ###
        anomaly_aoii[k, :] = anomaly_aoii[k - 1, :] + anomaly_state if k > 0 else anomaly_state
        drift_aoii[k, :] = drift_aoii[k - 1, :] + drift_detected if k > 0 else drift_detected

        ### UPDATE SCHEDULER PRIORS ###
        if push_sched_type == 3:
            push_scheduler.update_prior()
        pull_scheduler.update_prior()

        ### SUBFRAME ALLOCATION ###
        if push_sched_type == 3:
            anomaly_risk = push_scheduler.get_risk(aoii_thr)
        else:
            anomaly_risk = 0.
        drift_risk = pull_scheduler.get_average_risk
        P, Q = manager.allocate_resources(anomaly_risk, drift_risk)  # Allocate resources
        if debug_mode:
            print('anomaly_risk', anomaly_risk, 'drift_risk', drift_risk, 'ratio', anomaly_risk / drift_risk)
            print('P', P, 'Q', Q)

        ### PULL-BASED SUBFRAME ###
        # Get pull scheduler
        if pull_sched_type == 0:
            scheduled = pull_scheduler.schedule_pps(Q)
        elif pull_sched_type == 1:
            scheduled = pull_scheduler.schedule_cra(Q)
        else:  # dist_sched_type == 2
            scheduled = pull_scheduler.schedule_maf(Q, k)

        # Fix local anomalies in scheduled slots
        anomaly_aoii[k, scheduled] = 0
        anomaly_state[scheduled] = 0

        ### PUSH-BASED SUBFRAME ###
        if push_sched_type == 0:
            outcome = push_scheduler.schedule(P, scheduled)
            anomaly_state[outcome] = 0
            anomaly_aoii[k, outcome] = 0
            successful_push = outcome[outcome < num_clustered_nodes]
        else:
            if push_sched_type == 1:
                choices = np.random.randint(1, P + 1, num_nodes) * np.asarray(anomaly_aoii[k, :] > 0) * (rng.random(num_nodes) < tx_rate)
            elif push_sched_type == 2:
                choices = np.random.randint(1, P + 1, num_nodes) * np.asarray(anomaly_aoii[k, :] > 0) * (rng.random(num_nodes) < push_scheduler.rate)
            else:    # local_sched_type == 3:
                # Get local anomaly threshold
                threshold = push_scheduler.schedule(P, collision_thr, scheduled)
                # Select random slots for active nodes
                choices = rng.integers(1, P + 1, num_nodes) * np.asarray(anomaly_aoii[k, :] > threshold)

            # Check the outcome of the random access subframe
            outcome = np.zeros(P, dtype=int)
            successful_push = []
            for p in range(1, P + 1):
                chosen = np.where(choices == p)[0]
                if chosen.size != 0:
                    if chosen.size == 1:
                        if chosen[0] < num_clustered_nodes:
                            successful_push.append(chosen[0])
                        outcome[p - 1] = chosen[0] + 1
                        anomaly_state[chosen[0]] = 0
                        anomaly_aoii[k, chosen[0]] = 0
                    else:
                        outcome[p - 1] = -1

        ### POST-FRAME UPDATE ###
        # Local and distributed anomaly belief update
        if push_sched_type == 2:
            push_scheduler.update_rate(outcome)
        elif push_sched_type == 3:
            push_scheduler.update_psi(threshold, outcome)
        successful = np.append(scheduled, np.asarray(successful_push, dtype=int))
        cluster_in_anomaly = pull_scheduler.update_posterior_pmf(successful, drift_state[successful],
                                                             realign_thr)

        ### LOGGING ###
        if debug_mode:
            print('s', anomaly_state)
            print('t', threshold)
            print('c', choices)
            print('out', outcome)
            print('la', anomaly_aoii[k, :])
            print('sch', scheduled)
            print('o', drift_state[successful])
            print('y', drift_state)
            print('dr', drift_risk)
            print('z', drift_detected)
            print('da', drift_aoii[k, :])
            input("Press Enter to continue...")

        # Reset state, anomaly and aoii for cluster where an anomaly was found
        for cluster in cluster_in_anomaly:
            drift_state[pull_scheduler.cluster_map == cluster] = 0
            drift_detected[cluster] = 0
            drift_aoii[k, cluster] = 0

    anomaly_aoii_tot = np.reshape(anomaly_aoii, max_num_frame * num_nodes)
    drift_aoii_tot = np.reshape(drift_aoii, max_num_frame * num_cluster)
    return (np.histogram(anomaly_aoii_tot, bins=num_bins + 1, range=(-0.5, num_bins + 0.5), density=True),
            np.histogram(drift_aoii_tot, bins=num_bins + 1, range=(-0.5, num_bins + 0.5), density=True))


if __name__ == '__main__':
    # Parse arguments, if any
    parallel, savedir, debug, overwrite = cmn.common_parser()
    if savedir is not None:
        coexistence_folder = savedir
    else:
        coexistence_folder = cmn.coexistence_folder

    # Simulation variables
    dec = 6
    P_vec = np.arange(2, 19)
    aoii_thr = 2


    # Order of saving data
    column_titles = ['ThetaAvg', 'Theta99', 'Theta999', 'PsiAvg', 'Psi99', 'Psi999']
    # Schedulers dictionary
    sched_dict = {'MAFMAF': (0, 2), 'MAFCRA': (0, 1),
                  'FSAMAF': (1, 2), 'FSACRA': (1, 1),
                  'AFSAMAF': (2, 2), 'AFSACRA': (2, 1),
                  'MAFPPS': (0, 0), 'FSAPPS': (1, 0), 'AFSAPPS': (2, 0),
                  'PPSMAF': (3, 2), 'PPSCRA': (3,1)} # , 'PPSPPS': (3, 0)}

    # Start cases
    for schedulers in list(sched_dict.keys()):
        push_type, pull_type = sched_dict[schedulers]
        for load in ['hom', 'het']:
            # Check if files exist and load it if there
            prefix = 'coexistence_benchmark_' + schedulers + '_' + load
            filename = os.path.join(coexistence_folder, prefix + '.csv')

            if os.path.exists(filename) and not overwrite:
                aoii = pd.read_csv(filename).iloc[:, 1:].to_numpy()
            else:
                aoii = np.full((len(P_vec), len(column_titles)), np.nan)

            # Get load
            anomaly_rate = 0.03 if load == 'hom' else 0.035
            p_01 = cmn.het_p01 * cmn.het_multipliers[2] if load == 'hom' else cmn.p01_25

            # Start iterations
            for p, P in enumerate(P_vec):
                ### Logging ###
                print(f"Schedulers {schedulers}; load {load}; P={P:02d}. Status:")

                # Check if data is there
                if overwrite or np.all(np.isnan(aoii[p])):
                    args = (push_type, pull_type, cmn.M, cmn.T, cmn.R, cmn.N, cmn.max_age, anomaly_rate, cmn.SIGMA, aoii_thr,
                            cmn.C, cmn.D, p_01, cmn.p11, cmn.dt_realign_thr, 0, P, cmn.ETA, debug)

                    start_time = time.time()
                    if parallel:
                        with ProcessPoolExecutor() as executor:
                            futures = [executor.submit(run_episode, ep, *args) for ep in range(cmn.E)]
                            results = [f.result() for f in futures]
                    else:
                        results = []
                        for ep in range(cmn.E):
                            print(f'\tEpisode: {ep:02d}/{cmn.E - 1:02d}')
                            results.append(run_episode(ep, *args))

                    # Separate and average the results
                    anom_aoii_hist = np.mean(np.array([res[0][0] for res in results]), axis=0)
                    drift_aoii_hist = np.mean(np.array([res[1][0] for res in results]), axis=0)

                    # Anomalies
                    anom_aoii_cdf = np.cumsum(anom_aoii_hist)
                    aoii[p, 0] = np.dot(anom_aoii_hist, np.arange(0, cmn.M + 1, 1))
                    aoii[p, 1] = np.where(anom_aoii_cdf > 0.99)[0][0]
                    aoii[p, 2] = np.where(anom_aoii_cdf > 0.999)[0][0]

                    # DT drifts
                    drift_aoii_cdf = np.cumsum(drift_aoii_hist)
                    aoii[p, 3] = np.dot(drift_aoii_hist, np.arange(0, cmn.M + 1, 1))
                    aoii[p, 4] = np.where(drift_aoii_cdf > 0.99)[0][0]
                    aoii[p, 5] = np.where(drift_aoii_cdf > 0.999)[0][0]

                    # Generate data frame and save it (redundant but to avoid to lose data for any reason)
                    df = pd.DataFrame(aoii.round(dec), columns=column_titles)
                    df.insert(0, 'P', P_vec)
                    df.to_csv(filename, index=False)

                    # Print time
                    elapsed = time.time() - start_time
                    print(f"\t...done in {elapsed:.3f} seconds")

                else:
                    print("\t...already done!")
                    continue
