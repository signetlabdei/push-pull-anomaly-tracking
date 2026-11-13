# This file is part of the Push-Pull Medium Access repository:
 # https://github.com/signetlabdei/push-pull-anomaly-tracking
 # Copyright (c) 2026:
 # Fabio Saggese (fabio.saggese@ing.unipi.it)
 # Federico Chiariotti (federico.chiariotti@unipd.it)
 #
 # This program is free software: you can redistribute it and/or modify
 # it under the terms of the GNU General Public License as published by
 # the Free Software Foundation, version 3.
 #
 # This program is distributed in the hope that it will be useful, but
 # WITHOUT ANY WARRANTY; without even the implied warranty of
 # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 # General Public License for more details.
 #
 # You should have received a copy of the GNU General Public License
 # along with this program. If not, see <http://www.gnu.org/licenses/>.
 #

import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
import numpy as np
import pandas as pd

from push_scheduler import generate_anomalies, PushScheduler
from pull_kalman_scheduler import generate_drifts, generate_observations, PullScheduler
from push_pull_manager import ResourceManager
import common as cmn

from phy_mac_isac.anomaly_tracking.scenario import generate_environment


def generate_rectangle_coordinate(size: tuple):
    """Generate scenario with a single central AP in the network where the area_coordinates takes a generic rectangle"""
    a = size[0]
    b = size[1]
    return np.array([[-a/2, -b/2], [a/2, -b/2], [a/2, b/2], [-a/2, b/2]])


def run_episode(episode_idx: int,
                aoii_bins: int, drift_bins: int, mse_max: int, max_num_frame: int, resources: int,
                num_nodes: int, max_age: int, anomaly_rate: float, collision_thr: float, aoii_thr: int,
                mse_thr: float, cluster_size: int, num_cluster: int, F: np.ndarray, F_hat: np.ndarray,
                H : np.ndarray, sigma_w: float, sigma_v: float, sigma_w_hat: float, sigma_v_hat: float,
                manager_type: int, push_resources: int = 2, hysteresis: float = 0.005, bler_target: float = 0.01,
                debug_mode: bool = False):
    r"""Run a single episode of a push-pull scenario. Parallelization allowed.

    :param episode_idx: The index of the episode to run.
    :param aoii_bins: The number of bins to use for the AoII output histogram.
    :param drift_bins: The number of bins to use for the MSE output histogram.
    :param mse_max: The maximum MSE value to use for the output histogram.
    :param max_num_frame: The maximum number of frames to simulate.
    :param resources: The amount of resources available :math:`R`.
    :param num_nodes: The number of nodes :math:`N`.
    :param max_age: The maximum age that can be saved for anomalies.
    :param anomaly_rate: The anomaly rate :math:`\rho_a`.
    :param collision_thr: The collision threshold :math:`\sigma`.
    :param aoii_thr: The AoII risk threshold :math:`\hat{\theta}`.
    :param mse_thr: The MSE risk threshold.
    :param cluster_size: The size of the clusters :math:`C`.
    :param num_cluster: The number of clusters :math:`D`.
    :param F: state update matrix :math:`F`
    :param F_hat: estimated state update matrix :math:`F`
    :param H: observation matrix :math:`H`
    :param sigma_w: process noise variance :math:`sigma_w`
    :param sigma_v: observation noise variance :math:`sigma_v`
    :param sigma_w_hat: estimated process noise variance :math:`sigma_w`
    :param sigma_v_hat: estimated observation noise variance :math:`sigma_v`
    :param manager_type: The type of manager to use (0: fixed resources, 1: RSM, 2:SSM).
    :param push_resources: The amount of resources available for push if manager_type = 0 :math:`P` or the minimum resources :math:`R_{min}` otherwise.
    :param hysteresis: The SSM hysteresis threshold :math:`\eta`. Used only if manager_type = 2.
    :param debug_mode: If true, run in debug mode.
    :return: Tuple containing: (histogram of anomaly AoII, histogram of DT drift AoII)
    """
    # Instantiate scheduler
    rng = np.random.default_rng(episode_idx)

    num_clustered_nodes = num_cluster * cluster_size  # The first clustered nodes have distributed anomalies

    # Instantiate schedulers
    push_scheduler = PushScheduler(num_nodes, max_age, anomaly_rate, 1, debug_mode)
    pull_scheduler = PullScheduler(num_clustered_nodes, cluster_size, F_hat, H, sigma_w_hat, sigma_v_hat, rng = rng, debug_mode = debug_mode)
    manager = ResourceManager(manager_type, resources)
    if manager_type == 0:   # Set P beforehand
        manager.set_push_resources(push_resources)
    elif manager_type in [1, 2]:
        manager.set_min_threshold(push_resources)
        if manager_type == 2:
            manager.set_hysteresis(hysteresis)

    # Utility variables
    anomaly_state = np.zeros(num_nodes)
    drift_state = np.zeros(num_clustered_nodes, dtype=int)    # y(k) in the paper
    anomaly_aoii = np.zeros((max_num_frame, num_nodes))
    drift_mse = np.zeros((max_num_frame, num_cluster))

    # Generate scenario with a single central AP in the network where the area_coordinates takes a generic rectangle
    env_kwargs = {"num_ue": num_nodes, "num_slots": resources, "num_frames": max_num_frame,
                  "num_clusters": num_cluster, "cluster_size": cluster_size, "bler_target": bler_target}
    env = generate_environment(episode=episode_idx, env_kwargs=env_kwargs, device="cpu")


    for k in cmn.std_bar(range(max_num_frame)):
        ### ANOMALY GENERATION ###
        # Local
        anomaly_state = generate_anomalies(anomaly_rate, anomaly_state, rng)
        # Distributed
        drift_state = generate_drifts(drift_state, cluster_size, F, sigma_w, rng)

        ### COMPUTE AOII ###
        anomaly_aoii[k, :] = anomaly_aoii[k - 1, :] + anomaly_state if k > 0 else anomaly_state

        ### UPDATE SCHEDULER PRIORS ###
        push_scheduler.update_prior()
        pull_scheduler.update_prior()

        ### SUBFRAME ALLOCATION ###
        anomaly_risk = push_scheduler.get_risk(aoii_thr)
        drift_risk = np.min([1, np.mean(pull_scheduler.get_cluster_mse) / mse_thr])
        P, Q = manager.allocate_resources(anomaly_risk, drift_risk)  # Allocate resources
        if debug_mode:
            print('anomaly_risk', anomaly_risk, 'drift_risk', drift_risk, 'ratio', anomaly_risk / drift_risk)
            print('P', P, 'Q', Q)

        ### PULL-BASED SUBFRAME ###
        # Get pull schedule
        scheduled = pull_scheduler.schedule_pps(Q)


        ### PUSH-BASED SUBFRAME ###
        # Get local anomaly threshold
        threshold = push_scheduler.schedule(P, collision_thr, scheduled)
        # Make active the nodes with aoii larger than the threshold and not scheduled during pull
        bool_idx = (anomaly_aoii[k, :] > threshold) * ~np.array([n in scheduled for n in np.arange(num_nodes)])
        # Select random slots for active nodes
        choices = rng.integers(1, P + 1, num_nodes) * bool_idx
        # Prepare outcome vector
        outcome = np.zeros(P, dtype=int)
        successful_push = []
        # Keep track of singleton slots
        for p in range(1, P + 1):
            chosen = np.where(choices == p)[0]
            if chosen.size != 0:
                if chosen.size == 1:
                    if chosen[0] < num_clustered_nodes:
                        successful_push.append(chosen[0])
                    outcome[p - 1] = chosen[0] + 1
                    # anomaly_state[chosen[0]] = 0
                    # anomaly_aoii[k, chosen[0]] = 0
                else:
                    outcome[p - 1] = -1

        # Take the full scheduling vector to feed env.step with node indexing (<0 are silent or collided slots)
        full_scheduling = np.hstack([scheduled, outcome - 1])
        # Run Sionna SYS and output the decoded slots (bool) and successful nodes index
        decoded, successful, bler = env.step(full_scheduling)

        ### POST-FRAME UPDATE ###
        # Update local anomalies in decoded slots
        anomaly_aoii[k, successful] = 0
        anomaly_state[successful] = 0

        # Re-process push outcome to be sure to account for erasure
        silences = outcome == 0
        collisions = outcome == -1
        undecoded = ~decoded[Q:]
        # The erased ones are the one undecoded but that was not silent nor a collision
        erased = undecoded * ~silences * ~collisions

        push_outcome_after_decoding = outcome
        push_outcome_after_decoding[erased] = -1

        # Local and distributed anomaly belief update
        push_scheduler.update_psi(threshold, push_outcome_after_decoding, erasure=bler_target)
        # successful = np.append(scheduled, np.asarray(successful_push, dtype=int))
        observations = generate_observations(drift_state, cluster_size, H, sigma_v, rng)
        successful_clustered = successful[successful < num_clustered_nodes]
        pull_scheduler.update_posterior_pmf(successful_clustered, observations[successful_clustered])

        # Add an offset to the state equal to the state estimate to avoid state divergence
        drift_state -= pull_scheduler.reset_state_estimate()

        drift_mse[k, :] = pull_scheduler.get_actual_mse(drift_state)

        ### LOGGING ###
        if debug_mode:
            print('s', anomaly_state)
            print('t', threshold)
            print('c', choices)
            print('out', outcome)
            print('la', anomaly_aoii[k, :])
            print('sch', scheduled)
            print('o', drift_state[successful_clustered])
            print('y', drift_state)
            print('dr', drift_risk)
            print('da', drift_mse[k, :])
            input("Press Enter to continue...")


    anomaly_aoii_tot = np.reshape(anomaly_aoii, max_num_frame * num_nodes)
    drift_mse_tot = np.reshape(drift_mse, max_num_frame * num_cluster)
    return (np.histogram(anomaly_aoii_tot, bins=aoii_bins+1, range=(-0.5, aoii_bins + 0.5), density=True),
            np.histogram(drift_mse_tot, bins=drift_bins, range=(0, mse_max), density=True))

if __name__ == '__main__':
    # Parse arguments, if any
    parallel, savedir, debug, overwrite = cmn.common_parser()
    if savedir is not None:
        coexistence_folder = savedir
    else:
        coexistence_folder = cmn.coexistence_folder

    # Simulation variables
    dec = 6
    min_P = 5
    managers = np.array([1, 2])
    aoii_thr = 2
    mse_thr_vec = [5, 10, 20]
    anomaly_rate = 0.03
    manager_names = ['RSM', 'SSM']

    # Order of saving data
    metrics = ['aoii_avg', 'aoii_99', 'aoii_999', 'mse_avg', 'mse_99', 'mse_999']

    for mse_thr in mse_thr_vec:
        print(f"%------------------%\nThreshold: {mse_thr:02d}.\n%------------------%")

        # Start cases
        # Check if files exist and load it if there
        prefix = f"coexistence_manager_realistic_kalman_thr{mse_thr:02d}"
        data = (metrics, manager_names)
        outcomes, filename = cmn.check_data(data, prefix, coexistence_folder, overwrite_flag=overwrite)


        # Start iterations
        for m, manager in enumerate(managers):
            ### Logging ###
            print(f"Manager: {manager}. Status:")

            # Check if data is there
            if overwrite or np.all(np.isnan(outcomes[:, m])):

                args = (cmn.aoii_hbins, cmn.mse_hbins, cmn.mse_maxval, cmn.T, cmn.R, cmn.N, cmn.max_age, anomaly_rate, cmn.SIGMA, aoii_thr, mse_thr,
                        cmn.C, cmn.D, cmn.F, cmn.F, cmn.H, cmn.sigma_w, cmn.sigma_v, cmn.sigma_w_hat,
                        cmn.sigma_v_hat, manager, min_P, cmn.ETA, cmn.BLER_TARGET, debug)

                start_time = time.time()
                if parallel:
                    with ProcessPoolExecutor(mp_context=mp.get_context("spawn"), max_workers=os.cpu_count() // 2) as executor:
                        futures = [executor.submit(run_episode, ep, *args) for ep in range(cmn.E)]
                        results = [f.result() for f in futures]
                else:
                    results = []
                    for ep in range(cmn.E):
                        print(f'\tEpisode: {ep:02d}/{cmn.E - 1:02d}')
                        results.append(run_episode(ep, *args))

                # Separate and average the results
                anom_aoii_hist = np.mean(np.array([res[0][0] for res in results]), axis=0)
                mse_hist = np.mean(np.array([res[1][0] for res in results]), axis=0)
                mse_values = np.arange(0, cmn.mse_maxval, cmn.mse_maxval / cmn.mse_hbins) + cmn.mse_maxval / cmn.mse_hbins / 2

                # Anomalies
                anom_aoii_cdf = np.cumsum(anom_aoii_hist)
                outcomes[0, m] = np.dot(anom_aoii_hist, np.arange(0, cmn.aoii_hbins + 1, 1))
                outcomes[1, m] = np.where(anom_aoii_cdf > 0.99)[0][0]
                outcomes[2, m] = np.where(anom_aoii_cdf > 0.999)[0][0]

                # DT drifts
                mse_cdf = np.cumsum(mse_hist) / cmn.mse_hbins * cmn.mse_maxval
                outcomes[3, m] = np.dot(mse_values, mse_hist) / np.sum(mse_hist)
                outcomes[4, m] = mse_values[np.where(mse_cdf > 0.99)[0][0]]
                outcomes[5, m] = mse_values[np.where(mse_cdf > 0.999)[0][0]]

                # Generate data frame and save it (redundant but to avoid to lose data for any reason)
                df = pd.DataFrame(outcomes.T.round(dec), columns=metrics)
                df.insert(0, 'manager', manager_names)
                df.to_csv(filename, index=False)

                # Print time
                elapsed = time.time() - start_time
                print(f"\t...done in {elapsed:.3f} seconds")

            else:
                print("\t...already done!")
                continue
