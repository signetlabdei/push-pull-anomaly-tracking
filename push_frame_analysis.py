 #
 # This file is part of the Push-Pull Medium Access repository:
 # https://github.com/signetlabdei/push-pull-anomaly-tracking
 # Copyright (c) 2025:
 # Fabio Saggese (fabio.saggese@ing.unipd.it)
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

import os
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

from push_scheduler import PushScheduler, PushMAFScheduler, PushAlohaScheduler, generate_anomalies
import common as cmn


def run_episode(episode_idx: int,
                push_type: int,
                num_bins: int, push_res:int, max_num_frame:int,
                num_nodes: int, max_age: int, anomaly_rate: float, collision_thr: float, pps_mode: int,
                debug_mode: bool = False):
    r"""Run a single episode of a push-only scenario. Parallelization allowed.

    :param episode_idx: The index of the episode to run.
    :param push_type: The type of scheduler to run 0: MAF, 1: FSA, 2: AFSA, 3: PPS.
    :param num_bins: The number of bins to use for the output histogram.
    :param push_res: The amount of resources available for pull :math:`P`
    :param max_num_frame: The maximum number of frames to simulate.
    :param num_nodes: The number of nodes detecting anomalies :math:`N_a`.
    :param max_age: The maximum age that can be saved for anomalies.
    :param anomaly_rate: The anomaly rate :math:`\rho_a`.
    :param collision_thr: The collision threshold :math:`\sigma`.
    :param pps_mode: The kind of pps scheduler (deprecated, mode 1 is the one considered in the paper)
    :param debug_mode: If true, run in debug mode.
    :return: histogram of anomaly AoII.
    """

    rng = np.random.default_rng(episode_idx)

    # Instantiate scheduler
    if push_type == 0:
        push_scheduler = PushMAFScheduler(num_nodes)
    elif push_type == 1:
        # Maintain load close to 1
        tx_rate = 0.9 / (num_nodes * anomaly_rate / push_res)
    elif push_type == 2:
        push_scheduler = PushAlohaScheduler(num_nodes, anomaly_rate, push_res)
    else: # if push_type == 3:
        push_scheduler = PushScheduler(num_nodes, max_age, anomaly_rate, pps_mode, debug_mode)

    # Useful variables
    anomaly_state = np.zeros(num_nodes)
    aoii = np.zeros((max_num_frame, num_nodes))

    for k in cmn.std_bar(range(max_num_frame)):
        ### ANOMALY GENERATION ###
        anomaly_state = generate_anomalies(anomaly_rate, anomaly_state, rng)

        ### COMPUTE AOII ###
        aoii[k, :] = aoii[k - 1, :] + anomaly_state if k > 0 else anomaly_state

        ### UPDATE SCHEDULER PRIORS ###
        if push_type == 3:
            push_scheduler.update_prior()

        ### PUSH-BASED SUBFRAME ###
        if push_type == 0:
            outcome = push_scheduler.schedule(push_res, [])
            anomaly_state[outcome] = 0
            aoii[k, outcome] = 0
        else:
            if push_type == 1:
                choices = rng.integers(1, push_res + 1, num_nodes) * np.asarray(aoii[k, :] > 0) * (rng.random(num_nodes) < tx_rate)
            elif push_type == 2:
                choices = rng.integers(1, push_res + 1, num_nodes) * np.asarray(aoii[k, :] > 0) * (rng.random(num_nodes) < push_scheduler.rate)
            else:    # push_type == 3:
                # Get anomaly threshold
                threshold = push_scheduler.schedule(push_res, collision_thr, [])
                # Select random slots for active nodes
                choices = rng.integers(1, push_res + 1, num_nodes) * np.asarray(aoii[k, :] > threshold)

            outcome = np.zeros(push_res, dtype=int)
            for p in range(1, push_res + 1):
                chosen = np.where(choices == p)[0]
                if chosen.size != 0:
                    if chosen.size == 1:
                        outcome[p - 1] = chosen[0] + 1
                        anomaly_state[chosen[0]] = 0
                        aoii[k, chosen[0]] = 0
                    else:
                        outcome[p - 1] = -1

        ### POST-FRAME UPDATE ###
        # Local and distributed anomaly belief update
        if push_type == 2:
            push_scheduler.update_rate(outcome)
        if push_type == 3:
            push_scheduler.update_psi(threshold, outcome)

        ### DEBUG for visualization ###
        if debug_mode:
            print('o', outcome)
            print('x', anomaly_state)
            input("Press Enter to continue...")

    # Plotting local anomaly AoII
    aoii_tot = np.reshape(aoii, max_num_frame * num_nodes)
    return np.histogram(aoii_tot, bins=num_bins + 1, range=(-0.5, num_bins + 0.5), density=True)

if __name__ == "__main__":
    # Parse arguments, if any
    parallel, savedir, debug, overwrite = cmn.common_parser()
    if savedir is not None:
        push_folder = savedir
    else:
        push_folder = cmn.push_folder
    # Simulation variables
    dec = 6
    schedulers = cmn.push_scheduler_names
    pps_scheduler_mode = 1
    frame_sizes = np.arange(5, 21, 1)
    rate = 0.03


    # Check if files exist and load it if there
    prefix = 'push_frame'
    filename_avg = os.path.join(push_folder, prefix + '_avg.csv')
    filename_99 = os.path.join(push_folder, prefix + '_99.csv')
    filename_999 = os.path.join(push_folder, prefix + '_999.csv')

    if os.path.exists(filename_avg) and not overwrite:
        prob_avg = pd.read_csv(filename_avg).iloc[:, 1:].to_numpy()
    else:
        prob_avg = np.full((len(frame_sizes), len(schedulers)), np.nan)
    if os.path.exists(filename_99) and not overwrite:
        prob_99 = pd.read_csv(filename_99).iloc[:, 1:].to_numpy()
    else:
        prob_99 = np.full((len(frame_sizes), len(schedulers)), np.nan)
    if os.path.exists(filename_999) and not overwrite:
        prob_999 = pd.read_csv(filename_999).iloc[:, 1:].to_numpy()
    else:
        prob_999 = np.full((len(frame_sizes), len(schedulers)), np.nan)

    for s, scheduler in enumerate(schedulers):
        for p, P in enumerate(frame_sizes):
            # Logging #
            print(f"Scheduler: {scheduler}; P={P:02d}. Status:")

            # Check if data is there
            if overwrite or np.isnan(prob_avg[p, s]):
                args = (s, cmn.M, P, cmn.T, cmn.N, cmn.max_age,
                        rate, cmn.SIGMA, pps_scheduler_mode, debug)

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

                # Average the results
                anomaly_aoii_hist = np.mean(np.array([res[0] for res in results]), axis=0)

                # Divide data
                anomaly_aoii_cdf = np.cumsum(anomaly_aoii_hist)
                prob_99[p, s] = np.where(anomaly_aoii_cdf > 0.99)[0][0]
                prob_999[p, s] = np.where(anomaly_aoii_cdf > 0.999)[0][0]
                prob_avg[p, s] = np.dot(anomaly_aoii_hist, np.arange(0, cmn.M + 1, 1))

                # Generate data frame and save it (redundant but to avoid to lose data for any reason)
                for res, file in [(prob_avg, filename_avg), (prob_99, filename_99), (prob_999, filename_999)]:
                    df = pd.DataFrame(res.round(dec), columns=schedulers)
                    df.insert(0, 'P', frame_sizes)
                    df.to_csv(file, index=False)

                # Print time
                elapsed = time.time() - start_time
                print(f"\t...done in {elapsed:.3f} seconds")

            else:
                print("\t...already done!")
                continue
