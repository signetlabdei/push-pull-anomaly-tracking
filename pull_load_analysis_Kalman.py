import os
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

from pull_kalman_scheduler import generate_drifts, generate_observations, PullScheduler
import common as cmn
from drift_rate_test import compute_absorption_rate



def run_episode(episode_idx: int,
                pull_type: int,
                num_bins: int, max_num_frame: int,
                cluster_size: int, num_cluster: int, pull_res: int,
                F: np.ndarray, H : np.ndarray,
                sigma_w: float, sigma_v: float,
                debug_mode: bool = False):
    r"""Run a single episode of a push-pull scenario. Parallelization allowed.

    :param episode_idx: The index of the episode to run.
    :param pull_type: The type of scheduler to run 0: PPS, 1: CRA, 2: MAF
    :param num_bins: The number of bins to use for the output histogram.
    :param max_num_frame: The maximum number of frames to simulate.
    :param cluster_size: The size of the clusters :math:`C`.
    :param num_cluster: The number of clusters :math:`D`.
    :param pull_res: The amount of resources available for pull :math:`Q`
    :param F: state update matrix :math:`F`
    :param H: observation matrix :math:`H`
    :param sigma_w: process noise variance :math:`sigma_w`
    :param sigma_v: observation noise variance :math:`sigma_v`
    :param debug_mode: If true, run in debug mode.
    :return: histogram of DT drift AoII.
    """
    # Instantiate scheduler
    assert F.shape[0] == cluster_size, "The probability of detecting an anomaly has to be the size of the cluster"

    rng = np.random.default_rng(episode_idx)

    num_clustered_nodes = num_cluster * cluster_size   # The first clustered nodes have distributed anomalies
    pull_scheduler = PullScheduler(num_clustered_nodes, cluster_size, F, H, sigma_w, sigma_v, rng = rng, debug_mode = debug_mode)

    # Utility variables
    drift_state = np.zeros(num_clustered_nodes, dtype=int)    # y(k) in the paper
    mse = np.zeros((max_num_frame, num_cluster))

    # Start frames
    for k in cmn.std_bar(range(max_num_frame)):
        ### ANOMALY GENERATION ###
        drift_state = generate_drifts(drift_state, cluster_size, F, sigma_w, rng)

        ### UPDATE SCHEDULER PRIORS ###
        pull_scheduler.update_prior()

        ### PULL-BASED SUBFRAME ###
        # Get pull scheduler
        if pull_type == 0:
            scheduled = pull_scheduler.schedule_pps(pull_res)
        elif pull_type == 1:
            scheduled = pull_scheduler.schedule_cra(pull_res)
        else:   # sched_type == 2
            scheduled = pull_scheduler.schedule_maf(pull_res, k)

        ### POST-FRAME UPDATE ###
        # Distributed anomaly belief update
        successful = scheduled  # Ignoring push
        observations = generate_observations(drift_state, cluster_size, H, sigma_v, rng)
        cluster_in_anomaly = pull_scheduler.update_posterior_pmf(successful,
                                                             observations[successful])

        ### DEBUG for visualization ###
        if debug_mode:
            risk = pull_scheduler.get_cluster_mse
            print('s', scheduled)
            print('o', drift_state[successful])
            print('y', drift_state)
            print('r', risk)
            print('t', pull_scheduler.get_total_mse)
            input("Press Enter to continue...")

        ### COMPUTE TOTAL RISK
        mse[k] = pull_scheduler.get_total_mse


    return mse


if __name__ == '__main__':
    # Parse arguments, if any
    parallel, savedir, debug, overwrite = cmn.common_parser()
    if savedir is not None:
        pull_folder = savedir
    else:
        pull_folder = cmn.pull_folder
    # Simulation variables
    schedulers = cmn.pull_scheduler_names
    Q = 10

    # Check if files exist and load it if there
    prefix = 'pull_load_kalman'
    filename_avg = os.path.join(pull_folder, prefix + '_avg.csv')
    filename_99 = os.path.join(pull_folder, prefix + '_99.csv')
    filename_999 = os.path.join(pull_folder, prefix + '_999.csv')
    filename_cdf = os.path.join(pull_folder, prefix + '_cdf.csv')

    if os.path.exists(filename_avg) and not overwrite:
        prob_avg = pd.read_csv(filename_avg).iloc[:, 1:].to_numpy().T
    else:
        prob_avg = np.full(len(schedulers), np.nan)
    if os.path.exists(filename_99) and not overwrite:
        prob_99 = pd.read_csv(filename_99).iloc[:, 1:].to_numpy().T
    else:
        prob_99 = np.full(len(schedulers), np.nan)
    if os.path.exists(filename_999) and not overwrite:
        prob_999 = pd.read_csv(filename_999).iloc[:, 1:].to_numpy().T
    else:
        prob_999 = np.full(len(schedulers), np.nan)

    for s, _ in enumerate(schedulers):
        # Check if data is there
        if overwrite or np.isnan(prob_avg[s]):
            args = (s, cmn.bins, cmn.T, cmn.C, cmn.D, Q,
                    cmn.F, cmn.H, cmn.sigma_w, cmn.sigma_v, debug)

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
            mse_hist = np.histogram(np.asarray(results).flatten(), bins = cmn.bins, range=(0,100), density = True)
            # Divide data
            mse_cdf = np.cumsum(mse_hist[0]) / cmn.bins * 100
            prob_99[s] = mse_hist[1][np.where(mse_cdf > 0.99)[0][0]]
            prob_999[s] = mse_hist[1][np.where(mse_cdf > 0.999)[0][0]]
            prob_avg[s] = np.mean(np.asarray(results).flatten())

            # Generate data frame and save it (redundant but to avoid to lose data for any reason)
            for res, file in [(prob_avg, filename_avg), (prob_99, filename_99), (prob_999, filename_999)]:
                df = pd.DataFrame([res], columns=schedulers)
                print(df)
                df.to_csv(file, index=False)

            # Print time
            elapsed = time.time() - start_time
            print(f"\t...done in {elapsed:.3f} seconds")

        else:
            print("\t...already done!")
            continue
