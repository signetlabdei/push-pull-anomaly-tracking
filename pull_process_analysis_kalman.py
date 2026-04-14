import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

from pull_kalman_scheduler import generate_drifts, generate_observations, PullScheduler
import common as cmn


def run_episode(episode_idx: int,
                pull_type: int,
                num_bins: int, max_num_frame: int,
                cluster_size: int, num_cluster: int, pull_res: int,
                F: np.ndarray, H : np.ndarray,
                sigma_w: float, sigma_v: float,
                sigma_w_hat: float, sigma_v_hat: float,
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
    :param sigma_w_hat: estimated process noise variance :math:`sigma_w`
    :param sigma_v_hat: estimated observation noise variance :math:`sigma_v`
    :param debug_mode: If true, run in debug mode.
    :return: histogram of DT drift AoII.
    """
    # Instantiate scheduler
    assert F.shape[0] == cluster_size, "The probability of detecting an anomaly has to be the size of the cluster"

    rng = np.random.default_rng(episode_idx)

    num_clustered_nodes = num_cluster * cluster_size   # The first clustered nodes have distributed anomalies
    pull_scheduler = PullScheduler(num_clustered_nodes, cluster_size, F, H, sigma_w_hat, sigma_v_hat, rng = rng, debug_mode = debug_mode)

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
        mse[k] = pull_scheduler.get_cluster_mse

    return np.histogram(mse, bins=num_bins, range=(0,100), density = True)


if __name__ == '__main__':
    # Parse arguments, if any
    parallel, savedir, debug, overwrite = cmn.common_parser()
    if savedir is not None:
        pull_folder = savedir
    else:
        pull_folder = cmn.pull_folder
    # Simulation variables
    schedulers = cmn.pull_scheduler_names
    dec = 6
    Q = 10
    sigma_w_vec = np.arange(0.05, 1.0, 0.05).round(dec)


    # Check if results data exist and load it if there
    data_shape = (len(schedulers), len(sigma_w_vec))
    prefix = 'pull_process_kalman'
    prob_avg, filename_avg = cmn.check_data(data_shape, prefix + '_avg', pull_folder, overwrite_flag=overwrite)
    prob_99, filename_99 = cmn.check_data(data_shape, prefix + '_99', pull_folder, overwrite_flag=overwrite)
    prob_999, filename_999 = cmn.check_data(data_shape, prefix + '_999', pull_folder, overwrite_flag=overwrite)

    for s, _ in enumerate(schedulers):
        for m, sigma_w in enumerate(sigma_w_vec):
            print(f"Test: sched={schedulers[s]}, sigma_w={sigma_w:.2f}. Status:")

            # Check if data is there
            if overwrite or np.isnan(prob_avg[s, m]):
                args = (s, cmn.bins, cmn.T, cmn.C, cmn.D, Q,
                        cmn.F, cmn.H, sigma_w, cmn.sigma_v,
                        sigma_w, cmn.sigma_v_hat, debug)

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
                mse_hist = np.mean(np.array([res[0] for res in results]), axis=0)
                mse_values = (results[0][1][:-1] + results[0][1][1:]) / 2    # Taking the center of each bin
                # Divide data
                mse_cdf = np.cumsum(mse_hist) / cmn.bins * 100
                prob_avg[s, m] = np.dot(mse_values, mse_hist) / np.sum(mse_hist)
                prob_99[s, m] = mse_values[np.where(mse_cdf > 0.99)[0][0]]
                prob_999[s, m] = mse_values[np.where(mse_cdf > 0.999)[0][0]]

                # Generate data frame and save it (redundant but to avoid to lose data for any reason)
                for res, file in [(prob_avg, filename_avg), (prob_99, filename_99), (prob_999, filename_999)]:
                    df = pd.DataFrame(res.T.round(dec), columns=schedulers)
                    df.insert(0, 'sigma_w', sigma_w_vec)
                    df.to_csv(file, index=False)

                # Print time
                elapsed = time.time() - start_time
                print(f"\t...done in {elapsed:.3f} seconds")

            else:
                print("\t...already done!")
                continue
