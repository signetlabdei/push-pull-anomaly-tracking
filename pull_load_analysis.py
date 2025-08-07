import os
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

from pull_scheduler import generate_drifts, PullScheduler
import common as cmn
from drift_rate_test import compute_absorption_rate



def run_episode(episode_idx: int,
                pull_type: int,
                num_bins: int, max_num_frame: int,
                cluster_size: int, num_cluster: int, pull_res: int,
                p_01_vec: np.ndarray, p_11: float, realign_thr: float,
                debug_mode: bool = False):
    r"""Run a single episode of a push-pull scenario. Parallelization allowed.

    :param episode_idx: The index of the episode to run.
    :param pull_type: The type of scheduler to run 0: PPS, 1: CRA, 2: MAF
    :param num_bins: The number of bins to use for the output histogram.
    :param max_num_frame: The maximum number of frames to simulate.
    :param cluster_size: The size of the clusters :math:`C`.
    :param num_cluster: The number of clusters :math:`D`.
    :param pull_res: The amount of resources available for pull :math:`Q`
    :param p_01_vec: The probability of detecting a drift (0 to 1) for each node in a cluster.
    :param p_11: The probability of remaining in the drift state (equal for each node in the cluster).
    :param realign_thr: The DT re-alignment threshold :math:`\nu_{reset}`.
    :param debug_mode: If true, run in debug mode.
    :return: histogram of DT drift AoII.
    """
    # Instantiate scheduler
    assert len(p_01_vec) == cluster_size, "The probability of detecting an anomaly has to be the size of the cluster"

    rng = np.random.default_rng(episode_idx)

    num_clustered_nodes = num_cluster * cluster_size   # The first clustered nodes have distributed anomalies
    pull_scheduler = PullScheduler(num_clustered_nodes, cluster_size, p_01_vec, p_11, rng, debug_mode)

    # Utility variables
    drift_state = np.zeros(num_clustered_nodes, dtype=int)    # y(k) in the paper
    drift_aoii = np.zeros((max_num_frame, num_cluster))

    # Start frames
    for k in cmn.std_bar(range(max_num_frame)):
        ### ANOMALY GENERATION ###
        drift_state = generate_drifts(p_01_vec, p_11, drift_state, rng)

        # Compute distributed state z^{(i)}(k)
        drift_detected = np.asarray(np.sum(drift_state.reshape(num_clustered_nodes // cluster_size, cluster_size), axis=1)
                                >= cluster_size / 2, dtype=int)

        # Compute AoII
        drift_aoii[k, :] = drift_aoii[k - 1, :] + drift_detected if k > 0 else drift_detected

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
        cluster_in_anomaly = pull_scheduler.update_posterior_pmf(successful,
                                                             drift_state[successful],
                                                             realign_thr)

        ### DEBUG for visualization ###
        if debug_mode:
            risk = pull_scheduler.get_cluster_risk
            print('s', scheduled)
            print('o', drift_state[successful])
            print('y', drift_state)
            print('r', risk)
            print('z', drift_detected)
            print('a', drift_aoii[k, :])
            input("Press Enter to continue...")

        # Reset state, anomaly and aoii for cluster where an anomaly was found
        for cluster in cluster_in_anomaly:
            drift_state[pull_scheduler.cluster_map == cluster] = 0
            drift_detected[cluster] = 0
            drift_aoii[k, cluster] = 0

    drift_aoii_tot = np.reshape(drift_aoii, max_num_frame * num_cluster)
    return np.histogram(drift_aoii_tot, bins=num_bins + 1, range=(-0.5, num_bins + 0.5), density=True)


if __name__ == '__main__':
    # Parse arguments, if any
    parallel, savedir, debug, overwrite = cmn.common_parser()
    if savedir is not None:
        pull_folder = savedir
    else:
        pull_folder = cmn.pull_folder
    # Simulation variables
    dec = 6
    schedulers = cmn.pull_scheduler_names
    Q = 10
    multipliers = np.linspace(cmn.het_multipliers[0], cmn.het_multipliers[-1], 30)
    absorption_rates = compute_absorption_rate(cmn.het_p01, cmn.p11, multipliers, show=False)

    # Check if files exist and load it if there
    prefix = 'pull_load'
    filename_avg = os.path.join(pull_folder, prefix + '_avg.csv')
    filename_99 = os.path.join(pull_folder, prefix + '_99.csv')
    filename_999 = os.path.join(pull_folder, prefix + '_999.csv')
    filename_cdf = os.path.join(pull_folder, prefix + '_cdf.csv')

    if os.path.exists(filename_avg) and not overwrite:
        prob_avg = pd.read_csv(filename_avg).iloc[:, 1:].to_numpy().T
    else:
        prob_avg = np.full((len(schedulers), len(multipliers)), np.nan)
    if os.path.exists(filename_99) and not overwrite:
        prob_99 = pd.read_csv(filename_99).iloc[:, 1:].to_numpy().T
    else:
        prob_99 = np.full((len(schedulers), len(multipliers)), np.nan)
    if os.path.exists(filename_999) and not overwrite:
        prob_999 = pd.read_csv(filename_999).iloc[:, 1:].to_numpy().T
    else:
        prob_999 = np.full((len(schedulers), len(multipliers)), np.nan)
    if os.path.exists(filename_cdf) and not overwrite:
        cdf = pd.read_csv(filename_cdf).iloc[:, 1:].to_numpy().T
    else:
        cdf = np.full((len(multipliers), cmn.M + 1), np.nan)

    for s, _ in enumerate(schedulers):
        for m, mult in enumerate(multipliers):
            p_01 = cmn.het_p01 * mult
            ### Logging ###
            print(f"load={absorption_rates[m]:1.3f}, sched={schedulers[s]}. Status:")

            # Check if data is there
            if overwrite or np.isnan(prob_avg[s, m]):
                args = (s, cmn.M, cmn.T, cmn.C, cmn.D, Q,
                        p_01, cmn.p11, cmn.dt_realign_thr, debug)

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
                drift_aoii_hist = np.mean(np.array([res[0] for res in results]), axis=0)

                # Divide data
                drift_aoii_cdf = np.cumsum(drift_aoii_hist)
                prob_99[s, m] = np.where(drift_aoii_cdf > 0.99)[0][0]
                prob_999[s, m] = np.where(drift_aoii_cdf > 0.999)[0][0]
                prob_avg[s, m] = np.dot(drift_aoii_hist, np.arange(0, cmn.M + 1, 1))

                # Save CDF for the PPS only
                if s == 0:
                    cdf[m] = drift_aoii_cdf
                    cdf_df = pd.DataFrame(cdf.T, columns=absorption_rates.round(dec))
                    cdf_df.insert(0, 'Psi', np.arange(cmn.M + 1))
                    cdf_df.to_csv(filename_cdf, index=False)

                # Generate data frame and save it (redundant but to avoid to lose data for any reason)
                for res, file in [(prob_avg, filename_avg), (prob_99, filename_99), (prob_999, filename_999)]:
                    df = pd.DataFrame(res.T.round(dec), columns=schedulers)
                    df.insert(0, 'abs_rate', absorption_rates)
                    df.to_csv(file, index=False)

                # Print time
                elapsed = time.time() - start_time
                print(f"\t...done in {elapsed:.3f} seconds")

            else:
                print("\t...already done!")
                continue
