import numpy as np
import pandas as pd
import os
from distributed_scheduler import generate_distributed_anomalies, DistributedAnomalyScheduler
from common import M, C, D, T, E,  het_p01, het_multipliers, p11, dt_detection_thr, std_bar, dist_schedulers, pull_folder
from dist_rate_test import compute_absorption_rate



def run_episode(sched_type: int,
                num_bins: int, cluster_size: int, num_cluster: int, max_num_frame: int,
                pull_res: int, p_01_vec: np.array or list, p_11: float, detection_thr: float,
                rng: np.random.Generator = np.random.default_rng(),
                debug_mode: bool = False):
    """ Generate a single episode

    :param sched_type: int representing the scheduler type, 0: PPS, 1: CRA, 2: MAF
    :param cluster_size: int representing the cluster size
    :param num_cluster: int representing the number of clusters
    :param max_num_frame: int representing the maximum number of frames under test
    :param pull_res: int representing the number of pull resources (:math:`Q` in the paper)
    :param p_01_vec: vector containing [p_01, p_11]
    :param p_11: float
    :param detection_thr:
    :param rng: random number generator
    :param debug_mode:
    :return:
    """
    # Instantiate scheduler
    assert len(p_01_vec) == cluster_size, "The probability of detecting an anomaly has to be the size of the cluster"

    num_clustered_nodes = num_cluster * cluster_size   # The first clustered nodes have distributed anomalies
    dist_sched = DistributedAnomalyScheduler(num_clustered_nodes, cluster_size, p_01_vec, p_11, rng, debug_mode)

    # Utility variables
    distributed_state = np.zeros(num_clustered_nodes, dtype=int)    # y(k) in the paper
    distributed_aoii = np.zeros((max_num_frame, num_cluster))

    # Start frames
    for k in std_bar(range(max_num_frame)):
        ### ANOMALY GENERATION ###
        distributed_state = generate_distributed_anomalies(p_01_vec, p_11, distributed_state, rng)

        # Compute distributed state z^{(i)}(k)
        distributed_anomaly = np.asarray(np.sum(distributed_state.reshape(num_clustered_nodes // cluster_size, cluster_size), axis=1)
                                >= cluster_size / 2, dtype=int)

        # Compute AoII
        distributed_aoii[k, :] = distributed_aoii[k - 1, :] + distributed_anomaly if k > 0 else distributed_anomaly

        ### UPDATE SCHEDULER PRIORS ###
        dist_sched.update_prior()

        ### PULL-BASED SUBFRAME ###
        # Get pull scheduler
        if sched_type == 0:
            scheduled = dist_sched.schedule(pull_res)
        elif sched_type == 1:
            scheduled = dist_sched.schedule_cra(pull_res)
        else:   # sched_type == 2
            scheduled = dist_sched.schedule_maf(pull_res, k)

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
    rng = np.random.default_rng(0)
    schedulers = dist_schedulers

    debug_mode = False
    overwrite = True

    # Parameters
    Q = 10
    multipliers = np.linspace(het_multipliers[0], het_multipliers[-1], 30)
    absorption_rates = compute_absorption_rate(het_p01, p11, multipliers, show=False)

    # Check if files exist and load it if there
    prefix = 'pull_load'
    filename_avg = os.path.join(pull_folder, prefix + '_avg.csv')
    filename_99 = os.path.join(pull_folder, prefix + '_99.csv')
    filename_999 = os.path.join(pull_folder, prefix + '_999.csv')
    filename_cdf = os.path.join(pull_folder, prefix + '_cdf.csv')

    if os.path.exists(filename_avg) and not overwrite:
        prob_avg = pd.read_csv(filename_avg).iloc[:, 1:].to_numpy()
    else:
        prob_avg = np.full((len(schedulers), len(multipliers)), np.nan)
    if os.path.exists(filename_99) and not overwrite:
        prob_99 = pd.read_csv(filename_99).iloc[:, 1:].to_numpy()
    else:
        prob_99 = np.full((len(schedulers), len(multipliers)), np.nan)
    if os.path.exists(filename_999) and not overwrite:
        prob_999 = pd.read_csv(filename_999).iloc[:, 1:].to_numpy()
    else:
        prob_999 = np.full((len(schedulers), len(multipliers)), np.nan)
    if os.path.exists(filename_cdf) and not overwrite:
        cdf = pd.read_csv(filename_cdf).iloc[:, 1:].to_numpy()
    else:
        cdf = np.full((len(multipliers), M + 1), np.nan)

    for s, _ in enumerate(schedulers):
        for m, mult in enumerate(multipliers):
            p_01 = het_p01 * mult
            ### Logging ###
            print(f"load={absorption_rates[m]:1.3f}, sched={schedulers[s]}. Status:")

            # Check if data is there
            if overwrite or np.isnan(prob_avg[m, s]):
                dist_aoii_hist = np.zeros(M + 1)
                for ep in range(E):
                    print(f'\tEpisode: {ep:02d}/{E-1:02d}')
                    dist_aoii_hist += run_episode(s, M, C, D, T, Q,
                                                  p_01, p11, dt_detection_thr,
                                                  rng,
                                                  debug_mode)[0] / E
                dist_aoii_cdf = np.cumsum(dist_aoii_hist)
                prob_99[s, m] = np.where(dist_aoii_cdf > 0.99)[0][0]
                prob_999[s, m] = np.where(dist_aoii_cdf > 0.999)[0][0]
                prob_avg[s, m] = np.dot(dist_aoii_hist, np.arange(0, M + 1, 1))
                # Save CDF for the PPS-0.5 only
                if s == 0:
                    cdf[m] = dist_aoii_cdf

                # Generate data frame and save it (redundant but to avoid to lose data for any reason)
                for res, file in [(prob_avg, filename_avg), (prob_99, filename_99), (prob_999, filename_999)]:
                    df = pd.DataFrame(res.T.round(dec), columns=schedulers)
                    df.insert(0, 'abs_rate', absorption_rates)
                    df.to_csv(file, index=False)

                if s == 0:
                    cdf_df = pd.DataFrame(cdf.T, columns=absorption_rates.round(dec))
                    cdf_df.insert(0, 'Psi', np.arange(M + 1))
                    cdf_df.to_csv(filename_cdf, index=False)

            else:
                print("\t...already done!")
                continue
