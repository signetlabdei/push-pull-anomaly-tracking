import numpy as np
import pandas as pd
import os
from common import M, C, D, T, E, Q_vec, het_p01, het_multipliers, hom_p01, hom_multipliers, p11, dt_detection_thr, pull_folder, dist_schedulers
from pull_load_analysis import run_episode


if __name__ == '__main__':
    # Simulation variables
    dec = 6
    rng = np.random.default_rng(0)
    schedulers = dist_schedulers
    debug_mode = False
    overwrite = False


    for clu in ['het', 'hom']:
        # Check if files exist and load it if there
        prefix = 'pull_frame_' + clu
        filename_avg = os.path.join(pull_folder, prefix + '_avg.csv')
        filename_99 = os.path.join(pull_folder, prefix + '_99.csv')
        filename_999 = os.path.join(pull_folder, prefix + '_999.csv')

        if os.path.exists(filename_avg) and not overwrite:
            prob_avg = pd.read_csv(filename_avg).iloc[:, 1:].to_numpy()
        else:
            prob_avg = np.full((len(Q_vec), len(schedulers)), np.nan)
        if os.path.exists(filename_99) and not overwrite:
            prob_99 = pd.read_csv(filename_99).iloc[:, 1:].to_numpy()
        else:
            prob_99 = np.full((len(Q_vec), len(schedulers)), np.nan)
        if os.path.exists(filename_999) and not overwrite:
            prob_999 = pd.read_csv(filename_999).iloc[:, 1:].to_numpy()
        else:
            prob_999 = np.full((len(Q_vec), len(schedulers)), np.nan)

        # Get load
        p_01 = het_p01 * het_multipliers[2] if clu == 'het' else hom_p01 * hom_multipliers[2]

        for s, _ in enumerate(schedulers):
            for q, Q in enumerate(Q_vec):
                ### Logging ###
                print(f"Test {clu}; sched={schedulers[s]}, Q={Q:02d}. Status:")

                # Check if data is there
                if overwrite or np.isnan(prob_avg[q, s]):
                    dist_aoii_hist = np.zeros(M + 1)
                    for ep in range(E):
                        print(f'\tEpisode: {ep:02d}/{E-1:02d}')
                        dist_aoii_hist += run_episode(s, M, C, D, T, Q,
                                                      p_01, p11, dt_detection_thr,
                                                      rng,
                                                      debug_mode)[0] / E
                    dist_aoii_cdf = np.cumsum(dist_aoii_hist)
                    prob_99[q, s] = np.where(dist_aoii_cdf > 0.99)[0][0]
                    prob_999[q, s] = np.where(dist_aoii_cdf > 0.999)[0][0]
                    prob_avg[q, s] = np.dot(dist_aoii_hist, np.arange(0, M + 1, 1))

                    # Generate data frame and save it (redundant but to avoid to lose data for any reason)
                    for res, file in [(prob_avg, filename_avg), (prob_99, filename_99), (prob_999, filename_999)]:
                        df = pd.DataFrame(res.round(dec), columns=schedulers)
                        df.insert(0, 'Q', Q_vec)
                        df.to_csv(file, index=False)
                else:
                    print("\t...already done!")
                    continue
