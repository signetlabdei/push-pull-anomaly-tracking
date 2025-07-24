import numpy as np
import pandas as pd
import os
from coexistence_frame_analysis import run_episode
from common import N, R, C, D, T, E, M, RNG, max_age, SIGMA, ETA, het_p01, het_multipliers, p01_25, p11, dt_detection_thr, std_bar, coexistence_folder

if __name__ == '__main__':
    # Simulation variables
    dec = 6
    debug = False
    overwrite = True

    # Parameters
    # aoii_thr = 3
    min_P = 5
    managers = np.array([1, 2])
    manager_names = ['RSM', 'SSM']

    # Order of saving data
    column_titles = ['ThetaAvg', 'Theta99', 'Theta999', 'PsiAvg', 'Psi99', 'Psi999']

    # Start cases
    for load in ['hom', 'het']:

        for aoii_thr in [2, 3]:
            # Check if files exist and load it if there
            prefix = 'coexistence_manager_' + load + f'_aoii{aoii_thr:1d}'
            filename = os.path.join(coexistence_folder, prefix + '.csv')

            if os.path.exists(filename) and not overwrite:
                aoii = pd.read_csv(filename).iloc[:, 1:].to_numpy()
            else:
                aoii = np.full((2, 6), np.nan)

            # Get load
            anomaly_rate = 0.03 if load == 'hom' else 0.035
            p_01 = het_p01 * het_multipliers[2] if load == 'hom' else p01_25

            # Test aoii threshold


            # Start iterations
            for m, manager in enumerate(managers):
                ### Logging ###
                print(f"Load {load}; AoII Th: {aoii_thr:1d}; manager={manager:02d}. Status:")

                # Check if data is there
                if overwrite or np.all(np.isnan(aoii[m])):
                    loca_aoii_hist = np.zeros(M + 1)
                    dist_aoii_hist = np.zeros(M + 1)
                    for ep in range(E):
                        print(f'\tEpisode: {ep:02d}/{E - 1:02d}')
                        loc_tmp, dist_tmp = run_episode(M, T, R,
                                                        N, max_age, anomaly_rate, SIGMA, aoii_thr,
                                                        C, D, p_01, p11, dt_detection_thr,
                                                        manager, min_P, ETA,
                                                        RNG, debug)
                        loca_aoii_hist += loc_tmp[0] / E
                        dist_aoii_hist += dist_tmp[0] / E

                    # Local
                    loca_aoii_cdf = np.cumsum(loca_aoii_hist)
                    aoii[m, 0] = np.dot(loca_aoii_hist, np.arange(0, M + 1, 1))
                    aoii[m, 1] = np.where(loca_aoii_cdf > 0.99)[0][0]
                    aoii[m, 2] = np.where(loca_aoii_cdf > 0.999)[0][0]

                    # Dist
                    dist_aoii_cdf = np.cumsum(dist_aoii_hist)
                    aoii[m, 3] = np.dot(dist_aoii_hist, np.arange(0, M + 1, 1))
                    aoii[m, 4] = np.where(dist_aoii_cdf > 0.99)[0][0]
                    aoii[m, 5] = np.where(dist_aoii_cdf > 0.999)[0][0]

                    # Generate data frame and save it (redundant but to avoid to lose data for any reason)
                    df = pd.DataFrame(aoii.round(dec), columns=column_titles)
                    df.insert(0, 'manager', manager_names)
                    df.to_csv(filename, index=False)

                else:
                    print("\t...already done!")
                    continue
