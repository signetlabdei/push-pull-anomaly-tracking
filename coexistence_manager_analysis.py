import os
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

from coexistence_frame_analysis import run_episode
import common as cmn

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
                aoii = np.full((2, len(column_titles)), np.nan)

            # Get load
            anomaly_rate = 0.03 if load == 'hom' else 0.035
            p_01 = cmn.het_p01 * cmn.het_multipliers[2] if load == 'hom' else cmn.p01_25

            # Start iterations
            for m, manager in enumerate(managers):
                ### Logging ###
                print(f"Load {load}; AoII Th: {aoii_thr:1d}; manager={manager:02d}. Status:")

                # Check if data is there
                if overwrite or np.all(np.isnan(aoii[m])):
                    args = (cmn.M, cmn.T, cmn.R, cmn.N, cmn.max_age, anomaly_rate, cmn.SIGMA, aoii_thr,
                            cmn.C, cmn.D, p_01, cmn.p11, cmn.dt_realign_thr, manager, min_P, cmn.ETA, debug)

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
                    aoii[m, 0] = np.dot(anom_aoii_hist, np.arange(0, cmn.M + 1, 1))
                    aoii[m, 1] = np.where(anom_aoii_cdf > 0.99)[0][0]
                    aoii[m, 2] = np.where(anom_aoii_cdf > 0.999)[0][0]

                    # DT drifts
                    drift_aoii_cdf = np.cumsum(drift_aoii_hist)
                    aoii[m, 3] = np.dot(drift_aoii_hist, np.arange(0, cmn.M + 1, 1))
                    aoii[m, 4] = np.where(drift_aoii_cdf > 0.99)[0][0]
                    aoii[m, 5] = np.where(drift_aoii_cdf > 0.999)[0][0]

                    # Generate data frame and save it (redundant but to avoid to lose data for any reason)
                    df = pd.DataFrame(aoii.round(dec), columns=column_titles)
                    df.insert(0, 'manager', manager_names)
                    df.to_csv(filename, index=False)

                else:
                    print("\t...already done!")
                    continue
