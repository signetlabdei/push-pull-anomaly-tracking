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

from pull_load_analysis import run_episode
import common as cmn


if __name__ == '__main__':
    # Parse arguments, if any
    parallel, savedir, debug, overwrite = cmn.common_parser()
    if savedir is not None:
        pull_folder = savedir
    else:
        pull_folder = cmn.pull_folder
    # Simulation variables
    dec = 6
    Q_vec = np.arange(5, 21)
    schedulers = cmn.pull_scheduler_names
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
        p_01 = cmn.het_p01 * cmn.het_multipliers[2] if clu == 'het' else cmn.hom_p01 * cmn.hom_multipliers[2]

        for s, _ in enumerate(schedulers):
            for q, Q in enumerate(Q_vec):
                ### Logging ###
                print(f"Test {clu}; sched={schedulers[s]}, Q={Q:02d}. Status:")

                # Check if data is there
                if overwrite or np.isnan(prob_avg[q, s]):
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
                    dist_aoii_cdf = np.cumsum(drift_aoii_hist)
                    prob_99[q, s] = np.where(dist_aoii_cdf > 0.99)[0][0]
                    prob_999[q, s] = np.where(dist_aoii_cdf > 0.999)[0][0]
                    prob_avg[q, s] = np.dot(drift_aoii_hist, np.arange(0, cmn.M + 1, 1))

                    # Generate data frame and save it (redundant but to avoid to lose data for any reason)
                    for res, file in [(prob_avg, filename_avg), (prob_99, filename_99), (prob_999, filename_999)]:
                        df = pd.DataFrame(res.round(dec), columns=schedulers)
                        df.insert(0, 'Q', Q_vec)
                        df.to_csv(file, index=False)

                    # Print time
                    elapsed = time.time() - start_time
                    print(f"\t...done in {elapsed:.3f} seconds")

                else:
                    print("\t...already done!")
                    continue
