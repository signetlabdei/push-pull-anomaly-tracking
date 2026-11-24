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

import os
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
import pandas as pd

import common as cmn
from coexistence_realistic_kalman import run_episode


if __name__ == '__main__':
    # Parse arguments, if any
    parallel, savedir, debug, overwrite = cmn.common_parser()
    if savedir is not None:
        coexistence_folder = savedir
    else:
        coexistence_folder = cmn.phy_folder

    # Simulation variables
    dec = 6
    P_vec = np.arange(2, 19)
    aoii_thr = 2    # Unused when testing coexistence as a search grid
    mse_thr = 10    # Unused when testing coexistence as a search grid
    anomaly_rate = 0.03
    manager = 0

    # Order of saving data
    metrics = ['aoii_avg', 'aoii_99', 'aoii_999', 'mse_avg', 'mse_99', 'mse_999']

    # Start cases
    # Check if files exist and load it if there
    prefix = f"coexistence_frame_kalman"
    data = (metrics, P_vec)
    outcomes, filename = cmn.check_data(data, prefix, coexistence_folder, overwrite_flag=overwrite)


    # Start iterations
    for p, P in enumerate(P_vec):
        ### Logging ###
        print(f"P={P:02d}. Status:")

        # Check if data is there
        if overwrite or np.all(np.isnan(outcomes[:, p])):

            args = (cmn.aoii_hbins, cmn.mse_hbins, cmn.mse_maxval, cmn.T, cmn.R, cmn.N, cmn.max_age, anomaly_rate, cmn.SIGMA, aoii_thr, mse_thr,
                        cmn.C, cmn.D, cmn.F, cmn.F, cmn.H, cmn.sigma_w, cmn.sigma_v, cmn.sigma_w_hat,
                        cmn.sigma_v_hat, manager, P, cmn.ETA, cmn.BLER_TARGET, debug)

            start_time = time.time()
            if parallel:
                with ProcessPoolExecutor(mp_context=mp.get_context("spawn"), max_workers=min(10, os.cpu_count() // 4)) as executor:
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
            outcomes[0, p] = np.dot(anom_aoii_hist, np.arange(0, cmn.aoii_hbins + 1, 1))
            outcomes[1, p] = np.where(anom_aoii_cdf > 0.99)[0][0]
            outcomes[2, p] = np.where(anom_aoii_cdf > 0.999)[0][0]

            # DT drifts
            mse_cdf = np.cumsum(mse_hist) / cmn.mse_hbins * cmn.mse_maxval
            outcomes[3, p] = np.dot(mse_values, mse_hist) / np.sum(mse_hist)
            outcomes[4, p] = mse_values[np.where(mse_cdf > 0.99)[0][0]]
            outcomes[5, p] = mse_values[np.where(mse_cdf > 0.999)[0][0]]

            # Generate data frame and save it (redundant but to avoid to lose data for any reason)
            df = pd.DataFrame(outcomes.T.round(dec), columns=metrics)
            df.insert(0, 'P', P_vec)
            df.to_csv(filename, index=False)

            # Print time
            elapsed = time.time() - start_time
            print(f"\t...done in {elapsed:.3f} seconds")

        else:
            print("\t...already done!")
            continue
