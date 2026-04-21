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

import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

from coexistence_frame_analysis_kalman import run_episode
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
    P_vec = [5, 10, 15]
    aoii_thr = 2    # Unused when testing coexistence as a search grid
    mse_thr = 10    # Unused when testing coexistence as a search grid
    anomaly_rate = 0.03
    manager = 0
    hbins = 1000

    columns = [f"{P:02d}" for P in P_vec]

    # Start cases
    # Check if files exist and load it if there
    data = (np.arange(0, hbins), P_vec)
    prefix = f"coexistence_kalman_cdf_mse"
    out_mse, filename_mse = cmn.check_data(data, prefix, coexistence_folder, overwrite_flag=overwrite)

    data = (np.arange(0, hbins + 1), P_vec)
    prefix = f"coexistence_kalman_cdf_aoii"
    out_aoii, filename_aoii = cmn.check_data(data, prefix, coexistence_folder, overwrite_flag=overwrite)

    # Start iterations
    for p, P in enumerate(P_vec):
        ### Logging ###
        print(f"P={P:02d}. Status:")

        # Check if data is there
        if overwrite or np.all(np.isnan(out_mse[:, p])):

            args = (hbins, hbins, cmn.mse_maxval, cmn.T, cmn.R, cmn.N, cmn.max_age, anomaly_rate, cmn.SIGMA, aoii_thr, mse_thr,
                    cmn.C, cmn.D, cmn.F, cmn.F, cmn.H, cmn.sigma_w, cmn.sigma_v, cmn.sigma_w_hat,
                    cmn.sigma_v_hat, manager, P, cmn.ETA, debug)

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
            mse_hist = np.mean(np.array([res[1][0] for res in results]), axis=0)
            mse_values = np.arange(0, cmn.mse_maxval, cmn.mse_maxval / hbins) + cmn.mse_maxval / hbins / 2

            # Anomalies
            mse_cdf = np.cumsum(mse_hist) / hbins * cmn.mse_maxval
            anom_aoii_cdf = np.cumsum(anom_aoii_hist)
            out_mse[:, p] = mse_cdf
            out_aoii[:, p] = anom_aoii_cdf

            # Generate data frame and save it (redundant but to avoid to lose data for any reason)
            df_mse = pd.DataFrame(out_mse.round(dec), columns=columns)
            df_mse.insert(0, 'MSE', mse_values.round(dec))
            df_mse.to_csv(filename_mse, index=False)

            df_aoii = pd.DataFrame(out_aoii.round(dec), columns=columns)
            df_aoii.insert(0, 'AoII', np.arange(0, hbins + 1))
            df_aoii.to_csv(filename_aoii, index=False)

            # Print time
            elapsed = time.time() - start_time
            print(f"\t...done in {elapsed:.3f} seconds")

        else:
            print("\t...already done!")
            continue
