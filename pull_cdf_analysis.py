import os
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

from pull_load_analysis import run_episode
import common as cmn
from drift_rate_test import compute_absorption_rate



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
    multipliers = cmn.het_multipliers
    absorption_rates = compute_absorption_rate(cmn.het_p01, cmn.p11, multipliers, show=False)

    # Check if files exist and load it if there
    prefix = 'pull_load'
    filename_cdf = os.path.join(pull_folder, prefix + '_cdf.csv')

    if os.path.exists(filename_cdf) and not overwrite:
        cdf = pd.read_csv(filename_cdf).iloc[:, 1:].to_numpy().T
    else:
        cdf = np.full((len(multipliers), cmn.M + 1), np.nan)

    for m, mult in enumerate(multipliers):
        p_01 = cmn.het_p01 * mult
        ### Logging ###
        print(f"load={absorption_rates[m]:1.3f}. Status:")

        # Check if data is there
        if overwrite or np.any(np.isnan(cdf[m])):
            args = (0, cmn.M, cmn.T, cmn.C, cmn.D, Q,
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

            # Save data
            cdf[m] = drift_aoii_cdf
            cdf_df = pd.DataFrame(cdf.T, columns=absorption_rates.round(dec))
            cdf_df.insert(0, 'Psi', np.arange(cmn.M + 1))
            cdf_df.to_csv(filename_cdf, index=False)

            # Print time
            elapsed = time.time() - start_time
            print(f"\t...done in {elapsed:.3f} seconds")

        else:
            print("\t...already done!")
            continue
