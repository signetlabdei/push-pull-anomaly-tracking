import os
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

from push_frame_analysis import run_episode
import common as cmn


if __name__ == "__main__":
    # Parse arguments, if any
    parallel, savedir, debug, overwrite = cmn.common_parser()
    if savedir is not None:
        push_folder = savedir
    else:
        push_folder = cmn.push_folder
    # Simulation variables
    dec = 6
    pps_scheduler_mode = 1
    P = 10
    # Anomaly and algorithm parameters
    rates = np.arange(0.01, 0.051, 0.01)

    # Check if files exist and load it if there
    prefix = 'push'
    filename_cdf = os.path.join(push_folder, prefix + '_cdf.csv')

    if os.path.exists(filename_cdf) and not overwrite:
        cdf = pd.read_csv(filename_cdf).iloc[:, 1:].to_numpy().T
    else:
        cdf = np.full((len(rates), cmn.M + 1), np.nan)


    for r, rate in enumerate(rates):
        # Logging #
        print(f"Rate={rate * 100:.1f}. Status:")

        # Check if data is there
        if overwrite or np.any(np.isnan(cdf[r])):
            args = (3, cmn.M, P, cmn.T, cmn.N, cmn.max_age,
                    rate, cmn.SIGMA, pps_scheduler_mode, debug)

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
            anomaly_aoii_hist = np.mean(np.array([res[0] for res in results]), axis=0)

            # Divide data
            cdf[r] = np.cumsum(anomaly_aoii_hist)

            # Generate data frame and save it (redundant but to avoid to lose data for any reason)
            df = pd.DataFrame(cdf.T.round(dec), columns=['p1', 'p2', 'p3', 'p4', 'p5'])
            df.insert(0, 'Theta', np.arange(cmn.M + 1))
            df.to_csv(filename_cdf, index=False)

            # Print time
            elapsed = time.time() - start_time
            print(f"\t...done in {elapsed:.3f} seconds")

        else:
            print("\t...already done!")
            continue

