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
    schedulers = cmn.push_scheduler_names
    pps_scheduler_mode = 1
    P = 10
    # Anomaly and algorithm parameters
    rates = np.arange(0.01, 0.051, 0.001)

    # Check if files exist and load it if there
    prefix = 'push_load'
    filename_avg = os.path.join(push_folder, prefix + '_avg.csv')
    filename_99 = os.path.join(push_folder, prefix + '_99.csv')
    filename_999 = os.path.join(push_folder, prefix + '_999.csv')

    if os.path.exists(filename_avg) and not overwrite:
        prob_avg = pd.read_csv(filename_avg).iloc[:, 1:].to_numpy().T
    else:
        prob_avg = np.full((len(schedulers), len(rates)), np.nan)
    if os.path.exists(filename_99) and not overwrite:
        prob_99 = pd.read_csv(filename_99).iloc[:, 1:].to_numpy().T
    else:
        prob_99 = np.full((len(schedulers), len(rates)), np.nan)
    if os.path.exists(filename_999) and not overwrite:
        prob_999 = pd.read_csv(filename_999).iloc[:, 1:].to_numpy().T
    else:
        prob_999 = np.full((len(schedulers), len(rates)), np.nan)

    for s, scheduler in enumerate(schedulers):
        for r, rate in enumerate(rates):
            # Logging #
            print(f"Scheduler: {scheduler}; rate={rate*100:.1f}. Status:")

            # Check if data is there
            if overwrite or np.isnan(prob_avg[s, r]):
                args = (s, cmn.M, P, cmn.T, cmn.N, cmn.max_age,
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
                anomaly_aoii_cdf = np.cumsum(anomaly_aoii_hist)
                prob_99[s, r] = np.where(anomaly_aoii_cdf > 0.99)[0][0]
                prob_999[s, r] = np.where(anomaly_aoii_cdf > 0.999)[0][0]
                prob_avg[s, r] = np.dot(anomaly_aoii_hist, np.arange(0, cmn.M + 1, 1))

                # Generate data frame and save it (redundant but to avoid to lose data for any reason)
                for res, file in [(prob_avg, filename_avg), (prob_99, filename_99), (prob_999, filename_999)]:
                    df = pd.DataFrame(res.T.round(dec), columns=schedulers)
                    df.insert(0, 'rate', rates * 100)
                    df.to_csv(file, index=False)

                # Print time
                elapsed = time.time() - start_time
                print(f"\t...done in {elapsed:.3f} seconds")

            else:
                print("\t...already done!")
                continue
