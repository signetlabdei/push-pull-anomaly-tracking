import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

import common as cmn
from pull_process_analysis_kalman import run_episode


if __name__ == '__main__':
    # Parse arguments, if any
    parallel, savedir, debug, overwrite = cmn.common_parser()
    if savedir is not None:
        pull_folder = savedir
    else:
        pull_folder = cmn.pull_folder
    # Simulation variables
    schedulers = cmn.pull_scheduler_names
    dec = 6
    Q_vec = np.arange(5, 21)

    # Check if results data exist and load it if there
    data_shape = (len(schedulers), len(Q_vec))
    prefix = 'pull_frame_kalman'
    prob_avg, filename_avg = cmn.check_data(data_shape, prefix + '_avg', pull_folder, overwrite_flag=overwrite)
    prob_99, filename_99 = cmn.check_data(data_shape, prefix + '_99', pull_folder, overwrite_flag=overwrite)
    prob_999, filename_999 = cmn.check_data(data_shape, prefix + '_999', pull_folder, overwrite_flag=overwrite)

    for s, _ in enumerate(schedulers):
        for q, Q in enumerate(Q_vec):

            print(f"Test: sched={schedulers[s]}, Q={Q:02d}. Status:")
            # Check if data is there
            if overwrite or np.isnan(prob_avg[s, q]):
                args = (s, cmn.bins, cmn.T, cmn.C, cmn.D, Q,
                        cmn.F, cmn.H, cmn.sigma_w, cmn.sigma_v,
                        cmn.sigma_w, cmn.sigma_v_hat, debug)

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
                mse_hist = np.mean(np.array([res[0] for res in results]), axis=0)
                mse_values = (results[0][1][:-1] + results[0][1][1:]) / 2    # Taking the center of each bin
                # Divide data
                mse_cdf = np.cumsum(mse_hist) / cmn.bins * 100
                prob_avg[s, q] = np.dot(mse_values, mse_hist) / np.sum(mse_hist)
                prob_99[s, q] = mse_values[np.where(mse_cdf > 0.99)[0][0]]
                prob_999[s, q] = mse_values[np.where(mse_cdf > 0.999)[0][0]]

                # Generate data frame and save it (redundant but to avoid to lose data for any reason)
                for res, file in [(prob_avg, filename_avg), (prob_99, filename_99), (prob_999, filename_999)]:
                    df = pd.DataFrame(res.T.round(dec), columns=schedulers)
                    df.insert(0, 'Q', Q_vec)
                    df.to_csv(file, index=False)

                # Print time
                elapsed = time.time() - start_time
                print(f"\t...done in {elapsed:.3f} seconds")

            else:
                print("\t...already done!")
                continue
