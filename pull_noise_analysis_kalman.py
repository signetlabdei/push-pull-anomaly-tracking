import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

from pull_process_analysis_kalman import run_episode
import common as cmn


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
    Q = 10
    sigma_w_hat_vec = np.arange(0.05, 1.05, 0.05).round(dec)
    sigma_w = 0.5


    # Check if results data exist and load it if there
    data_results = (schedulers, sigma_w_hat_vec)
    prefix = 'pull_noise_kalman_' + f"{int(sigma_w * 100)}"
    prob_avg, filename_avg = cmn.check_data(data_results, prefix + '_avg', pull_folder, overwrite_flag=overwrite)
    prob_99, filename_99 = cmn.check_data(data_results, prefix + '_99', pull_folder, overwrite_flag=overwrite)
    prob_999, filename_999 = cmn.check_data(data_results, prefix + '_999', pull_folder, overwrite_flag=overwrite)

    for s, _ in enumerate(schedulers):
        for m, sigma_w_hat in enumerate(sigma_w_hat_vec):
            print(f"Test: sched={schedulers[s]}, sigma_w={sigma_w_hat:.2f}. Status:")

            # Check if data is there
            if overwrite or np.isnan(prob_avg[s, m]):
                args = (s, cmn.bins, cmn.maxval, cmn.T, cmn.C, cmn.D,
                        Q, cmn.F, cmn.H, sigma_w, cmn.sigma_v,
                        sigma_w_hat, cmn.sigma_v_hat, debug)

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
                mse_cdf = np.cumsum(mse_hist) / cmn.bins * cmn.maxval
                prob_avg[s, m] = np.dot(mse_values, mse_hist) / np.sum(mse_hist)
                prob_99[s, m] = mse_values[np.where(mse_cdf > 0.99)[0][0]]
                prob_999[s, m] = mse_values[np.where(mse_cdf > 0.999)[0][0]]
                if debug:
                    print(prob_avg, prob_99, prob_999)

                # Generate data frame and save it (redundant but to avoid to lose data for any reason)
                for res, file in [(prob_avg, filename_avg), (prob_99, filename_99), (prob_999, filename_999)]:
                    df = pd.DataFrame(res.T.round(dec), columns=schedulers)
                    df.insert(0, 'sigma_w_hat', sigma_w_hat_vec)
                    df.to_csv(file, index=False)

                # Print time
                elapsed = time.time() - start_time
                print(f"\t...done in {elapsed:.3f} seconds")

            else:
                print("\t...already done!")
                continue
