import numpy as np
import matplotlib.pyplot as plt
import local_scheduler as l_sch
import distributed_scheduler as d_sch




def run_episode(M, P, T, nodes, max_age, local_anomaly_rate, p_c, mode, debug_mode):
    # Instantiate scheduler
    local_sched = l_sch.LocalAnomalyScheduler(nodes, max_age, local_anomaly_rate, mode, debug_mode)

    local_state = np.zeros(nodes)
    aoii = np.zeros((T, nodes))

    for t in range(T):
        ### ANOMALY GENERATION ###
        local_state = np.minimum(np.ones(nodes), local_state + np.asarray(np.random.rand(nodes) < local_anomaly_rate))
        if t > 0:
            aoii[t, :] = aoii[t - 1, :] + local_state
        else:
            aoii[t, :] = local_state

        ### PUSH-BASED SUBFRAME ###
        # Get local anomaly threshold
        threshold = local_sched.schedule(P, p_c, [])
        # Select random slots for active nodes
        choices = np.random.randint(1, P + 1, nodes) * np.asarray(aoii[t, :] > threshold)
        outcome = np.zeros(P, dtype=int)
        for p in range(1, P + 1):
            chosen = np.where(choices == p)[0]
            if chosen.size != 0:
                if chosen.size == 1:
                    outcome[p - 1] = chosen[0] + 1
                    local_state[chosen[0]] = 0
                    aoii[t, chosen[0]] = 0
                else:
                    outcome[p - 1] = -1

        ### POST-FRAME UPDATE ###
        # Local and distributed anomaly belief update
        local_sched.update_psi(threshold, outcome)

        ### LOGGING ###
        if np.mod(t,1000) == 0:
            print('Step:', t)

    # Plotting local anomaly AoII
    aoii_tot = np.reshape(aoii, T * nodes)
    return np.histogram(aoii_tot, bins=M + 1, range=(-0.5, M + 0.5), density=True)

def main():
    # Main system parameters
    nodes = 100
    max_age = 100
    M = 100     # S = 20
    T = int(5e3)
    episodes = 10
    mode = 1
    debug_mode = False

    # Anomaly and algorithm parameters
    frame_sizes = np.arange(5, 21, 1)
    rates = np.arange(0.01, 0.051, 0.01)
    p_c = 0.2

    prob_avg = np.zeros((len(frame_sizes), len(rates) + 1))
    prob_95 = np.zeros((len(frame_sizes), len(rates) + 1))
    prob_99 = np.zeros((len(frame_sizes), len(rates) + 1))
    prob_999 = np.zeros((len(frame_sizes), len(rates) + 1))

    prob_avg[:, 0] = frame_sizes
    prob_95[:, 0] = frame_sizes
    prob_99[:, 0] = frame_sizes
    prob_999[:, 0] = frame_sizes

    for pi in range(len(frame_sizes)):
        P = frame_sizes[pi]
        print('Frame size: ', P)
        for r in range(len(rates)):
            rate = rates[r]
            print('Rate: ', rate)
            aoii_hist = np.zeros(M + 1)
            for ep in range(episodes):
                print('Episode: ', ep)
                aoii_hist += run_episode(M, P, T, nodes, max_age, rate, p_c, mode, debug_mode)[0] / episodes
            aoii_cdf = np.cumsum(aoii_hist)
            prob_95[pi, r + 1] = np.where(aoii_cdf > 0.95)[0][0]
            prob_99[pi, r + 1] = np.where(aoii_cdf > 0.99)[0][0]
            prob_999[pi, r + 1] = np.where(aoii_cdf > 0.999)[0][0]
            prob_avg[pi, r + 1] = np.dot(aoii_hist, np.arange(0, M + 1, 1))

            np.savetxt("push_frame_avg.csv", prob_avg, delimiter=",")
            np.savetxt("push_frame_95.csv", prob_95, delimiter=",")
            np.savetxt("push_frame_99.csv", prob_99, delimiter=",")
            np.savetxt("push_frame_999.csv", prob_999, delimiter=",")


if __name__ == "__main__":
    main()

