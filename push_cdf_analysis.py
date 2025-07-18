import numpy as np
from local_scheduler import LocalAnomalyScheduler


def run_episode(M, P, T, nodes, max_age, local_anomaly_rate, p_c, mode, debug_mode):
    # Instantiate scheduler
    local_sched = LocalAnomalyScheduler(nodes, max_age, local_anomaly_rate, mode, debug_mode)

    local_state = np.zeros(nodes)
    aoii = np.zeros((T, nodes))

    for t in range(T):
        ### ANOMALY GENERATION ###
        local_state = np.minimum(np.ones(nodes), local_state + np.asarray(np.random.rand(nodes) < local_anomaly_rate))

        ### COMPUTE AOII ###
        if t > 0:
            aoii[t, :] = aoii[t - 1, :] + local_state
        else:
            aoii[t, :] = local_state

        ### UPDATE SCHEDULER PRIORS ###
        local_sched.update_prior()

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
        # Local anomaly belief update
        local_sched.update_psi(threshold, outcome)

        ### LOGGING ###
        if np.mod(t,1000) == 0:
            print('Step:', t)

    # Plotting local anomaly AoII
    aoii_tot = np.reshape(aoii, T * nodes)
    return np.histogram(aoii_tot, bins=M + 1, range=(-0.5, M + 0.5), density=True)

if __name__ == "__main__":
    # Main system parameters
    nodes = 100
    max_age = 100
    M = 100     # S = 20
    P = 10
    T = int(1e4)
    episodes = 10
    mode = 1
    debug_mode = False

    # Anomaly and algorithm parameters
    rates = np.arange(0.01, 0.051, 0.01)
    p_c = 0.2

    aoii_hist = np.zeros((len(rates) + 1, M + 1))

    for r in range(len(rates)):
        rate = rates[r]
        print('Rate: ', rate)
        for ep in range(episodes):
            print('Episode: ', ep)
            aoii_hist[r + 1, :] += np.cumsum(run_episode(M, P, T, nodes, max_age, rate, p_c, mode, debug_mode)[0]) / episodes
    aoii_hist[0, :] = np.arange(0, M + 1, 1)

    np.savetxt("push_cdf.csv", np.transpose(aoii_hist), delimiter=",")

