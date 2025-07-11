import numpy as np
import matplotlib.pyplot as plt
import local_scheduler as l_sch
import local_round_robin as rr_sch
import local_aloha as a_sch

def run_episode(sched_type, M, P, T, nodes, max_age, local_anomaly_rate, p_c, mode, debug_mode):
    # Instantiate scheduler
    if (sched_type == 0):
        local_sched = rr_sch.LocalAnomalyRoundRobinScheduler(nodes)
    if (sched_type == 1):
        # Maintain load close to 1
        tx_rate = 0.9 / (nodes * local_anomaly_rate / P)
    if (sched_type == 2):
        local_sched = a_sch.LocalAnomalyAlohaScheduler(nodes, local_anomaly_rate, P)
    if (sched_type == 3):
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
        if (sched_type == 0):
            outcome = local_sched.schedule(P, [])
            local_state[outcome] = 0
            aoii[t, outcome] = 0
        else:
            if (sched_type == 1):
                choices = np.random.randint(1, P + 1, nodes) * np.asarray(aoii[t, :] > 0) * (np.random.rand(nodes) < tx_rate)
            if (sched_type == 2):
                choices = np.random.randint(1, P + 1, nodes) * np.asarray(aoii[t, :] > 0) * local_sched.schedule()
            if (sched_type == 3):
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
        if (sched_type == 2):
            local_sched.update_rate(outcome)
        if (sched_type == 3):
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
    P = 10
    T = int(1e4)
    episodes = 10
    mode = 1
    debug_mode = False

    # Anomaly and algorithm parameters
    rates = np.arange(0.01, 0.051, 0.001)
    p_c = 0.2

    prob_avg = np.zeros((5, len(rates)))
    prob_95 = np.zeros((5, len(rates)))
    prob_99 = np.zeros((5, len(rates)))
    prob_999 = np.zeros((5, len(rates)))

    prob_avg[0, :] = rates * 100
    prob_95[0, :] = rates * 100
    prob_99[0, :] = rates * 100
    prob_999[0, :] = rates * 100

    for pi in range(4):
        print('Scheduler: ', pi)
        for r in range(len(rates)):
            rate = rates[r]
            print('Rate: ', rate)
            aoii_hist = np.zeros(M + 1)
            for ep in range(episodes):
                print('Episode: ', ep)
                aoii_hist += run_episode(pi, M, P, T, nodes, max_age, rate, p_c, mode, debug_mode)[0] / episodes
            aoii_cdf = np.cumsum(aoii_hist)
            prob_95[pi+1, r] = np.where(aoii_cdf > 0.95)[0][0]
            prob_99[pi+1, r] = np.where(aoii_cdf > 0.99)[0][0]
            prob_999[pi+1, r] = np.where(aoii_cdf > 0.999)[0][0]
            prob_avg[pi+1, r] = np.dot(aoii_hist, np.arange(0, M + 1, 1))

            np.savetxt("push_load_avg.csv", np.transpose(prob_avg), delimiter=",")
            np.savetxt("push_load_95.csv", np.transpose(prob_95), delimiter=",")
            np.savetxt("push_load_99.csv", np.transpose(prob_99), delimiter=",")
            np.savetxt("push_load_999.csv", np.transpose(prob_999), delimiter=",")



if __name__ == "__main__":
    main()

