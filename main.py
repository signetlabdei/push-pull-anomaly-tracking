import numpy as np
import matplotlib.pyplot as plt
import local_scheduler as l_sch
import distributed_scheduler as d_sch

# Main system parameters
nodes = 100
max_age = 100
M = 100
P = 10
Q = 10
T = int(1e5)
p_c = 0.2
mode = 1
debug_mode = False

# Anomaly parameters
local_anomaly_rate = 0.035
distributed_cluster_size = 4
distributed_cluster_number = 10
p_01 = 0.01
p_11 = 0.9

# Prioritization
aoii_threshold = 5

# Utility variables
clustered = distributed_cluster_number * distributed_cluster_size   # The first clustered nodes have distributed anomalies
local_state = np.zeros(nodes)
distributed_state = np.zeros(clustered)
aoii = np.zeros((T, nodes))
scheduled = []

# Instantiate schedulers
local_sched = l_sch.LocalAnomalyScheduler(nodes, max_age, local_anomaly_rate, mode, debug_mode)
dist_sched = d_sch.DistributedAnomalyScheduler(clustered, distributed_cluster_size, p_01, p_11, debug_mode)

for t in range(T):
    ### ANOMALY GENERATION ###
    local_state = np.minimum(np.ones(nodes), local_state + np.asarray(np.random.rand(nodes) < local_anomaly_rate))
    new_state = np.zeros(clustered)
    for i in range(clustered):
        p = p_01
        if (distributed_state[i] == 1):
            # TODO check if cluster is in anomaly state!
            if (p == p_01):
                p = p_11
            else:
                p = 1
        new_state[i] = np.random.rand() < p
    distributed_state = new_state
    if (t > 0):
        aoii[t, :] = aoii[t - 1, :] + local_state
    else:
        aoii[t, :] = local_state

    ### SUBFRAME ALLOCATION ###
    local_risk = local_sched.get_risk(aoii_threshold)
    dist_risk = dist_sched.get_risk()
    # TODO outer loop: decide the values of P and Q


    ### PULL-BASED SUBFRAME ###
    scheduled = dist_sched.schedule(Q)
    # print('s', scheduled)
    # Fix local anomalies in scheduled slots
    aoii[t, scheduled] = 0
    local_state[scheduled] = 0
    # TODO do we consider this in the push threshold decision?

    ### PUSH-BASED SUBFRAME ###
    # Get local anomaly threshold
    threshold = local_sched.schedule(P, p_c, scheduled)

    # Select random slots for active nodes
    choices = np.random.randint(1, P + 1, nodes) * np.asarray(aoii[t, :] > threshold)
    outcome = np.zeros(P, dtype=int)
    successful_push = []
    for p in range(1, P + 1):
        chosen = np.where(choices == p)[0]
        if (chosen.size != 0):
            if (chosen.size == 1):
                if (chosen[0] < clustered):
                    successful_push.append(chosen[0])
                outcome[p - 1] = chosen[0] + 1
                local_state[chosen[0]] = 0
                aoii[t, chosen[0]] = 0
            else:
                outcome[p - 1] = -1

    # Local and distributed anomaly belief update
    local_sched.update_psi(threshold, outcome)
    successful = np.append(scheduled, np.asarray(successful_push, dtype=int))
    dist_sched.update_zeta(successful, distributed_state[successful])
    # TODO distributed scheduler: find anomalies and reset them

    ### LOGGING ###
    if(np.mod(t,1000) == 0):
        print('Step:', t)
    if (debug_mode):
        print('s', local_state)
        print('t', threshold)
        print('c', choices)
        print('o', outcome)
        print('a',aoii[t,:])
        input("Press Enter to continue...")


# Plotting local anomaly AoII
aoii_tot = np.reshape(aoii, T * nodes)
values = np.arange(0, max_age + 1, 1)
plt.hist(aoii_tot, values - 0.5, density = True)
plt.show()
