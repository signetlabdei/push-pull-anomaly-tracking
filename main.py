import numpy as np
import matplotlib.pyplot as plt
import local_scheduler as l_sch
import distributed_scheduler as d_sch
import push_pull_manager as pp_man
from common import Q_vec, risk_thr_vec, C, D, p_01_qhet, multiplier, p_11, distributed_detection, std_bar, pull_folder

# Main system parameters
nodes = 100
max_age = 100
M = 100     # S = 20
R = 20
T = int(1e5)
mode = 1
debug_mode = False

# Anomaly parameters
local_anomaly_rate = 0.03
distributed_cluster_size = 4
distributed_cluster_number = 10
p_01 = p_01_qhet * multiplier[2]
p_11 = 0.9

# Algorithm parameters
p_c = 0.2
distributed_detection = 0.9
aoii_threshold = 3
manager_type = 1 # 0 for fixed, 1 to recompute each step, 2 for slow update (+/-1)

# Utility variables
rng = np.random.default_rng(0)
clustered = distributed_cluster_number * distributed_cluster_size   # The first clustered nodes have distributed anomalies
local_state = np.zeros(nodes)
distributed_state = np.zeros(clustered)
aoii = np.zeros((T, nodes))
scheduled = []

# Instantiate schedulers
local_sched = l_sch.LocalAnomalyScheduler(nodes, max_age, local_anomaly_rate, mode, debug_mode)
dist_sched = d_sch.DistributedAnomalyScheduler(clustered, distributed_cluster_size, p_01, p_11, debug_mode)
manager = pp_man.ResourceManager(manager_type, R)
manager.set_push_resources(10) # Set P (only used by manager 0)
manager.set_min_threshold(5) # Set minimum resources for each subframe (used by managers 1 and 2, ensures push doesn't go below 5)
manager.set_hysteresis(0.005) # Set hysteresis threshold (only used by manager 2, it won't change resource allocation if the difference between the two risks is smaller than the threshold)

for t in range(T):
    ### ANOMALY GENERATION ###
    local_state = np.minimum(np.ones(nodes), local_state + np.asarray(np.random.rand(nodes) < local_anomaly_rate))
    new_state = np.zeros(clustered)
    for node in range(clustered):
        cluster = dist_sched.cluster_map[node]
        # Rearrange the 01 transition probability on a cluster basis
        p = p_01[int(np.mod(node,distributed_cluster_size))]
        if distributed_state[node] == 1:
            # Check if the anomaly is present
            if np.sum(distributed_state[dist_sched.cluster_map == cluster]) >= distributed_cluster_size / 2:
                p = 1.
            else:
                p = p_11
        new_state[node] = rng.random() < p
    distributed_state = new_state
    if t > 0:
        aoii[t, :] = aoii[t - 1, :] + local_state
    else:
        aoii[t, :] = local_state


    ### UPDATE SCHEDULER PRIORS ###
    local_sched.update_prior()
    dist_sched.update_prior()
    # TODO add this at the beginning of each frame in other scripts

    ### SUBFRAME ALLOCATION ###
    local_risk = local_sched.get_risk(aoii_threshold)
    dist_risk = dist_sched.get_average_risk
    P, Q = manager.allocate_resources(local_risk, dist_risk) # Allocate resources
    print(local_risk,dist_risk,local_risk/dist_risk,P,Q)

    ### PULL-BASED SUBFRAME ###
    # Get pull schedule
    scheduled = dist_sched.schedule(Q, 0.3)
    # print('s', scheduled)
    # Fix local anomalies in scheduled slots
    aoii[t, scheduled] = 0
    local_state[scheduled] = 0

    ### PUSH-BASED SUBFRAME ###
    # Get local anomaly threshold
    threshold = local_sched.schedule(P, p_c, scheduled)
    # Select random slots for active nodes
    choices = np.random.randint(1, P + 1, nodes) * np.asarray(aoii[t, :] > threshold)
    outcome = np.zeros(P, dtype=int)
    successful_push = []
    for p in range(1, P + 1):
        chosen = np.where(choices == p)[0]
        if chosen.size != 0:
            if chosen.size == 1:
                if chosen[0] < clustered:
                    successful_push.append(chosen[0])
                outcome[p - 1] = chosen[0] + 1
                local_state[chosen[0]] = 0
                aoii[t, chosen[0]] = 0
            else:
                outcome[p - 1] = -1

    ### POST-FRAME UPDATE ###
    # Local and distributed anomaly belief update
    local_sched.update_psi(threshold, outcome)
    successful = np.append(scheduled, np.asarray(successful_push, dtype=int))
    cluster_in_anomaly = dist_sched.update_posterior_pmf(successful, distributed_state[successful], distributed_detection)
    # Reset state for cluster where an anomaly was found
    for cluster in cluster_in_anomaly:
        distributed_state[distributed_cluster_size * (cluster - 1): distributed_cluster_size * cluster] = 0
    # OLD DEPRECATED
    # Reset detected distributed anomalies
    # for c in range(distributed_cluster_number):
    #     anomaly = dist_sched.get_cluster_risk(c)
    #     if (anomaly >= distributed_detection):
    #         distributed_state[distributed_cluster_size * (c - 1) : distributed_cluster_size * c] = 0

    ### LOGGING ###
    if np.mod(t,1000) == 0:
        print('Step:', t)
    if debug_mode:
        print('s', local_state)
        print('t', threshold)
        print('c', choices)
        print('o', outcome)
        print('a',aoii[t,:])
        # input("Press Enter to continue...")


# Plotting local anomaly AoII
aoii_tot = np.reshape(aoii, T * nodes)
values = np.arange(0, max_age + 1, 1)
plt.hist(aoii_tot, values - 0.5, density = True)
plt.show()
