import numpy as np
import matplotlib.pyplot as plt
import local_scheduler as l_sch

## Main system parameters ##
nodes = 100
max_age = 100
activation = 0.04
M = 100
P = 10
Q = 10
T = int(1e4)
p_c = 0.1
debug_mode = True

## Utility variables ##
state = np.zeros(nodes)
aoii = np.zeros((T, nodes))

local_sched = l_sch.LocalAnomalyScheduler(nodes, max_age, activation, debug_mode)

for t in range(T):
    # Generate anomalies
    state = np.minimum(np.ones(nodes), state + np.asarray(np.random.rand(nodes) < activation))
    if (t > 0):
        aoii[t, :] = aoii[t - 1, :] + state
    else:
        aoii[t, :] = state
    threshold = local_sched.schedule(P, Q, p_c)
    # Select random slots for active nodes
    choices = np.random.randint(1, P + 1, nodes) * np.asarray(aoii[t, :] > threshold)
    outcome = np.zeros(P)
    for p in range(1, P + 1):
        chosen = np.where(choices == p)[0]
        if (chosen.size != 0):
            if (chosen.size == 1):
                outcome[p - 1] = chosen[0] + 1
                state[chosen[0]] = 0
                aoii[t, chosen[0]] = 0
            else:
                outcome[p - 1] = -1

    local_sched.update_psi(threshold, outcome)
    if(np.mod(t,1000) == 0):
        print('Step:', t)
    if (debug_mode):
        print('s', state)
        print('t', threshold)
        print('c', choices)
        print('o', outcome)
        print('a',aoii[t,:])
        input("Press Enter to continue...")


aoii_tot = np.reshape(aoii, T * nodes)
values = np.arange(0, 101, 1)
plt.hist(aoii_tot, values - 0.5, density = True)
plt.show()
