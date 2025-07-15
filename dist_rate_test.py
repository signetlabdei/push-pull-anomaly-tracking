import numpy as np
from common import C, frame_duration

def index_to_cluster_state(idx, cluster_size) -> np.ndarray:
    """Translator from an index in the pmf to a cluster state.

    :param idx: the index
    :param cluster_size: the number of nodes in the cluster
    :return: cluster state vector
    """
    state = np.array(list(np.binary_repr(idx)), dtype=int)
    return np.pad(state, cluster_size - len(state))[:cluster_size]

def init_transition_matrix(cluster_size, p_01, p_11) -> np.ndarray:
    """Initialize the transition matrix given the initial transition probabilities

    :param p_01: initial transition from non anomaly to anomaly
    :param p_11: initial transition from anomaly to anomaly

    :return: 2^C times 2^C transition matrix (U in the paper)
    """
    states = np.power(2, cluster_size)
    transition_matrix = np.ones((states, states))

    for state_ind in range(states):
        p = np.zeros((2,2))
        p[1, 0] = 1 - p_11
        p[1, 1] = p_11
        state = index_to_cluster_state(state_ind, cluster_size)
        anomalies = np.sum(state)
        # When an anomaly occurs, nodes in state 1 do not go back to state 0
        if anomalies >= cluster_size / 2:
            p[1,0] = 0
            p[1,1] = 1
        for next_state_ind in range(states):
            next_state = index_to_cluster_state(next_state_ind, cluster_size)
            for s in range(cluster_size):
                p[0,1] = p_01[s]
                p[0,0] = 1 - p_01[s]
                transition_matrix[state_ind, next_state_ind] *= p[state[s], next_state[s]]

    return transition_matrix

def get_transient_matrix(transition_matrix, cluster_size):
    transient_states = []
    states = np.size(transition_matrix, 0)
    for state_ind in range(states):
        state = index_to_cluster_state(state_ind, cluster_size)
        anomalies = np.sum(state)
        if anomalies < cluster_size / 2:
            transient_states.append(state_ind)
    Q = np.zeros((len(transient_states), len(transient_states)))
    for ind in range(len(transient_states)):
        s = transient_states[ind]
        for next_ind in range(len(transient_states)):
            ns = transient_states[next_ind]
            Q[ind, next_ind] = transition_matrix[s, ns]
    return Q

def get_absorption_time(transition_matrix, cluster_size):
    Q = get_transient_matrix(transition_matrix, cluster_size)
    t = np.size(Q, 0)
    np.eye(t) - Q
    N = np.linalg.inv(np.eye(t) - Q)
    return np.dot(N, np.ones(t))

### MAIN SCRIPT ###
if __name__ == '__main__':
    cluster_size = C
    p_11 = 0.9

    p_01_base = np.array([0.001, 0.00704, 0.00725, 0.00750])
    multiplier = np.array([1., 1.508, 1.939, 2.332])
    for mult in multiplier:
        p_01 = mult * p_01_base
        print(f'p_01: {np.round(p_01, 5)}')
        transition_matrix = init_transition_matrix(cluster_size, p_01, p_11)
        T = get_absorption_time(transition_matrix, cluster_size)
        print(f'\tAbsorption rate: {1/T[0] / frame_duration:0.3f}')
