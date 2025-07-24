import numpy as np
from common import frame_duration, p11

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

def nodes_ratio(p_vec):
    return np.tile(p_vec, (len(p_vec), 1)) / np.tile(p_vec, (len(p_vec), 1)).T


def compute_absorption_rate(p_01_vec: np.ndarray, p_11: float,  multiplier: np.ndarray, show :bool = True):
    """ Compute the absorption rate given the initial transition probabilities with different multipliers.
    Needed to find the actual multipliers used for running the tests.

    :param p_01_vec:
    :param p_11:
    :param multiplier:
    :param show:
    :return:
    """
    absorption_rate = np.zeros(len(multiplier))
    cluster_size = len(p_01_vec)
    # Start the loop for different loads
    for i, mult in enumerate(multiplier):
        # Different clusters
        p_01 = mult * p_01_vec
        transition_matrix = init_transition_matrix(cluster_size, p_01, p_11)
        T = get_absorption_time(transition_matrix, cluster_size)
        absorption_rate[i] = 1 / T[0] / frame_duration
        if show:
            print(f'p_01: {np.round(p_01, 5)}')
            print(f'\tAbsorption rate: {absorption_rate[i]:0.3f}')
    return absorption_rate

### MAIN SCRIPT ###
if __name__ == '__main__':

    ### QUASI HETEROGENEOUS ###
    qhet_p_01 = np.array([0.002332, 0.01642, 0.01691, 0.01749])
    qhet_mult_int = np.array([1., 1.5984, 2.1425, 2.663, 3.17])
    qhet_multipliers = np.linspace(qhet_mult_int[0], qhet_mult_int[-1], 40)  # useful for load test
    print('Quasi-heterogeneous clusters:')
    qhet_ratio = nodes_ratio(qhet_p_01)
    print('ratio\n', qhet_ratio, '\n')
    qhet_absorption_rate = compute_absorption_rate(qhet_p_01, p11, qhet_multipliers)

    compute_absorption_rate(np.array([0.00441, 0.03104, 0.03196, 0.03306]), p11, np.array([1.]))

    ### HETEROGENEOUS ###
    het_p_01 = np.array([0.001, 0.005, 0.009, 0.013])
    het_multipliers = np.array([1.974, 3.162, 4.245, 5.282, 6.292])
    print('\nHeterogeneous clusters:')
    het_ratio = nodes_ratio(het_p_01)
    print('ratio\n', het_ratio, '\n')
    het_absorption_rate = compute_absorption_rate(het_p_01, p11, het_multipliers)

    ### HOMOGENEOUS ###
    homo_p_01 = het_p_01.mean() * np.ones(len(het_p_01))
    homo_multipliers = np.array([1.82, 2.909, 3.899, 4.847, 5.771])
    print('\nHomogeneous clusters:')
    homo_ratio = nodes_ratio(homo_p_01)
    print('ratio\n', homo_ratio, '\n')
    homo_absorption_rate = compute_absorption_rate(homo_p_01, p11, homo_multipliers)
