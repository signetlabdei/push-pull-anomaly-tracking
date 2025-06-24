import numpy as np
import math

class DistributedAnomalyScheduler:
    N = 0
    C = 0
    num_clusters = 0
    cluster_map = []
    state_distributed = []
    debug_mode = False

    def __init__(self, N, C, debug_mode):
        assert N % C == 0, "Number of nodes must be a multiple of the number of cluster"
        self.N = N                      # Number of nodes
        self.C = C                      # Clusters size
        self.num_clusters = N // C      # Number of clusters (D in the paper)
        self.cluster_map = self.init_cluster_map()          # N times D vector
        self.state_distributed = np.zeros(N, dtype=int)     # y_n in the paper
        self.observations = np.zeros(N, dtype=int)          # o_n in the paper
        # The pmf is a 2^C vector per cluster denoting the probability of a particular state_distributed
        self.map_pmf = self.init_map_pmf()  # np.zeros((2 ** C, self.num_clusters)) # \zeta in the paper
        self.debug_mode = debug_mode

    @property
    def anomaly_appearance(self):   # latin z in the paper
        return np.sum(self.state_distributed[np.newaxis].T * self.cluster_map, axis=0) >= self.C / 2

    @property
    def state_distributed_cluster(self):    # [y^(i)]_{i=1}^D in the paper
        # For loop version
        tmp = np.zeros((self.C, self.num_clusters), dtype=int)
        for i in range(self.num_clusters):
            tmp[:, i] = self.state_distributed[self.cluster_map[:, i]]
        # Vectorial version (not working TODO)
        # self.state_distributed[np.newaxis].T.repeat(self.num_clusters, axis=1))[self.cluster_map]
        return tmp

    def init_cluster_map(self):
        """Initialize the cluster map with a simple clustering of contiguous N/C nodes

        :output cluster_map: N times D boolean matrix, each i-th column is a boolean vector saying which nodes belong
                            to cluster i-th
        """
        # TODO: see if nodes should be clustered with a more complex map
        cluster_map = np.zeros((self.N, self.num_clusters), dtype=bool)
        for i in range(self.num_clusters):
            cluster_map[i * self.C:(i + 1) * self.C, i] = True
        return cluster_map

    def init_map_pmf(self):
        self.map_pmf = np.zeros((2 ** C, self.num_clusters))
        self.map_pmf[:, 1] = 1   # Initialization of the state with the probability of being in full 0 state equal to 1

    @staticmethod
    def cluster_state_to_index(cluster_state):
        """Translator from a cluster state to an index in the pmf.

        :param cluster_state: D-dimensional integer vector
        """
        state_str = ''.join(str(x) for x in cluster_state)
        return int(state_str, base=2)

    def schedule(self, P, Q, p_thr):
        # TODO: to be initiated
        if self.debug_mode:
            print('p', self.psi)
        for threshold in np.arange(0, np.max(self.psi) - 1, 1):
            act_prob = np.zeros(self.N)
            for n in range(self.N):
                act_prob[n] = 1 - np.power(1 - self.lambdas[n], np.max([0, self.psi[n] - threshold]))
            A = np.sum(act_prob > 0)
            ## TODO separate nodes that are too high
            activation = np.mean(act_prob) * 5
            coll = 0
            for a in np.arange(2, A + 1, 1):
                p_a = np.power(activation, a) * np.power(1 - activation, A - a) * math.comb(A, a)
                if a < P:
                    p_c = 1 - math.factorial(P) / math.factorial(P - a) / np.power(P, a)
                else:
                    p_c = 1
                coll += p_a * p_c
            if self.debug_mode:
                print('z', threshold, coll)
            if coll < p_thr:
                return threshold
        return np.max([0, np.max(self.psi) - 2])

    def forward_rule(self, threshold, outcome):
        # TODO: to be initiaed
        for i in range(self.N):
            # Node i successfully transmitted, we reset AoII to 0
            if i in outcome:
                self.psi[i] = 0
        # There are no collisions
        if np.min(outcome) > -0.1:
            # No remaining node has an AoII higher than the threshold
            self.psi = np.minimum(self.psi, np.ones(self.N) * threshold)
        # Next time step
        self.psi += 1


if "__main__" == __name__:
    N = 40
    C = 4
    debug_mode = True
    local_sched = DistributedAnomalyScheduler(N, C, debug_mode)
    print(local_sched.anomaly_appearance)
    print(local_sched.state_distributed_cluster)