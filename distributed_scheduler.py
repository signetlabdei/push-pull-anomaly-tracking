import numpy as np
import math
import itertools

class DistributedAnomalyScheduler:
    N = 0
    C = 0
    num_clusters = 0
    map_pmf = []
    cluster_map = []
    state_distributed = []
    transition_matrix = []
    debug_mode = False

    def __init__(self, N, C, p_01, p_11, debug_mode):
        assert N % C == 0, "Number of nodes must be a multiple of the number of clusters"
        self.N = N                      # Number of nodes
        self.C = C                      # Clusters size
        self.num_clusters = N // C      # Number of clusters (D in the paper)
        self.cluster_map = self.init_cluster_map()          # N times 1 vector
        self.state_distributed = np.zeros(N, dtype=int)     # y_n in the paper
        self.observations = np.zeros(N, dtype=int)          # o_n in the paper
        self.create_transition_matrix(p_01, p_11)     # U in the paper
        # The pmf is a 2^C vector per cluster denoting the probability of a particular state_distributed
        self.init_map_pmf()             # \zeta in the paper
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

    def create_transition_matrix(self, p_01, p_11):
        states = 2 ** self.C
        self.transition_matrix = np.zeros((states, states))
        for idx in range(states):
            p = np.asarray([[1 - p_01, p_01], [1 - p_11, p_11]])
            state = DistributedAnomalyScheduler.index_to_cluster_state(idx, self.C)
            anomalies = np.sum(state)
            if (anomalies >= self.C / 2):
                p[1, 0] = 0
                p[1, 1] = 1
            for next_idx in range(states):
                next_state = DistributedAnomalyScheduler.index_to_cluster_state(next_idx, self.C)
                self.transition_matrix[idx, next_idx] = np.prod(p[state, next_state])

    def init_cluster_map(self):
        """Initialize the cluster map with a simple clustering of contiguous N/C nodes

        :output cluster_map: N times D boolean matrix, each i-th column is a boolean vector saying which nodes belong
                            to cluster i-th
        """
        cluster_map = np.zeros(self.N, dtype=int)
        for i in range(self.num_clusters):
            cluster_map[i * self.C:(i + 1) * self.C] = i
        return cluster_map

    def init_map_pmf(self):
        self.map_pmf = np.zeros((2 ** self.C, self.num_clusters))
        self.map_pmf[0, :] = 1   # Initialization of the state with the probability of being in full 0 state equal to 1

    @staticmethod
    def index_to_cluster_state(idx, C):
        """Translator from an index in the pmf to a cluster state.

        :param idx: the index
        :param C: the number of nodes in the cluster
        """
        state = np.array(list(np.binary_repr(idx)), dtype=int)
        return np.pad(state, C - len(state))[:C]

    @staticmethod
    def cluster_state_to_index(cluster_state):
        """Translator from a cluster state to an index in the pmf.

        :param cluster_state: D-dimensional integer vector
        """
        state_str = ''.join(str(x) for x in cluster_state)
        return int(state_str, base=2)

    def schedule(self, Q, p_thr = 0):
        cluster_risk = np.zeros(self.num_clusters)
        for c in range(self.num_clusters):
            cluster_risk[c] = self.get_cluster_risk(c)
        # Simple scheduler: sort clusters by risk, then fill
        cluster_priority = np.argsort(-cluster_risk)
        node_priority = -np.ones(self.N)
        free_slots = Q
        cluster_idx = 0
        scheduled = -np.ones(Q)
        while (free_slots > 0):
            # Check if there are high-risk clusters
            if (cluster_risk[cluster_idx] >= p_thr and free_slots >= self.C):
                # Iterate by cluster
                nodes = np.where(self.cluster_map == cluster_priority[cluster_idx])[0]
                scheduled[-free_slots : -free_slots + self.C] = nodes
                free_slots -= self.C
                cluster_idx += 1
            else:
                # Iterate by remaining node
                for node in range(self.N):
                    if (node not in scheduled and node_priority[node] < 0):
                        # The node can be scheduled
                        cluster_nodes = np.where(self.cluster_map == self.cluster_map[node])[0]
                        prev = np.intersect1d(cluster_nodes, scheduled)
                        if (np.size(cluster_nodes) > 0):
                            nodes = np.append(prev, node)
                        else:
                            nodes = np.asarray([node], dtype =int)
                        nodes -= self.cluster_map[node] * self.C
                        node_priority[node] = self.__get_information(self.cluster_map[node], nodes.astype(int))
                next_node = np.argmax(node_priority)
                scheduled[-free_slots] = next_node
                node_priority[np.where(self.cluster_map == self.cluster_map[next_node])[0]] = -1
                free_slots -= 1
        return scheduled.astype(int)

    def update_zeta(self, scheduled, observations):
        # A priori probability: update PMF for every cluster
        new_pmf = np.zeros((2 ** self.C, self.num_clusters))
        for cluster in range(self.num_clusters):
            for state in range(2 ** self.C):
                new_pmf[state, cluster] = np.dot(self.map_pmf[:, cluster], self.transition_matrix[:, state])
        self.map_pmf = new_pmf
        # print('p',self.map_pmf)

        # A posteriori probability: consider observation
        for c in range(self.num_clusters):
            cluster_obs = -np.ones(self.C)
            cluster_ind = np.where(self.cluster_map == c)[0]
            for i in range(self.C):
                node = cluster_ind[i]
                sched = np.where(scheduled == node)[0]
                if (len(sched) > 0):
                    cluster_obs[i] = observations[sched]
            self.__forward_rule(c, cluster_obs)
        # print('p1',self.map_pmf)

    def get_risk(self):
        total_risk = 0
        for cluster in range(self.num_clusters):
            total_risk += self.get_cluster_risk(cluster)
        return total_risk / self.num_clusters

    def get_cluster_risk(self, cluster):
        risk = 0
        for state in range(2 ** self.C):
            anomaly = np.sum(np.array(list(np.binary_repr(state)), dtype=int))
            if (anomaly >= self.C / 2):
                risk += self.map_pmf[state, cluster]
        return risk

    def __get_information(self, cluster, nodes):
        risk = self.get_cluster_risk(cluster)
        new_risk = 0
        patterns = list(itertools.product([0, 1], repeat=np.size(nodes)))
        for pattern in patterns:
            p_arr = np.asarray(pattern, dtype=int)
            p_pattern = 0
            p_anomaly = 0
            for state in range(2 ** self.C):
                if (np.dot(self.index_to_cluster_state(state, self.C)[nodes], 1 - p_arr) == 0 ):
                    p_pattern += self.map_pmf[state, cluster]
                    anomaly = np.sum(np.array(list(np.binary_repr(state)), dtype=int))
                    if (anomaly >= self.C / 2):
                        p_anomaly += self.map_pmf[state, cluster]
            new_risk += p_pattern * p_anomaly
        old_prob = np.asarray([1 - risk, risk])
        new_prob = np.asarray([1 - new_risk, new_risk])
        return np.dot(np.log2(new_prob), new_prob) - np.dot(np.log2(old_prob), old_prob)

    def __forward_rule(self, cluster, observation):
        missing = len(np.where(observation < 0)[0])
        states = 2 ** self.C
        for idx in range(states):
            state = DistributedAnomalyScheduler.index_to_cluster_state(idx, self.C)
            for i in range(self.C):
                if (observation[i] >= 0 and observation[i] != state[i]):
                    self.map_pmf[idx, cluster] = 0

        self.map_pmf[:, cluster] /= np.sum(self.map_pmf[:, cluster])

if "__main__" == __name__:
    N = 40
    C = 4
    debug_mode = True
    local_sched = DistributedAnomalyScheduler(N, C, debug_mode)
    print(local_sched.anomaly_appearance)
    print(local_sched.state_distributed_cluster)
