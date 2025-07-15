import numpy as np
import itertools

class DistributedAnomalyScheduler:
    num_nodes = 0
    cluster_size = 0
    num_clusters = 0
    cluster_map = []
    num_states = 2 ** cluster_size
    state_pmf = []
    transition_matrix = []
    debug_mode = False

    def __init__(self, num_nodes, cluster_size, p_01, p_11, debug_mode):
        """Constructor of the class

        :param num_nodes: number of nodes in the system :math:`N`
        :param cluster_size: size of the cluster :math:`C`
        :param p_01: initial probability :math:`U^{(i)}_{0,1}
        :param p_11: initial probability :math:`U^{(i)}_{1,1}
        :param debug_mode: boolean flag to enable debug mode`
        """
        assert num_nodes % cluster_size == 0, "Number of nodes must be a multiple of the number of clusters"
        self.num_nodes = num_nodes                      # Number of nodes (N in the paper)
        self.cluster_size = cluster_size                # Clusters size (C in the paper)
        self.num_clusters = num_nodes // cluster_size   # Number of clusters (D in the paper)
        self.cluster_map = self.init_cluster_map()      # N times 1 vector
        self.observations = np.zeros(num_nodes, dtype=int)          # o_n in the paper
        # The pmf is a 2^C vector per cluster denoting the probability of a particular state_distributed
        self.num_states = 2 ** cluster_size             # number of possible states for the state pmf
        self.state_pmf = self.init_state_pmf()          # \zeta in the paper
        self.transition_matrix = self.init_transition_matrix(p_01, p_11)    # U in the paper
        self.debug_mode = debug_mode

    def init_cluster_map(self):
        """Initialize the cluster map with a simple clustering of contiguous N/C nodes

        :return: cluster_map N-size int vector, listing the cluster of each node
        """
        cluster_map = np.zeros(self.num_nodes, dtype=int)
        for i in range(self.num_clusters):
            cluster_map[i * self.cluster_size:(i + 1) * self.cluster_size] = i
        return cluster_map

    def init_state_pmf(self) -> np.ndarray:
        """Initialize the state PMF (zeta) over the 2^C states with C clusters

        :return: state_pmf 2^C times C float matrix, each column representing the state PMF for a cluster
        """
        pmf = np.zeros((self.num_states, self.num_clusters))
        # Initialization of the state PMF with the probability of being in full 0 state equal to 1
        pmf[0, :] = 1
        return pmf

    def init_transition_matrix(self, p_01, p_11) -> np.ndarray:
        """Initialize the transition matrix given the initial transition probabilities

        :param p_01: initial transition from non anomaly to anomaly
        :param p_11: initial transition from anomaly to anomaly

        :return: 2^C times 2^C transition matrix (U in the paper)
        """
        transition_matrix = np.ones((self.num_states, self.num_states))
        # for state_ind in range(self.num_states):
        #     p = np.asarray([[1 - p_01, p_01], [1 - p_11, p_11]])
        #     state = self.index_to_cluster_state(state_ind, self.cluster_size)
        #     anomalies = np.sum(state)
        #     # When an anomaly occurs, nodes in state 1 do not go back to state 0
        #     if anomalies >= self.cluster_size / 2:
        #         p[1, 0] = 0
        #         p[1, 1] = 1
        #     for next_state_ind in range(self.num_states):
        #         next_state = self.index_to_cluster_state(next_state_ind, self.cluster_size)
        #         transition_matrix[state_ind, next_state_ind] = np.prod(p[state, next_state])

        for state_ind in range(self.num_states):
            p = np.zeros((2,2))
            p[1,0]=1-p_11
            p[1,1]=p_11
            state = self.index_to_cluster_state(state_ind, self.cluster_size)
            anomalies = np.sum(state)
            # When an anomaly occurs, nodes in state 1 do not go back to state 0
            if anomalies >= self.cluster_size / 2:
                p[1,0] = 0
                p[1,1] = 1
            for next_state_ind in range(self.num_states):
                next_state = self.index_to_cluster_state(next_state_ind, self.cluster_size)
                for s in range(self.cluster_size):
                    p[0,1] = p_01[s]
                    p[0,0] = 1 - p_01[s]
                    transition_matrix[state_ind, next_state_ind] *= p[state[s], next_state[s]]

        return transition_matrix


    def schedule(self, pull_resources: int, cluster_risk_thr: float = 0.) -> np.ndarray:
        """Scheduling method for nodes

        :param pull_resources: int, number of REs for pull communication (Q in the paper)
        :param cluster_risk_thr: float, threshold for cluster's risk. All nodes in clusters with risk of anomaly
                                higher than this should be scheduled

        :return: np.ndarray of ints representing the indexes of scheduled nodes
        """
        # A priori probability: update PMF for every cluster
        for cluster in range(self.num_clusters):
            self.state_pmf[:, cluster] = np.squeeze(np.matmul(self.state_pmf[:, cluster][np.newaxis], self.transition_matrix))

        cluster_risk = np.zeros(self.num_clusters)
        for cluster in range(self.num_clusters):
            cluster_risk[cluster] = self.get_cluster_risk(cluster)
        # Cluster-based scheduler: sort clusters by risk, then fill
        cluster_priority = np.argsort(-cluster_risk)
        node_priority = -np.ones(self.num_nodes)
        free_resources = pull_resources
        cluster_idx = 0
        scheduled = -np.ones(pull_resources, dtype=int)
        # Start iterating until the resources are full
        while free_resources > 0:
            print(free_resources)
            # Check if there are high-risk clusters and if there is space to schedule their nodes
            if cluster_risk[cluster_priority[cluster_idx]] >= cluster_risk_thr and free_resources >= self.cluster_size:
                # Iterate by cluster
                nodes = np.where(self.cluster_map == cluster_priority[cluster_idx])[0]
                # Cases to avoid that array[-x:0] returns an empty array and break the code
                if -free_resources + self.cluster_size == 0:
                    scheduled[-free_resources:] = nodes
                else:
                    scheduled[-free_resources : -free_resources + self.cluster_size] = nodes
                free_resources -= self.cluster_size
                cluster_idx += 1
            else:
                # Iterate through remaining nodes
                for node in range(self.num_nodes):
                    # We need to recompute the information gain
                    # Check if the node is not scheduled yet and its priority has not been computed yet
                    if node not in scheduled and node_priority[node] < 0:
                        # Check whether the node can be scheduled
                        cluster_nodes = np.where(self.cluster_map == self.cluster_map[node])[0]
                        prev = np.intersect1d(cluster_nodes, scheduled) - self.cluster_map[node] * self.cluster_size
                        node_id = node - self.cluster_map[node] * self.cluster_size
                        # Find previously scheduled nodes in the same cluster
                        if np.size(cluster_nodes) > 0:
                            nodes = np.append(prev, node_id)
                        else:
                            nodes = np.asarray([node_id], dtype=int)
                        # Compute the information gain
                        info_prev = self.__get_information(self.cluster_map[node], prev)
                        info_node = self.__get_information(self.cluster_map[node], nodes)
                        # print('p',prev,nodes,info_prev,info_node)
                        node_priority[node] = info_prev - info_node
                # Schedule the node with the highest priority
                next_node = np.argmax(node_priority)
                scheduled[-free_resources] = next_node
                if self.debug_mode:
                    print(node_priority)
                # Reset priority value for all the nodes of the cluster of next_node
                node_priority[np.where(self.cluster_map == self.cluster_map[next_node])[0]] = -1
                free_resources -= 1
        return scheduled

    def update_state_pmf(self,
                         scheduled: list or np.ndarray,
                         observations: np.ndarray,
                         detection_threshold: float) -> list:
        """Update map_pmf according to (16)

        :param scheduled: list of ints, index of nodes scheduled in the current frame
        :param observations: observation vector in the current frame
        :param detection_threshold: float in [0, 1], probability threshold for detection of distributed anomalies
        :return: clusters_in_anomaly, list containing cluster indexes where an anomaly was recognized
        """
        # Initialize a list containing the cluster in anomaly
        clusters_in_anomaly = []
        for cluster in range(self.num_clusters):

            # A posteriori probability: consider observation
            cluster_obs = -np.ones(self.cluster_size)
            cluster_ind = np.where(self.cluster_map == cluster)[0]
            for n in range(self.cluster_size):
                node = cluster_ind[n]
                sched = np.where(scheduled == node)[0]
                if len(sched) > 0:
                    cluster_obs[n] = observations[sched]
            # Apply indicator function and normalization given the observed values and update the PMF
            self.state_pmf[:, cluster] = self.__forward_rule(self.state_pmf[:, cluster], cluster_obs)

            # Reset distributed state if an anomaly is identified
            risk_anomaly = self.get_cluster_risk(cluster)
            # if the probability of the anomaly is higher than the threshold, the anomaly is recognized
            # Thus, we reset the state and the PMF
            if risk_anomaly >= detection_threshold:
                self.state_pmf[:, cluster] = np.hstack((1, np.zeros(self.num_states - 1)))  # Delta on the first state
                clusters_in_anomaly.append(cluster)
        return clusters_in_anomaly

    def get_risk(self) -> float:
        """Compute the average risk of the clusters

        :return: float being the average of the clusters' risk
        """
        total_risk = 0.
        for cluster in range(self.num_clusters):
            total_risk += self.get_cluster_risk(cluster)
        return total_risk / self.num_clusters

    def get_cluster_risk(self, cluster: int) -> float:
        """Compute the risk of a cluster according to (17)

        :param cluster: int representing the cluster index
        :return: float, cluster risk (belief) (17)
        """
        risk = 0
        # Check for all states
        for state_ind in range(self.num_states):
            anomaly = np.sum(np.array(list(np.binary_repr(state_ind)), dtype=int))
            if anomaly >= self.cluster_size / 2:
                risk += self.state_pmf[state_ind, cluster]
        return risk

    def __get_information(self, cluster: int, nodes: np.ndarray) -> float:
        r"""Compute the sum of the posteriori entropy times the belief :math:`\eta_k` according to eq. (21) (each line)

        :param cluster: int representing the cluster index
        :param nodes: array of the observed nodes index of the cluster
        :return: summation of posteriori entropy times the belief $\eta_k$ over the observed nodes
        """
        info = 0.
        if np.size(nodes) == 0:
            p_anomaly = 0.
            for state_ind in range(self.num_states):
                if np.sum(self.index_to_cluster_state(state_ind, self.cluster_size)) >= self.cluster_size / 2:
                        p_anomaly += self.state_pmf[state_ind, cluster]
            info = self.binary_entropy([1 - p_anomaly, p_anomaly])
            return info
        # Find possible combinations of outcomes for scheduled nodes
        patterns = np.array(list(itertools.product([0, 1], repeat=np.size(nodes))))
        for pattern in patterns:
            p_pattern = 0.       # \eta_k eq. (21) in the paper
            p_anomaly = 0.       # numerator of argument in eq. (20) in the paper
            for state_ind in range(self.num_states):
                # Check if the state matches the pattern
                if np.all(self.index_to_cluster_state(state_ind, self.cluster_size)[nodes] == pattern):
                    p_pattern += self.state_pmf[state_ind, cluster]
                    # If the state is anomalous, increase the risk
                    if np.sum(self.index_to_cluster_state(state_ind, self.cluster_size)) >= self.cluster_size / 2:
                        p_anomaly += self.state_pmf[state_ind, cluster]
            if p_pattern > 0:
                # Compute binary entropy over the conditional probability
                risk = p_anomaly / p_pattern
                # Law of total probability
                info += p_pattern * self.binary_entropy([1 - risk, risk])
        return info

    def __forward_rule(self, pmf, observation) -> np.ndarray:
        """It applies the indicator function and the normalization of eq. (16) for the forward rule

        :param pmf: 2^C times 1 np.ndarray representing the PMF of the state for a cluster
        :param observation: observation vector for the nodes in the cluster under consideration
        :return: 2^C times 1 np.ndarray representing the updated PMF of the state for the cluster
        """
        assert pmf.shape == (self.num_states,), "The PMF shape should be (self.num_states,), i.e., for a single cluster"
        missing = len(np.where(observation < 0)[0])
        # If observation are all "no transmission" nothing to do here
        if missing == self.cluster_size:
            return pmf
        # Check for each possible state
        for state_ind in range(self.num_states):
            # Convert index of the state to the state of each user
            state = self.index_to_cluster_state(state_ind, self.cluster_size)
            # Check each node
            for n in range(self.cluster_size):
                if observation[n] >= 0 and observation[n] != state[n]:
                    pmf[state_ind] = 0
        pmf /= np.sum(pmf)
        return pmf

    @staticmethod
    def index_to_cluster_state(idx, cluster_size) -> np.ndarray:
        """Translator from an index in the pmf to a cluster state.

        :param idx: the index
        :param cluster_size: the number of nodes in the cluster
        :return: cluster state vector
        """
        state = np.array(list(np.binary_repr(idx)), dtype=int)
        return np.pad(state, cluster_size - len(state))[:cluster_size]

    @staticmethod
    def cluster_state_to_index(cluster_state) -> int:
        """Translator from a cluster state to an index in the pmf.

        :param cluster_state: D-dimensional integer vector
        :return: cluster state vector index (from 0 to 2^C -1)
        """
        state_str = ''.join(str(x) for x in cluster_state)
        return int(state_str, base=2)

    @staticmethod
    def binary_entropy(prob_vector) -> float:
        """Compute the binary entropy given the probability vector.

        :param prob_vector: 2-dimensional real vector p, 1-p
        :return: binary entropy given the probability vector by eq. (19)
        """
        if prob_vector[0] == 0.5:
            return 1
        elif prob_vector[0] == 1.0 or prob_vector[0] == 0.0:
            return 0
        else:
            return -np.dot(np.log2(prob_vector), prob_vector)

