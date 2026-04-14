 #
 # This file is part of the Push-Pull Medium Access repository:
 # https://github.com/signetlabdei/push-pull-anomaly-tracking
 # Copyright (c) 2025:
 # Fabio Saggese (fabio.saggese@ing.unipd.it)
 # Federico Chiariotti (federico.chiariotti@unipd.it)
 #
 # This program is free software: you can redistribute it and/or modify
 # it under the terms of the GNU General Public License as published by
 # the Free Software Foundation, version 3.
 #
 # This program is distributed in the hope that it will be useful, but
 # WITHOUT ANY WARRANTY; without even the implied warranty of
 # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 # General Public License for more details.
 #
 # You should have received a copy of the GNU General Public License
 # along with this program. If not, see <http://www.gnu.org/licenses/>.
 #

import numpy as np

def generate_drifts(state, C, F, sigma_w, rng):
    new_states = np.zeros(len(state))
    for i in range(int(len(state) / C)):
        cluster_state = np.transpose(np.asarray(state[i * C : (i + 1) * C]))
        new_states[i * C : (i + 1) * C] = np.matmul(F, cluster_state)
        new_states[i * C : (i + 1) * C] += rng.normal(np.zeros(C), np.ones(C) * sigma_w, C)
    return new_states

def generate_observations(state, C, H, sigma_v, rng):
    observations = np.zeros(len(state))
    for i in range(int(len(state) / C)):
        cluster_state = np.asarray(state[i * C : (i + 1) * C])
        observed = np.matmul(H, cluster_state) + rng.normal(np.zeros(C), np.ones(C) * sigma_v, C)
        observations[i * C : (i + 1) * C] = observed
    return observations

class PullScheduler:
    num_nodes = 0
    cluster_size = 0
    num_clusters = 0
    cluster_map = []
    F = []
    H = []
    sigma_w = 0
    sigma_v = 0
    states = []
    pred_covariances = []
    debug_mode = False

    def __init__(self, num_nodes, cluster_size, F, H, sigma_w, sigma_v,
                 rng: np.random.Generator = np.random.default_rng(), debug_mode: bool = False):
        """Constructor of the class

        :param num_nodes: number of nodes in the system :math:`N`
        :param cluster_size: size of the cluster :math:`C`
        :param F: state update matrix :math:`F`
        :param H: observation matrix :math:`H`
        :param sigma_w: process noise standard deviation :math:`sigma_w`
        :param sigma_v: observation noise standard deviation :math:`sigma_v`
        :param rng: random number generator
        :param debug_mode: boolean flag to enable debug mode`
        """
        self.F = F                                      # State update matrix
        self.H = H                                      # Observation matrix
        self.sigma_w = sigma_w                          # Process noise standard deviation
        self.sigma_v = sigma_v                          # Observation noise standard deviation
        self.num_nodes = num_nodes                      # Number of nodes (N in the paper)
        self.cluster_size = F.shape[0]                  # Clusters size (C in the paper)
        self.states = []
        self.pred_covariances = []
        assert num_nodes % self.cluster_size == 0, "Number of nodes must be a multiple of the number of clusters"
        self.num_clusters = num_nodes // cluster_size   # Number of clusters (D in the paper)
        self.cluster_map = self.init_cluster_map()      # N times 1 vector

        # Create states (we start out knowing the state and in state 0)
        for c in range(self.num_clusters):
            self.states.append(np.zeros((self.cluster_size, 1)))
            self.pred_covariances.append(np.zeros((self.cluster_size, self.cluster_size)))

        # RNG
        self.rng = rng
        # Debug
        self.debug_mode = debug_mode

    def init_cluster_map(self):
        """Initialize the cluster map with a simple clustering of contiguous N/C nodes

        :return: cluster_map N-size int vector, listing the cluster of each node
        """
        cluster_map = np.zeros(self.num_nodes, dtype=int)
        for i in range(self.num_clusters):
            cluster_map[i * self.cluster_size:(i + 1) * self.cluster_size] = i
        return cluster_map

    def update_prior(self):
        # A priori probability: update Kalman filter state for every cluster
        for cluster in range(self.num_clusters):
            self.states[cluster] = np.matmul(self.F, self.states[cluster])
            self.pred_covariances[cluster] = np.matmul(np.matmul(self.F, self.pred_covariances[cluster]), np.transpose(self.F)) + np.eye(self.cluster_size) * self.sigma_w * self.sigma_w

    def schedule_pps(self, pull_resources: int) -> np.ndarray:
        """Scheduling Pull-Push Scheduler method for nodes

        :param pull_resources: int, number of REs for pull communication (:math:`Q` in the paper)
        :return: np.ndarray of ints representing the indexes of scheduled nodes
        """
        # Node-based scheduler using
        node_priority = -np.ones(self.num_nodes)
        free_resources = pull_resources
        scheduled = -np.ones(pull_resources, dtype=int)
        # Start iterating until the resources are full
        while free_resources > 0:
            # Iterate through nodes
            for node in range(self.num_nodes):
                # We need to recompute the MSE gain
                # Check if the node is not scheduled yet and its priority has not been computed yet
                if node not in scheduled and node_priority[node] < 0:
                    # Check whether the node can be scheduled
                    cluster_nodes = np.where(self.cluster_map == self.cluster_map[node])[0]
                    # Take nodes of the cluster already scheduled (if any) and perform a mod(x, self.cluster_size) (needed for get_information)
                    prev = np.remainder(np.intersect1d(cluster_nodes, scheduled), self.cluster_size)
                    # Take the node_id and perform a mod(x, self.cluster_size) (needed for get_information)
                    node_id = np.remainder(node, self.cluster_size)
                    # Find previously scheduled nodes in the same cluster
                    if np.size(cluster_nodes) > 0:
                        nodes = np.append(prev, node_id)
                    else:
                        nodes = np.asarray([node_id], dtype=int)
                    # Compute the MSE gain
                    mse_prev = self.__get_information(self.cluster_map[node], prev)
                    mse_node = self.__get_information(self.cluster_map[node], nodes)
                    node_priority[node] = mse_prev - mse_node
            # Schedule the node with the highest priority
            next_node = np.argmax(node_priority)
            scheduled[-free_resources] = next_node
            # Reset priority value for all the nodes of the cluster of next_node
            node_priority[np.where(self.cluster_map == self.cluster_map[next_node])[0]] = -1
            free_resources -= 1
            if self.debug_mode:
                print(node_priority)
        return scheduled

    def schedule_cra(self, pull_resources: int) -> np.ndarray:
        """Benchmark Cluster Risk Aware Scheduling method

        :param pull_resources: int, number of REs for pull communication (:math:`Q` in the paper)
        :return: np.ndarray of ints representing the indexes of scheduled nodes
        """
        # Get cluster risk with the prior pmf
        cluster_risk = self.get_cluster_mse
        # Cluster-based scheduler: sort clusters by risk, then fill
        cluster_priority = np.argsort(-cluster_risk)
        free_resources = pull_resources
        cluster_idx = 0
        scheduled = -np.ones(pull_resources, dtype=int)
        # Start iterating until the resources are full
        while free_resources > 0:
            # Check if there are high-risk clusters and if there is space to schedule their nodes
            if free_resources >= self.cluster_size:
                # Iterate by cluster
                nodes = np.where(self.cluster_map == cluster_priority[cluster_idx])[0]
                # Cases to avoid that array[-x:0] returns an empty array and breaks the code
                if -free_resources + self.cluster_size == 0:
                    scheduled[-free_resources:] = nodes
                else:
                    scheduled[-free_resources : -free_resources + self.cluster_size] = nodes
                free_resources -= self.cluster_size
                cluster_idx += 1
            else:
                # Randomly take the remaining nodes from the cluster with high priority
                nodes = np.where(self.cluster_map == cluster_priority[cluster_idx])[0]
                scheduled[-free_resources:] = self.rng.choice(nodes, size=free_resources, replace=False)
                free_resources = 0
        return scheduled

    def schedule_maf(self, pull_resources: int, present_frame_idx: int) -> np.ndarray:
        """Benchmark Maximum Age First Scheduling method

        :param pull_resources: int, number of REs for pull communication (:math:`Q` in the paper)
        :param present_frame_idx: int, index of the present frame"""
        start = pull_resources * present_frame_idx
        end = start + pull_resources
        scheduled = np.arange(start, end) % self.num_nodes
        return scheduled

    def update_posterior_pmf(self,
                             scheduled: np.ndarray,
                             observations: np.ndarray) -> list:
        """Update posterior distribution using the Kalman filter gain

        :param scheduled: list of ints, index of nodes scheduled in the current frame
        :param observations: observation vector in the current frame
        :return: mse, array of the MSE for each cluster estimate
        """
        mse = np.zeros(self.num_clusters)
        for cluster in range(self.num_clusters):
            # A posteriori probability: consider observation
            cluster_ind = np.where(self.cluster_map == cluster)[0]
            received = []
            cluster_obs = []
            for n in range(self.cluster_size):
                node = cluster_ind[n]
                sched = np.where(scheduled == node)[0]
                if len(sched) > 0:
                    received.append(np.remainder(sched, self.cluster_size))
                    cluster_obs.append(observations[sched])
            received = np.intersect1d(cluster_ind, scheduled)
            # Apply posterior information over the Kalman state
            self.states[cluster], self.pred_covariances[cluster] = self.__kalman_update(self.states[cluster], self.pred_covariances[cluster], np.mod(received, self.cluster_size), cluster_obs)
            # Compute MSE
            mse[n] = np.trace(self.pred_covariances[cluster])
        return mse

    def reset_state_estimate(self) -> np.ndarray:
        """Return a vector with the state estimate across all the clusters
        and reset state estimate to 0 for numerical convergence

        :return: np.ndarray CD times 1, state estimate
        """
        state_vector = np.zeros(self.cluster_size * self.num_clusters)
        for i in range(self.num_clusters):
            state_vector[i * self.cluster_size : (i + 1) * self.cluster_size] = self.states[i].flatten()
            self.states[i] = np.zeros((self.cluster_size, 1))
        return state_vector

    @property
    def get_total_mse(self) -> float:
        """Compute the total MSE across all the clusters

        :return: float being the sum of the clusters' MSEs
        """
        return np.sum(self.get_cluster_mse)

    @property
    def get_cluster_mse(self) -> np.ndarray:
        """Compute the MSE for all clusters

        :return: np.ndarray D times 1, cluster MSE per cluster
        """
        mse = np.zeros(self.num_clusters)
        for i in range(self.num_clusters):
            mse[i] = np.trace(self.pred_covariances[i])
        return mse

    def get_actual_mse(self, actual_state) -> np.ndarray:
        """Compute the MSE for all clusters

        :param actual_state: np.ndarray CD times 1, real state of the system
        :return: np.ndarray D times 1, cluster MSE per cluster
        """
        mse = np.zeros(self.num_clusters)
        for i in range(self.num_clusters):
            cluster_state = np.asarray(actual_state[i * self.cluster_size : (i + 1) * self.cluster_size])
            error = cluster_state
            # print('a', cluster_state, np.transpose(self.states[i]))
            # print('b', error, np.sum(np.power(error, 2)))
            mse[i] = np.sum(np.power(error, 2))
        return mse


    def __get_information(self, cluster: int, nodes: np.ndarray) -> float:
        r"""Compute the sum of the a posteriori entropy times the belief :math:`\eta_k` according to eq. (21) (each line)

        :param cluster: int representing the cluster index
        :param nodes: array of the observed nodes of the cluster having a relative index from 0 to cluster_size
        :return: summation of posteriori entropy times the belief :math:`\eta_k` over the observed nodes
        """
        # If nodes is empty compute the a posteriori entropy over all the nodes of the cluster.
        # Belief \eta_k is 1 in this case
        if np.size(nodes) == 0:
            # Get the risk of the cluster itself
            info = self.get_cluster_mse[cluster]
            return info
        # If nodes is not empty compute the equation on the set of possible observations
        info = 0.
        # Find updated MSE for scheduled nodes
        covariance = self.pred_covariances[cluster]
        H = self.H[nodes, :]
        residual = np.matmul(np.matmul(H, covariance), np.transpose(H)) + np.eye(len(nodes)) * self.sigma_v * self.sigma_v
        kalman_gain = np.matmul(np.matmul(covariance, np.transpose(H)), np.linalg.inv(residual))
        new_covariance = np.matmul(np.eye(np.shape(covariance)[0]) - np.matmul(kalman_gain, H), covariance)
        return np.trace(new_covariance)

    def __kalman_update(self, state, covariance, received, observation) -> (np.ndarray, np.ndarray):
        """Kalman filter posterior update

        :param state: C times 1 np.ndarray representing the estimated prior state
        :param covariance: C times C np.ndarray representing the covariance matrix
        :param received: vector of received node indexes
        :param observation: observation vector for the nodes in the cluster under consideration
        :return: updated state and covariance matrix
        """
        H = self.H[received, :]
        if (H.shape[0] > 0):
            innovation = observation - np.matmul(H, state)
            residual = np.matmul(np.matmul(H, covariance), np.transpose(H)) + np.eye(len(received)) * self.sigma_v * self.sigma_v
            kalman_gain = np.matmul(np.matmul(covariance, np.transpose(H)), np.linalg.inv(residual))
            state += np.matmul(kalman_gain, innovation)
            covariance = np.matmul(np.eye(np.shape(covariance)[0]) - np.matmul(kalman_gain, H), covariance)
        return state, covariance
