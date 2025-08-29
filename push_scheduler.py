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
import math

# Generate anomalies from a given state
def generate_anomalies(anomaly_rate: float, state: np.ndarray, rng: np.random.Generator = np.random.default_rng()):
    return np.minimum(np.ones(state.shape), state + np.asarray(rng.random(state.shape) < anomaly_rate))

class PushScheduler:
    N = 0
    psi = []
    active = 0
    debug_mode = False
    scheduler_type = 0

    def __init__(self, num_nodes, max_age, activation, scheduler_type, debug_mode):
        """Constructor of the class

        :param N: number of nodes in the system :math:`N`
        :param max_age: maximum considered AoII :math:`M`
        :param activation: initial probability :math:`\lambda`
        :param scheduler_type: 0 for naive belief, 1 for full PPS
        :param debug_mode: boolean flag to enable debug mode`
        """
        self.num_nodes = num_nodes
        self.psi = np.zeros((num_nodes, max_age))
        self.psi[:, 0] = 1 - activation
        self.psi[:, 1] = activation
        self.active = activation
        self.scheduler_type = scheduler_type
        self.debug_mode = debug_mode

    def update_prior(self):
        """Update the prior distribution over the AoII for all nodes
        """
        self.psi[:, 2:] = self.psi[:, 1:-1]
        self.psi[:, 0] *= (1 - self.active)
        self.psi[:, 1] = self.psi[:, 0] * self.active

    def schedule(self, P, p_thr, pull_sch):
        """Set the AoII threshold for transmission

        :param P: number of resources in the push subframe
        :param p_thr: maximum acceptable collision probability
        :param pull_sch: list of nodes scheduled in the pull subframe

        :return: the threshold value (integer)
        """
        if self.debug_mode:
            print('p', self.psi[0,:])
        # Remove pulled nodes from schedulable
        for pull_node in pull_sch:
            self.psi[pull_node, 0] = 1
            self.psi[pull_node, 1:] = 0

        max_ages = np.ones(self.num_nodes)
        for n in range(self.num_nodes):
            max_age_n = np.where(self.psi[n, :] > 0)[0]
            if len(max_age_n) > 0:
                max_ages [n] = max_age_n[-1]
        max_age = np.max(max_ages)
        p_coll = 0
        threshold = max_age - 2
        valid = False
        while p_coll < p_thr and threshold >= 0:
            act_prob = np.zeros(self.num_nodes)
            if self.scheduler_type == 0:
                for n in range(self.num_nodes):
                    if max_ages[n] > threshold:
                        act_prob[n] = np.power(self.active, max_ages[n] - threshold)
            if self.scheduler_type == 1:
                    for n in range(self.num_nodes):
                        act_prob[n] = np.sum(self.psi[n, int(threshold) + 1:])
            p_coll = self.__eval_threshold(P, act_prob)
            if self.debug_mode:
                print('z',threshold, act_prob, p_coll)
            if p_coll < p_thr:
                valid = True
            threshold -= 1
        if valid:
            return int(threshold + 1)
        else:
            return int(np.max([0, max_age - 2]))

    def update_psi(self, threshold, outcome):
        """Update the posterior belief distribution over AoII

        :param threshold: AoII threshold for transmission
        :param outcome: outcome of each slot in the push subframe (1+index of the successful node, 0 for silence, -1 for collision)

        :return: the belief distribution (NxM matrix)
        """
        if self.scheduler_type == 0:
            return self.__pessimistic_update_psi(threshold, outcome)
        if self.scheduler_type == 1:
            return self.__bayes_update_psi(threshold, outcome)

    def get_risk(self, threshold):
        """Get the risk of the AoII of any node passing a certain threshold

        :param threshold: the minimum dangerous AoII

        :return: the average risk (float in 0-1)
        """
        return np.sum(self.psi[:, threshold:]) / self.num_nodes

    def __eval_threshold(self, P, act_prob):
        """Get the risk of collision when setting a certain transmission threshold

        :param P: number of resources in the push subframe
        :param act_prob: the transmission probability for each node

        :return: the collision risk (float in 0-1)
        """
        A = np.sum(act_prob > 0)
        activation = np.mean(act_prob)
        p_coll = 0
        for a in np.arange(2, A + 1, 1):
            p_a = np.power(activation, a) * np.power(1 - activation, A - a) * math.comb(A, a)
            if a < P:
                p_c = 1 - math.factorial(P) / math.factorial(P - a) / np.power(P, a)
            else:
                p_c = 1
            p_coll += p_a * p_c
        return p_coll

    def __pessimistic_update_psi(self, threshold, outcome):
        """Perform a pessimistic posterior update over the belief (scheduler mode 0)

        :param threshold: AoII threshold for transmission
        :param outcome: outcome of each slot in the push subframe (1+index of the successful node, 0 for silence, -1 for collision)

        :return: the belief distribution (NxM matrix)
        """
        for i in range(self.num_nodes):
            # Node i successfully transmitted, we reset AoII to 0
            if i + 1 in outcome:
                self.psi[i, :] = 0
                self.psi[i, 0] = 1
        # There are no collisions
        if np.min(outcome) > -0.1:
            # No remaining node has an AoII higher than the threshold
            for n in range(self.num_nodes):
                idx = np.where(self.psi[n, :] == 1)[0]
                if idx > threshold:
                    self.psi[n, threshold] = 1
                    self.psi[n, idx] = 0
        # Next time step
        for n in range(self.num_nodes):
            self.psi[n, 1:] = self.psi[n, :-1]
            self.psi[n, 0] = 0

    def __bayes_update_psi(self, threshold, outcome):
        """Perform a Bayesian posterior update over the belief (scheduler mode 1)

        :param threshold: AoII threshold for transmission
        :param outcome: outcome of each slot in the push subframe (1+index of the successful node, 0 for silence, -1 for collision)

        :return: the belief distribution (NxM matrix)
        """
        p_node = 0
        act_prob = np.zeros(self.num_nodes)
        for n in range(self.num_nodes):
            act_prob[n] = np.sum(self.psi[n, threshold + 1:])
        # There are no collisions
        if np.min(outcome) > -0.1:
            # No remaining node has an AoII higher than the threshold
            for n in range(self.num_nodes):
                self.psi[n, threshold + 1:] = 0
                self.psi[n, :] /= np.sum(self.psi[n, :])
        else:
            # Compute posterior probability over the number of colliders
            A = np.sum(act_prob > 0)
            c = len(np.where(outcome < -0.1)[0])
            s = len(np.where(outcome > 0.1)[0])
            # print('uc',A,c,s)
            p_a_cs = self.__bayes_collisions(len(outcome), A, c, s, np.mean(act_prob), threshold)
            p_node = 0
            # Total probability that a specific node is a collider
            for a in np.arange(2 * c + s, A + 1, 1):
                if p_a_cs[a] > 0:
                    p_node += (a - s) / (A - s) * p_a_cs[a]
                    # print('uc', p_c, p_a_cs[a], a)
        for n in range(self.num_nodes):
            # Node n successfully transmitted, we reset AoII to 0
            if n + 1 in outcome:
                self.psi[n, 0] = 1
                self.psi[n, 1:] = 0
            else:
                # The node is a potential collider
                if act_prob[n] > 0:
                    self.psi[n, threshold + 1:] *= p_node / act_prob[n]
                    if np.sum(self.psi[n, :threshold]) > 0:
                        self.psi[n, :threshold] *= (1 - p_node) / np.sum(self.psi[n, :threshold])
        if self.debug_mode:
            print('u', threshold, self.psi[0,:])

    def __bayes_collisions(self, P, A, c, s, activation, threshold):
        """Use Bayes' theorem to compute the probability of a given number of nodes attempting a transmission for a given outcome

        :param P: number of resources (slots) in the push subframe
        :param A: number of potential transmitting nodes
        :param c: number of collided slots
        :param s: number of successful slots
        :param activation: average activation probability
        :param threshold: AoII threshold for transmission

        :return: the belief distribution (NxM matrix)
        """
        p_cs_a = np.zeros(A + 1)
        p_a = np.zeros(A + 1)
        for a in range(A + 1):
            p_a[a] = np.power(activation, a) * np.power(1 - activation, A - a) * math.comb(A, a)
            if a >= 2 * c + s and p_a[a] > 1e-6:
               p_cs_a[a] = np.power(1 / P, s) * math.factorial(P) / math.factorial(P - s) * math.comb(a, s)
               p_cs_a[a] *= np.power(1 / P, 2 * c) * math.factorial(P - s) / math.factorial(P - s - c) * math.factorial(c) * math.comb(a - s, c) * math.comb(a - s - c, c) / np.power(2, c)
               p_cs_a[a] *= np.power(c / P, a - 2 * c - s) / np.power(3, a - 2 * c - s)
        p_a_cs = np.zeros(A + 1)
        p_cs = 0
        for a in range(A + 1):
            p_cs += p_cs_a[a] * p_a[a]
        if p_cs > 0:
            for a in range(A + 1):
                p_a_cs[a] = p_cs_a[a] * p_a[a] / p_cs
        if np.sum(p_a_cs) > 0:
            p_a_cs /= np.sum(p_a_cs)
        return p_a_cs


class PushAlohaScheduler:

    def __init__(self, num_nodes, anomaly_rate, P):
        """Constructor of the class

        :param num_nodes: number of nodes in the system :math:`N`
        :param P: number of resources in the push subframe
        :param anomaly_rate: rate of anomaly generation :math:`\lambda`
        """
        self.num_nodes = num_nodes
        self.P = P
        self.rate = 0.9 / (N * anomaly_rate / P)

    def schedule(self):
        """Randomly pick nodes for transmission

        :return: the list of transmitting nodes
        """
        return np.random.rand(self.num_nodes) < self.rate

    def update_rate(self, outcome):
        """Update the rate of transmission for nodes based on feedback

        :param outcome: outcome of each slot in the push subframe (1+index of the successful node, 0 for silence, -1 for collision)
        """

        # Count collisions and silent slots
        P = np.size(outcome)
        collisions = np.size(np.where(outcome < -0.1)) / P
        silence = np.size(np.where(outcome == 0)) / P
        self.rate += 0.1 * (collisions - silence)
        self.rate = np.max([self.rate, 0.2])
        self.rate = np.min([self.rate, 1])


class PushMAFScheduler:

    def __init__(self, num_nodes):
        """Constructor of the class

        :param num_nodes: number of nodes in the system :math:`N`
        """
        self.num_nodes = num_nodes
        self.n = 0

    def schedule(self, P, pull_sch=[]):
        """Select the maximum age nodes for transmission

        :param P: number of resources in the push subframe
        :param pull_sch: list of nodes scheduled in the pull subframe

        :return: the list of scheduled nodes
        """
        scheduled = np.zeros(P, dtype=int)
        for p in range(P):
            while (self.n in pull_sch):
                self.n = np.mod(self.n + 1, self.num_nodes)
            scheduled[p] = self.n
            self.n = np.mod(self.n + 1, self.num_nodes)
        return scheduled
