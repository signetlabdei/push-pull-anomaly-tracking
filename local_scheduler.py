import numpy as np
import math

class LocalAnomalyScheduler:
    N = 0
    psi = []
    active = 0
    debug_mode = False
    scheduler_type = 0

    def __init__(self, N, M, activation, scheduler_type, debug_mode):
        self.N = N
        self.psi = np.zeros((N,M))
        self.psi[:, 0] = 1 - activation
        self.psi[:, 1] = activation
        self.active = activation
        self.scheduler_type = scheduler_type
        self.debug_mode = debug_mode

    def update_prior(self):
        self.psi[:, 2:] = self.psi[:, 1:-1]
        self.psi[:, 0] *= (1 - self.active)
        self.psi[:, 1] = self.psi[:, 0] * self.active

    def schedule(self, P, p_thr, pull_sch):
        if (self.debug_mode):
            print('p', self.psi[0,:])
        # Remove pulled nodes from schedulable
        for pull_node in pull_sch:
            self.psi[pull_node, 0] = 1
            self.psi[pull_node, 1:] = 0

        max_ages = np.ones(self.N)
        for n in range(self.N):
            max_age_n = np.where(self.psi[n, :] > 0)[0]
            if(len(max_age_n) > 0):
                max_ages [n] = max_age_n[-1]
        max_age = np.max(max_ages)
        p_coll = 0
        threshold = max_age - 2
        valid = False
        while (p_coll < p_thr and threshold >= 0):
            act_prob = np.zeros(self.N)
            if (self.scheduler_type == 0):
                for n in range(self.N):
                    if (max_ages[n] > threshold):
                        act_prob[n] = np.power(self.active, max_ages[n] - threshold)
            if (self.scheduler_type == 1):
                    for n in range(self.N):
                        act_prob[n] = np.sum(self.psi[n, int(threshold) + 1:])
            p_coll = self.__eval_threshold(P, act_prob)
            if (self.debug_mode):
                print('z',threshold, act_prob, p_coll)
            if (p_coll < p_thr):
                valid = True
            threshold -= 1
        if (valid):
            return int(threshold + 1)
        else:
            return int(np.max([0, max_age - 2]))

    def update_psi(self, threshold, outcome):
        if (self.scheduler_type == 0):
            return self.__pessimistic_update_psi(threshold, outcome)
        if (self.scheduler_type == 1):
            return self.__bayes_update_psi(threshold, outcome)

    def get_risk(self, threshold):
        return np.sum(self.psi[:, threshold:]) / self.N

    def __eval_threshold(self, P, act_prob):
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
        for i in range(self.N):
            # Node i successfully transmitted, we reset AoII to 0
            if (i + 1 in outcome):
                self.psi[i, :] = 0
                self.psi[i, 0] = 1
        # There are no collisions
        if (np.min(outcome) > -0.1):
            # No remaining node has an AoII higher than the threshold
            for n in range(self.N):
                idx = np.where(self.psi[n, :] == 1)[0]
                if (idx > threshold):
                    self.psi[n, threshold] = 1
                    self.psi[n, idx] = 0
        # Next time step
        for n in range(self.N):
            self.psi[n, 1:] = self.psi[n, :-1]
            self.psi[n, 0] = 0

    def __bayes_update_psi(self, threshold, outcome):
        p_node = 0
        act_prob = np.zeros(self.N)
        for n in range(self.N):
            act_prob[n] = np.sum(self.psi[n, threshold + 1:])
        # There are no collisions
        if np.min(outcome) > -0.1:
            # No remaining node has an AoII higher than the threshold
            for n in range(self.N):
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
                if (p_a_cs[a] > 0):
                    p_node += (a - s) / (A - s) * p_a_cs[a]
                    # print('uc', p_c, p_a_cs[a], a)
        for n in range(self.N):
            # Node n successfully transmitted, we reset AoII to 0
            if (n + 1 in outcome):
                self.psi[n, 0] = 1
                self.psi[n, 1:] = 0
            else:
                # The node is a potential collider
                if (act_prob[n] > 0):
                    self.psi[n, threshold + 1:] *= p_node / act_prob[n]
                    if (np.sum(self.psi[n, :threshold]) > 0):
                        self.psi[n, :threshold] *= (1 - p_node) / np.sum(self.psi[n, :threshold])
        if (self.debug_mode):
            print('u', threshold, self.psi[0,:])

    def __bayes_collisions(self, P, A, c, s, activation, threshold):
        p_cs_a = np.zeros(A + 1)
        p_a = np.zeros(A + 1)
        for a in range(A + 1):
            p_a[a] = np.power(activation, a) * np.power(1 - activation, A - a) * math.comb(A, a)
            if (a >= 2 * c + s and p_a[a] > 1e-6):
               p_cs_a[a] = np.power(1 / P, s) * math.factorial(P) / math.factorial(P - s) * math.comb(a, s)
               p_cs_a[a] *= np.power(1 / P, 2 * c) * math.factorial(P - s) / math.factorial(P - s - c) * math.factorial(c) * math.comb(a - s, c) * math.comb(a - s - c, c) / np.power(2, c)
               p_cs_a[a] *= np.power(c / P, a - 2 * c - s) / np.power(3, a - 2 * c - s)
        p_a_cs = np.zeros(A + 1)
        p_cs = 0
        for a in range(A + 1):
            p_cs += p_cs_a[a] * p_a[a]
        if (p_cs > 0):
            for a in range(A + 1):
                p_a_cs[a] = p_cs_a[a] * p_a[a] / p_cs
        if (np.sum(p_a_cs) > 0):
            p_a_cs /= np.sum(p_a_cs)
        return p_a_cs
