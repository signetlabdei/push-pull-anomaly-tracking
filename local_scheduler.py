import numpy as np
import math

class LocalAnomalyScheduler:
    N = 0
    psi = []
    active = 0
    debug_mode = False

    def __init__(self, N, M, activation, debug_mode):
        self.N = N
        self.psi = np.zeros((N,M))
        self.psi[:, 0] = 1 - activation
        self.psi[:, 1] = activation
        self.active = activation
        self.debug_mode = debug_mode

    def schedule(self, P, Q, p_thr):
        if (self.debug_mode):
            print('p', self.psi[0,:])
        max_age = 1
        for n in range(self.N):
            max_age_n = np.where(self.psi[n, :] > 0)[0]
            if(len(max_age_n) > 0):
                max_age = np.max([max_age, max_age_n[-1]])

        for threshold in np.arange(0, max_age - 1, 1):
            act_prob = np.zeros(self.N)
            for n in range(self.N):
                act_prob[n] = np.sum(self.psi[n, threshold + 1:])
            A = np.sum(act_prob > 0)
            activation = np.mean(act_prob)
            coll = 0
            for a in np.arange(2, A + 1, 1):
                p_a = np.power(activation, a) * np.power(1 - activation, A - a) * math.comb(A, a)
                if a < P:
                    p_c = 1 - math.factorial(P) / math.factorial(P - a) / np.power(P, a)
                else:
                    p_c = 1
                coll += p_a * p_c
            if (self.debug_mode):
                print('z',threshold,A,activation,coll)
            if (coll < p_thr):
                return int(threshold)
        return int(np.max([0, max_age - 2]))

    def update_psi(self, threshold, outcome):
        p_c = 0
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
            p_a_cs = self.bayes_collisions(len(outcome), A, c, s, threshold)
            p_c = 0
            # Total probability that a specific node is a collider
            for a in np.arange(1, A + 1, 1):
                if (p_a_cs[a] > 0):
                    p_c += a / (A - s) * p_a_cs[a]
        for n in range(self.N):
            # Node i successfully transmitted, we reset AoII to 0
            if (n + 1 in outcome):
                self.psi[n, 0] = 1
                self.psi[n, 1:] = 0
            else:
                # The node is a potential collider
                if (act_prob[n] > 0):
                    self.psi[n, threshold + 1:] *= p_c
                    self.psi[n, :] /= np.sum(self.psi[n, :])
        # Next time step
        self.psi[:, 1:] = self.psi[:, :-1]
        self.psi[:, 0] = self.psi[:, 1] * (1 - self.active)
        self.psi[:, 1] *= self.active
        if (self.debug_mode):
            print('u', threshold, self.psi[0,:])


    def bayes_collisions(self, P, A, c, s, threshold):
        act_prob = np.zeros(self.N)
        for n in range(self.N):
            act_prob[n] = np.sum(self.psi[n, threshold + 1:])
        activation = np.mean(act_prob)
        p_cs_a = np.zeros(A + 1)
        p_a = np.zeros(A + 1)
        for a in range(A + 1):
            p_a[a] = np.power(activation, a) * np.power(1 - activation, A - a) * math.comb(A, a)
            if (a >= 2 * c + s and p_a[a] > 1e-6):
                print(a-c-s,c)
                p_cs_a[a] = math.comb(a, c + s) * math.comb(a - c - s, c) * math.factorial(P) * math.factorial(c + s) *np.power(c, a - 2 * c - s) / (math.factorial(P - c - s) * math.factorial(c) * np.power(P, a))
                print('m',a,activation,p_a[a],p_cs_a[a],math.comb(a, c + s) * math.comb(a - c - s, c) * math.factorial(P))
        p_a_cs = np.zeros(A + 1)
        p_cs = np.sum(p_cs_a)
        print(p_a_cs,p_cs)
        for a in range(A + 1):
            p_a_cs[a] = p_cs_a[a] * p_a[a] / p_cs
        return p_a_cs
