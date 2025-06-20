import numpy as np
import math

class LocalAnomalyScheduler:
    N = 0
    psi = []
    lambdas = []
    debug_mode = False

    def __init__(self, N, activation, debug_mode):
        self.N = N
        self.psi = np.ones(N)
        self.lambdas = np.ones(N) * activation
        self.debug_mode = debug_mode

    def schedule(self, P, Q, p_thr):
        if self.debug_mode:
            print('p', self.psi)
        for threshold in np.arange(0, np.max(self.psi) - 1, 1):
            act_prob = np.zeros(self.N)
            for n in range(self.N):
                act_prob[n] = 1 - np.power(1 - self.lambdas[n], np.max([0, self.psi[n] - threshold]))
            A = np.sum(act_prob > 0)
            ## TODO separate nodes that are too high
            activation = np.mean(act_prob)
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

    def update_psi(self, threshold, outcome):
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
