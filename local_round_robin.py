import numpy as np
import math

class LocalAnomalyRoundRobinScheduler:
    N = 0
    n = 0

    def __init__(self, N):
        self.N = N
        self.n = 0

    def schedule(self, P, pull_sch=[]):
        scheduled = np.zeros(P, dtype=int)
        for p in range(P):
            while (self.n in pull_sch):
                self.n = np.mod(self.n + 1, self.N)
            scheduled[p] = self.n
            self.n = np.mod(self.n + 1, self.N)
        return scheduled
