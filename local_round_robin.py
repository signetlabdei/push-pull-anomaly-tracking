import numpy as np
import math

class LocalAnomalyScheduler:
    N = 0
    n = 0

    def __init__(self, N):
        self.N = N
        self.n = 0

    def schedule(self, P, pull_sch):
        scheduled = np.zeros(P)
        for p in range(P):
            while (n in pull_sch):
                n = np.mod(n + 1, N)
            scheduled[p] = n
            n += 1
        return scheduled
