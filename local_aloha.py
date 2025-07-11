import numpy as np
import math

class LocalAnomalyAlohaScheduler:
    N = 0
    rate = 0
    P = 0

    def __init__(self, N, anomaly_rate, P):
        self.N = N
        self.P = P
        self.rate = 0.9 / (N * anomaly_rate / P)

    def schedule(self):
        return np.random.rand(self.N) < self.rate

    def update_rate(self, outcome):
        # Count collisions and silent slots
        P = np.size(outcome)
        collisions = np.size(np.where(outcome < -0.1)) / P
        silence = np.size(np.where(outcome == 0)) / P
        self.rate += 0.1 * (collisions - silence)
        self.rate = np.max([self.rate, 0.2])
        self.rate = np.min([self.rate, 1])
