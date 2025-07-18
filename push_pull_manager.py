import numpy as np


class ResourceManager:

    manager_type = 0
    P = 0
    R = 0
    min_res = 0
    hysteresis = 0

    def __init__(self, manager_type, R):
        self.manager_type = manager_type
        self.R = R
        self.P = int(np.floor(R / 2))
        self.min_res = 1
        self.hysteresis = 0

    def set_push_resources(self, P):
        self.P = P

    def set_min_threshold(self, min_res):
        self.min_res = min_res

    def set_hysteresis(self, hyst):
        self.hysteresis = hyst

    def allocate_resources(self, local_risk, twin_risk):
        push_ratio = 1
        if (local_risk + twin_risk > 0):
            push_ratio = local_risk / (local_risk + twin_risk)
        if (self.manager_type == 1):        # Step by step allocation
            self.P = int(np.floor(self.R * push_ratio))
        if (self.manager_type == 2):        # Adaptive rate allocation
            if (local_risk != twin_risk):
                if (local_risk > twin_risk + self.hysteresis):
                    self.P += 1
                if (local_risk < twin_risk - self.hysteresis):
                    self.P -= 1

        if (self.P < self.min_res):
            self.P = self.min_res
        if (self.P > self.R - self.min_res):
            self.P = self.R - self.min_res
        return self.P, self.R - self.P

