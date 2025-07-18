import numpy as np


class ResourceManager:
    manager_type = 0
    P = 0
    R = 0
    min_res = 0
    hysteresis = 0

    def __init__(self, manager_type: int, total_resources: int):
        """ Constructor of the class.

        :param manager_type: 0: fixed, 1: proportional allocation given by the risk, 2: slow adaptive
        :param total_resources: int, number of total resource elements in the system :math:`R`
        """
        self.manager_type = manager_type
        self.R = total_resources
        self.P = int(np.floor(total_resources / 2))
        self.min_res = 1
        self.hysteresis = 0

    def set_push_resources(self, resources: int):
        self.P = resources

    def set_min_threshold(self, min_res: int):
        """Set minimum resources for push and pull. Used by manager types 1 and 2"""
        self.min_res = min_res

    def set_hysteresis(self, hyst: float):
        """Set hysteresis. Used only by manager type 2"""
        self.hysteresis = hyst

    def allocate_resources(self, local_risk: float = 0., twin_risk: float = 0.) -> tuple:
        """Allocate resources for push and pull  based on the local risk and digital twin risks."""
        # Manager is parametric
        if self.manager_type == 0:
            return self.P, self.R - self.P
        # Set push ratio depending on the risks
        push_ratio = 1
        if local_risk + twin_risk > 0:
            push_ratio = local_risk / (local_risk + twin_risk)
        # Step by step allocation
        if self.manager_type == 1:
            self.P = int(np.floor(self.R * push_ratio))
        # Adaptive rate allocation
        if self.manager_type == 2:
            if local_risk != twin_risk:
                if local_risk > twin_risk + self.hysteresis:
                    self.P += 1
                if local_risk < twin_risk - self.hysteresis:
                    self.P -= 1
        # Check minimum resources
        if self.P < self.min_res:
            self.P = self.min_res
        if self.P > self.R - self.min_res:
            self.P = self.R - self.min_res
        return self.P, self.R - self.P

