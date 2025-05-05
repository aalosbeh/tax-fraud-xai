import numpy as np

def calculate_adaptive_thresholds(operational_context):
    base_threshold = 0.7
    capacity_factor = operational_context['available_investigators'] / 100
    time_factor = operational_context['days_remaining'] / 365
    budget_factor = operational_context['budget_remaining'] / 1e6

    adjusted_threshold = base_threshold * (1 + capacity_factor) * (1 + time_factor) * (1 + budget_factor)
    return max(0.5, min(adjusted_threshold, 0.9))


class ThresholdOptimizer:
    def __init__(self, base_threshold=0.7):
        self.base_threshold = base_threshold
        self.capacity = 1.0  # 0 = full, 1 = exhausted

    def update_capacity(self, current_workload: float):
        self.capacity = current_workload

    @property
    def dynamic_threshold(self):
        return self.base_threshold * (1 + 0.5 * self.capacity)

    def should_investigate(self, risk_score: float):
        return risk_score > self.dynamic_threshold
