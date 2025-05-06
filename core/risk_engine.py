import numpy as np
from typing import Dict

class RiskCalculator:
    def __init__(self, alpha=0.7, dynamic_weighting=True):
        self.alpha = alpha
        self.dynamic_weighting = dynamic_weighting
        self.revenue_max = None
        
    def initialize(self, historical_data: Dict):
        """Calculate normalization baselines"""
        revenues = [r['revenue_impact'] for r in historical_data.values()]
        self.revenue_max = np.percentile(revenues, 90)  # Robust maximum
        
    def calculate_score(self, stage: str, fraud_prob: float, 
                       revenue_impact: float) -> float:
        """Time-aware risk scoring"""
        if self.dynamic_weighting:
            stage_weights = {
                'pre-filing': 0.5,
                'initial': 0.7,
                'post-processing': 0.9
            }
            alpha = stage_weights[stage]
        else:
            alpha = self.alpha
            
        normalized_rev = min(revenue_impact / self.revenue_max, 1.0)
        return alpha * fraud_prob + (1 - alpha) * normalized_rev

    def confidence_interval(self, stage: str, n_samples: int) -> tuple:
        """Stage-specific confidence bounds"""
        ci_map = {
            'pre-filing': (-0.25, 0.25),
            'initial': (-0.15, 0.15),
            'post-processing': (-0.05, 0.05)
        }
        return ci_map[stage]