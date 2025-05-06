import numpy as np
from typing import Dict, Tuple

class RiskCalculator:
    def __init__(self, alpha=0.7, dynamic_weighting=True):
        self.alpha = alpha
        self.dynamic_weighting = dynamic_weighting
        self.revenue_max = None

    def initialize(self, historical_data: Dict):
        """Calculate normalization baselines from historical data."""
        revenues = [r['revenue_impact'] for r in historical_data.values() if r['revenue_impact'] is not None]
        if not revenues:
            raise ValueError("No valid revenue impact values found in historical data.")
        self.revenue_max = np.percentile(revenues, 90)
        print(f"[Init] Set revenue_max to {self.revenue_max:.2f}")

    def calculate_score(self, stage: str, fraud_prob: float, revenue_impact: float) -> float:
        """Compute the weighted fraud risk score for a tax return."""
        if self.revenue_max is None:
            raise ValueError("Revenue max not initialized. Call `initialize()` first or set manually.")

        if revenue_impact is None:
            raise ValueError("Revenue impact is None. Cannot compute risk score.")

        if self.dynamic_weighting:
            stage_weights = {
                'pre-filing': 0.5,
                'initial': 0.7,
                'post-processing': 0.9
            }
            alpha = stage_weights.get(stage, self.alpha)
        else:
            alpha = self.alpha

        normalized_rev = min(revenue_impact / self.revenue_max, 1.0)
        score = alpha * fraud_prob + (1 - alpha) * normalized_rev
        return round(score, 4)

    def confidence_interval(self, stage: str, n_samples: int = 1) -> Tuple[float, float]:
        """Return confidence interval ranges per stage."""
        ci_map = {
            'pre-filing': (-0.25, 0.25),
            'initial': (-0.15, 0.15),
            'post-processing': (-0.05, 0.05)
        }
        return ci_map.get(stage, (-0.2, 0.2))  # default range if stage unknown