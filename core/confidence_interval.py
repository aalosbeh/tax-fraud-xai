import numpy as np
from sklearn.metrics import f1_score

class ConfidenceCalculator:
    def __init__(self, n_bootstraps=1000):
        self.n_bootstraps = n_bootstraps

    def calculate_confidence(self, y_true, y_pred):
        metrics = []
        for _ in range(self.n_bootstraps):
            idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
            metrics.append(f1_score(np.array(y_true)[idx], np.array(y_pred)[idx]))
        return np.percentile(metrics, [2.5, 97.5])