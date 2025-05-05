import numpy as np
import matplotlib.pyplot as plt

class TaxReturnPrioritizer:
    def __init__(self, alpha=0.7, revenue_threshold=1e6):
        self.alpha = alpha
        self.revenue_threshold = revenue_threshold

    def calculate_composite_score(self, fraud_prob, revenue_impact):
        normalized_revenue = revenue_impact / self.revenue_threshold
        return self.alpha * fraud_prob + (1 - self.alpha) * normalized_revenue

    def determine_priority(self, fraud_prob, revenue_impact):
        score = self.calculate_composite_score(fraud_prob, revenue_impact)
        if fraud_prob > 0.8 and revenue_impact > 0.7:
            return "Investigate Immediately"
        elif fraud_prob > 0.6 and revenue_impact > 0.5:
            return "Secondary Check"
        elif fraud_prob > 0.4 or revenue_impact > 0.4:
            return "Selective Review"
        else:
            return "Monitor Only"


class PriorityMatrix:
    def __init__(self, risk_bins=10, impact_bins=10):
        self.risk_bins = risk_bins
        self.impact_bins = impact_bins

    def generate_matrix(self, states):
        matrix = np.zeros((self.risk_bins, self.impact_bins))
        for state in states.values():
            r_bin = int(state["stages"]["post_processing"]["fraud_prob"] * (self.risk_bins - 1))
            i_bin = int(state["revenue_impact"] / 1e6 * (self.impact_bins - 1))
            matrix[r_bin, i_bin] += 1
        return matrix

    def plot_heatmap(self, matrix, save_path=None):
        plt.figure(figsize=(10, 8))
        plt.imshow(matrix, cmap="RdYlGn_r", interpolation="nearest")
        plt.colorbar(label="Case Count")
        plt.xlabel("Revenue Impact (Deciles)")
        plt.ylabel("Fraud Probability (Deciles)")
        plt.title("Fraud Prioritization Matrix")
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
