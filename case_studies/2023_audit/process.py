from core.risk_engine import RiskCalculator
from core.state_manager import StateManager
import pandas as pd

class CaseStudy:
    def __init__(self):
        self.results = []

    def simulate_processing(self, input_file):
        df = pd.read_csv(input_file)
        risk_engine = RiskCalculator()
        state_mgr = StateManager()

        for _, row in df.iterrows():
            state_mgr.add_return(row['return_id'])
            for stage in ['pre-filing', 'initial', 'post-processing']:
                score = risk_engine.calculate_score(stage, row[f'{stage}_fraud_prob'], row['revenue_impact'])
                state_mgr.update_stage(row['return_id'], stage, score)

            self.results.append({
                'return_id': row['return_id'],
                'final_score': state_mgr.get_current_risk(row['return_id']),
                'processing_stages': state_mgr.export_audit_trail(row['return_id'])
            })

        return pd.DataFrame(self.results)

if __name__ == "__main__":
    study = CaseStudy()
    results = study.simulate_processing('input_data.csv')
    results.to_csv('case_study_results.csv', index=False)
