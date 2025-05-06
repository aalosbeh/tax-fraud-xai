import os
import json
import pandas as pd
from core.risk_engine import RiskCalculator
from core.state_manager import StateManager

class CaseStudy:
    def __init__(self):
        self.results = []
        self.stage_decisions = []

    def simulate_processing(self, input_file):
        if not os.path.exists(input_file):
            raise FileNotFoundError(f" File not found: {input_file}")

        df = pd.read_csv(input_file)
        risk_engine = RiskCalculator()
        state_mgr = StateManager()

        #  Initialize normalization baseline
        historical_data = {
            row['return_id']: {'revenue_impact': row['revenue_impact']}
            for _, row in df.iterrows()
            if pd.notna(row['revenue_impact'])
        }
        risk_engine.initialize(historical_data)

        for _, row in df.iterrows():
            return_id = row['return_id']
            state_mgr.add_return(return_id)

            revenue_impact = row.get('revenue_impact')
            if pd.isna(revenue_impact):
                print(f"️ Skipping return_id {return_id} due to missing revenue_impact.")
                continue

            try:
                revenue_impact = float(revenue_impact)
            except ValueError:
                print(f"️ Skipping return_id {return_id} due to invalid revenue_impact format.")
                continue

            for stage in ['pre-filing', 'initial', 'post-processing']:
                stage_col = f"{stage}_fraud_prob"
                fraud_prob = row.get(stage_col)

                if pd.isna(fraud_prob):
                    print(f"️ Skipping stage '{stage}' for return_id {return_id} due to missing fraud probability.")
                    continue

                try:
                    fraud_prob = float(fraud_prob)
                except ValueError:
                    print(f"️ Skipping stage '{stage}' for return_id {return_id} due to invalid fraud probability.")
                    continue

                score = risk_engine.calculate_score(stage, fraud_prob, revenue_impact)
                state_mgr.update_stage(return_id, stage, score)

                # Determine action
                if score >= 0.75:
                    action = "Flag for audit"
                elif score >= 0.5:
                    action = "Review manually"
                else:
                    action = "No action"

                record = {
                    "return_id": return_id,
                    "stage": stage,
                    "fraud_prob": round(fraud_prob, 4),
                    "revenue_impact": round(revenue_impact, 2),
                    "composite_score": round(score, 4),
                    "recommended_action": action
                }

                print(record)
                self.stage_decisions.append(record)

            self.results.append({
                'return_id': return_id,
                'final_score': state_mgr.get_current_risk(return_id),
                'processing_stages': state_mgr.export_audit_trail(return_id)
            })

        return pd.DataFrame(self.results)

    def export_json(self, path):
        with open(path, "w") as f:
            json.dump(self.stage_decisions, f, indent=4)
        print(f"Stage-level JSON exported to {path}")

if __name__ == "__main__":
    input_path = "D:/myProjects/Proactive-Tax-Fraud-Detection_v2/pythonProject2/data/input_data.csv"
    output_csv = "D:/myProjects/Proactive-Tax-Fraud-Detection_v2/pythonProject2/case_studies/2023_audit/case_study_results.csv"
    output_json = "D:/myProjects/Proactive-Tax-Fraud-Detection_v2/pythonProject2/case_studies/2023_audit/stage_decisions.json"

    study = CaseStudy()
    results = study.simulate_processing(input_path)
    results.to_csv(output_csv, index=False)
    study.export_json(output_json)