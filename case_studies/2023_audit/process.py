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
            raise FileNotFoundError(f"File not found: {input_file}")

        df = pd.read_csv(input_file)
        risk_engine = RiskCalculator()
        state_mgr = StateManager()

        # Initialize revenue normalization using income_reported
        historical_data = {
            row['taxpayer_id']: {'revenue_impact': row['income_reported']}
            for _, row in df.iterrows()
            if pd.notna(row['income_reported'])
        }
        risk_engine.initialize(historical_data)

        for _, row in df.iterrows():
            taxpayer_id = row['taxpayer_id']
            state_mgr.add_return(taxpayer_id)

            income_reported = row.get('income_reported')
            if pd.isna(income_reported):
                print(f"️ Skipping taxpayer_id {taxpayer_id} due to missing income_reported.")
                continue

            try:
                income_reported = float(income_reported)
            except ValueError:
                print(f"️ Skipping taxpayer_id {taxpayer_id} due to invalid income format.")
                continue

            for stage in ['pre-filing', 'initial', 'post-processing']:
                stage_col = f"{stage}_fraud_prob"
                fraud_prob = row.get(stage_col, 0.0)  # default to 0 if not available

                if pd.isna(fraud_prob):
                    print(f"️ Skipping stage '{stage}' for taxpayer_id {taxpayer_id} due to missing fraud probability.")
                    continue

                try:
                    fraud_prob = float(fraud_prob)
                except ValueError:
                    print(f"️ Skipping stage '{stage}' for taxpayer_id {taxpayer_id} due to invalid fraud probability.")
                    continue

                score = risk_engine.calculate_score(stage, fraud_prob, income_reported)
                state_mgr.update_stage(taxpayer_id, stage, score)

                # Determine recommended action
                if score >= 0.75:
                    action = "Flag for audit"
                elif score >= 0.5:
                    action = "Review manually"
                else:
                    action = "No action"

                record = {
                    "taxpayer_id": taxpayer_id,
                    "stage": stage,
                    "fraud_prob": round(fraud_prob, 4),
                    "income_reported": round(income_reported, 2),
                    "composite_score": round(score, 4),
                    "recommended_action": action
                }

                print(record)
                self.stage_decisions.append(record)

            self.results.append({
                'taxpayer_id': taxpayer_id,
                'final_score': state_mgr.get_current_risk(taxpayer_id),
                'processing_stages': state_mgr.export_audit_trail(taxpayer_id)
            })

        return pd.DataFrame(self.results)

    def export_json(self, path):
        with open(path, "w") as f:
            json.dump(self.stage_decisions, f, indent=4)
        print(f" Stage-level decisions saved to {path}")


if __name__ == "__main__":
    input_path = "D:/myProjects/Proactive-Tax-Fraud-Detection_v2/pythonProject2/data/synthetic_tax_fraud_dataset.csv"
    output_csv = "D:/myProjects/Proactive-Tax-Fraud-Detection_v2/pythonProject2/case_studies/2023_audit/case_study_results.csv"
    output_json = "D:/myProjects/Proactive-Tax-Fraud-Detection_v2/pythonProject2/case_studies/2023_audit/stage_decisions.json"

    study = CaseStudy()
    results_df = study.simulate_processing(input_path)
    results_df.to_csv(output_csv, index=False)
    study.export_json(output_json)