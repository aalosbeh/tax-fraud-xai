  # API Reference: Proactive Tax Fraud Detection


## RiskCalculator
- `calculate_score(stage, fraud_prob, revenue_impact)`:
  - Computes time-weighted risk score (0-1 scale)
  - Stage: ['pre-filing', 'initial', 'post-processing']
  
- `confidence_interval(stage, n_samples)`:
  Returns the confidence interval for each stage.

## StateManager
- `add_return(return_id)`:
  Initializes a new return state.

- `update_stage(return_id, stage, score)`:
  Updates the processing state and stores score.
  - Maintains processing state across stages
  - Persists audit trails for compliance

- `export_audit_trail(return_id)`:
  Exports audit log of processed return.

## AdaptiveThresholdSystem
- `adjust_thresholds(workload_capacity)`
  - Dynamically modifies investigation thresholds
  - Input: Current resource availability (0-1 scale)

## FraudDetectionPipeline
Processes real-time data and prioritizes returns based on composite risk score.

## Case Study Processor
Simulates full multi-stage processing of synthetic tax data.

