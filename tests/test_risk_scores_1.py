# test_risk_scores.py
def test_risk_calculation():
    calculator = RiskCalculator(alpha=0.7)
    calculator.initialize(historical_data)
    
    # Test pre-filing stage
    score = calculator.calculate_score(
        stage='pre-filing',
        fraud_prob=0.65,
        revenue_impact=50000
    )
    assert 0.4 < score < 0.7
    
    # Test threshold adaptation
    calculator.dynamic_weighting = True
    post_score = calculator.calculate_score(
        stage='post-processing',
        fraud_prob=0.8,
        revenue_impact=1e6
    )
    assert post_score > 0.85