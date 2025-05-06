from core.risk_engine import RiskCalculator

def test_risk_calculation():
    calculator = RiskCalculator(alpha=0.7)
    calculator.revenue_max = 100000

    score = calculator.calculate_score('pre-filing', 0.65, 50000)
    assert 0.4 < score < 0.7

    calculator.dynamic_weighting = True
    score_post = calculator.calculate_score('post-processing', 0.8, 100000)
    assert score_post > 0.85