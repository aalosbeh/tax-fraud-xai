def calculate_revenue_impact(return_data):
    base_impact = return_data['reported_income'] * 0.2
    deduction_impact = return_data['deductions'] * 0.25
    credit_impact = return_data['credits'] * 1.0
    return base_impact + deduction_impact + credit_impact