from prioritization import TaxReturnPrioritizer
from revenue_impact import calculate_revenue_impact

def process_returns(tax_returns, stage_model):
    prioritizer = TaxReturnPrioritizer()
    for return_data in tax_returns:
        stage_data = return_data['features']
        fraud_prob = stage_model.predict_proba([stage_data])[0][1]
        revenue_impact = calculate_revenue_impact(return_data)
        priority = prioritizer.determine_priority(fraud_prob, revenue_impact)
        print(f"Return ID: {return_data['id']} => Priority: {priority}, Score: {fraud_prob:.2f}, Revenue: ${revenue_impact:.2f}")