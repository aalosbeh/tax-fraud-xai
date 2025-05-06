import json
import apache_beam as beam
from core.risk_engine import RiskCalculator

class FraudDetectionPipeline:
    def __init__(self):
        self.risk_calculator = RiskCalculator()

    def run(self, input_stream):
        return (
            input_stream
            | 'ParseJSON' >> beam.Map(json.loads)
            | 'CalculateRisk' >> beam.ParDo(self.calculate_risk)
        )

    def calculate_risk(self, element):
        stage = element['processing_stage']
        fp = element['fraud_probability']
        ri = element['revenue_impact']
        element['composite_score'] = self.risk_calculator.calculate_score(stage, fp, ri)
        element['confidence_interval'] = self.risk_calculator.confidence_interval(stage, 1000)
        yield element
