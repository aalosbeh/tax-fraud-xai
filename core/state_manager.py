from datetime import datetime
import json

class ReturnState:
    def __init__(self, return_id):
        self.return_id = return_id
        self.stages = {
            'pre-filing': {'status': 'pending', 'score': None},
            'initial': {'status': 'pending', 'score': None},
            'post-processing': {'status': 'pending', 'score': None}
        }
        self.audit_history = []

    def update_stage(self, stage: str, score: float):
        self.stages[stage]['score'] = score
        self.stages[stage]['status'] = 'processed'
        self.stages[stage]['timestamp'] = datetime.now().isoformat()

    def get_current_risk(self):
        return max([s['score'] for s in self.stages.values() if s['score']])

class StateManager:
    def __init__(self):
        self.returns = {}

    def add_return(self, return_id: str):
        self.returns[return_id] = ReturnState(return_id)

    def update_stage(self, return_id: str, stage: str, score: float):
        self.returns[return_id].update_stage(stage, score)

    def get_current_risk(self, return_id: str):
        return self.returns[return_id].get_current_risk()

    def export_audit_trail(self, return_id: str):
        return json.dumps({
            'stages': self.returns[return_id].stages,
            'audit_history': self.returns[return_id].audit_history
        }, indent=2)
