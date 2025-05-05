import json
from datetime import datetime
from typing import Dict, Any

class ReturnState:
    def __init__(self):
        self.stages = {
            "pre_filing": {"fraud_prob": 0.0, "features": {}, "timestamp": None},
            "initial_filing": {"fraud_prob": 0.0, "features": {}, "timestamp": None},
            "post_processing": {"fraud_prob": 0.0, "features": {}, "timestamp": None}
        }
        self.priority = "monitor"
        self.revenue_impact = 0.0

class StateManager:
    def __init__(self):
        self.state_store = {}
    
    def update_stage(self, return_id: str, stage: str, data: Dict[str, Any]):
        if return_id not in self.state_store:
            self.state_store[return_id] = ReturnState()
        self.state_store[return_id].stages[stage] = {
            **data,
            "timestamp": datetime.now().isoformat()
        }
        self._recalculate_priority(return_id)

    def _recalculate_priority(self, return_id: str):
        state = self.state_store[return_id]
        risk_score = 0.7 * state.stages['post_processing']['fraud_prob'] + 0.3 * (state.revenue_impact / 1e6)
        if risk_score > 0.8:
            state.priority = "investigate_immediately"
        elif risk_score > 0.6:
            state.priority = "secondary_review"
        else:
            state.priority = "monitor"

    def save_state(self, path: str):
        with open(path, 'w') as f:
            json.dump({rid: vars(state) for rid, state in self.state_store.items()}, f)
