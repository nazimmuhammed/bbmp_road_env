from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel

SeverityLevel = Literal["low", "medium", "high", "critical"]
RepairType    = Literal["patch", "full_repair", "emergency"]
ZoneType      = Literal["residential", "commercial", "school", "hospital", "highway"]
WeatherType   = Literal["sunny", "cloudy", "rain", "heavy_rain"]

class RoadComplaint(BaseModel):
    complaint_id:        str
    ward:                str
    road_name:           str
    zone_type:           ZoneType
    severity:            SeverityLevel
    days_pending:        int
    traffic_impact:      float
    patch_cost:          int
    full_repair_cost:    int
    population_affected: int
    complaint_count:     int

class BBMPObservation(BaseModel):
    day:              int
    budget_remaining: float
    total_budget:     float
    crews_available:  int
    crews_total:      int
    weather:          WeatherType
    complaints:       List[RoadComplaint]
    resolved_today:   int
    total_resolved:   int
    task_id:          str
    task_description: str
    metrics:          Dict[str, float]

class BBMPAction(BaseModel):
    action_type:  Literal["repair", "inspect", "wait"]
    complaint_id: Optional[str]        = None
    repair_type:  Optional[RepairType] = None
    reason:       Optional[str]        = None

class StepResult(BaseModel):
    observation: BBMPObservation
    reward:      float
    done:        bool
    info:        Dict[str, Any]

class ResetResult(BaseModel):
    observation:      BBMPObservation
    task_id:          str
    task_description: str
    max_steps:        int

class StateResult(BaseModel):
    task_id:          str
    day:              int
    budget_remaining: float
    total_reward:     float
    steps_taken:      int
    done:             bool
    metrics:          Dict[str, float]