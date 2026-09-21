from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime

class AgentStep(BaseModel):
    """Represents one step in the agent's reasoning"""
    thought: str  # What agent is thinking
    tool_call: Optional[str] = None  # Tool it wants to call
    tool_result: Optional[Dict] = None  # Result from tool

class RetrainingAgent(BaseModel):
    """Complete agent state"""
    task: str = "Decide whether to promote retraining model"
    feedback_rows: int = 0
    
    # Agent reasoning history
    steps: List[AgentStep] = []
    
    # Final results
    current_f1: Optional[float] = None
    current_recall: Optional[float] = None
    candidate_f1: Optional[float] = None
    candidate_recall: Optional[float] = None
    feedback_quality: Optional[str] = None
    
    # Agent's final decision
    final_decision: Optional[str] = None  # "promote" or "reject"
    final_reasoning: Optional[str] = None
    confidence: Optional[str] = None
    
    version_number: Optional[int] = None
    timestamp: str = ""
    error: Optional[str] = None