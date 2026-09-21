import json
import os
from datetime import datetime
from src.agent_state import RetrainingAgent
from src.agent_tools import execute_tool
import requests

def call_ollama_for_decision(state) -> dict:
    """Call Ollama ONLY for the final decision"""
    prompt = f"""
Based on these metrics, decide whether to promote the fraud detection model.

Current Model:
- F1: {state.current_f1:.3f}
- Recall: {state.current_recall:.3f}

Candidate Model (trained with {state.feedback_rows} human feedback rows):
- F1: {state.candidate_f1:.3f}
- Recall: {state.candidate_recall:.3f}

Feedback Quality: {state.feedback_quality}

Improvement: F1 {state.current_f1:.3f} → {state.candidate_f1:.3f} (delta: {state.candidate_f1 - state.current_f1:+.3f})

Promotion rule: Candidate F1 >= Current F1 - 0.01 (tolerance for noise)

Decision: Should we PROMOTE or REJECT?

Respond EXACTLY:
DECISION: [PROMOTE or REJECT]
CONFIDENCE: [high/medium/low]
REASONING: [2-3 sentences explaining why]
"""
    
    try:
        print("\n[Ollama] Calling mistral for decision reasoning...")
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'mistral',
                'prompt': prompt,
                'stream': False,
                'temperature': 0.2
            },
            timeout=120
        )
        
        if response.status_code != 200:
            return {"status": "error", "message": f"Ollama error: {response.status_code}"}
        
        response_text = response.json()['response']
        
        decision = "promote" if "PROMOTE" in response_text.upper() else "reject"
        confidence = "high" if "HIGH" in response_text.upper() else ("low" if "LOW" in response_text.upper() else "medium")
        
        print(f"[Ollama] Decision: {decision.upper()}, Confidence: {confidence}")
        
        return {
            "status": "success",
            "decision": decision,
            "confidence": confidence,
            "reasoning": response_text
        }
    except Exception as e:
        return {"status": "error", "message": f"Ollama connection failed: {str(e)}"}


def run_agentic_retraining(feedback_rows: int = 0):
    """
    STRICT 6-STEP SEQUENCE (no loops, no free-form decisions)
    """
    
    state = RetrainingAgent(
        feedback_rows=feedback_rows,
        timestamp=datetime.now().isoformat()
    )
    
    print("\n" + "="*80)
    print("AGENTIC RETRAINING — STRICT SEQUENCE")
    print("="*80)
    print(f"Feedback rows: {feedback_rows}")
    
    # === STEP 1 ===
    print("\n[Step 1/6] Training candidate model...")
    result = execute_tool("train_candidate_model")
    if result.get("status") == "error":
        state.error = result.get("message")
        print(f"ERROR: {result.get('message')}")
        return state.model_dump()
    print(f"✓ Trained on {result.get('feedback_rows', 0)} feedback rows")
    
    # === STEP 2 ===
    print("\n[Step 2/6] Evaluating current model...")
    result = execute_tool("evaluate_current_model")
    if result.get("status") == "error":
        state.error = result.get("message")
        print(f"ERROR: {result.get('message')}")
        return state.model_dump()
    state.current_f1 = result.get("f1")
    state.current_recall = result.get("recall")
    print(f"✓ Current model — F1: {state.current_f1:.3f}, Recall: {state.current_recall:.3f}")
    
    # === STEP 3 ===
    print("\n[Step 3/6] Evaluating candidate model...")
    result = execute_tool("evaluate_candidate_model")
    if result.get("status") == "error":
        state.error = result.get("message")
        print(f"ERROR: {result.get('message')}")
        return state.model_dump()
    state.candidate_f1 = result.get("f1")
    state.candidate_recall = result.get("recall")
    print(f"✓ Candidate model — F1: {state.candidate_f1:.3f}, Recall: {state.candidate_recall:.3f}")
    
    # === STEP 4 ===
    print("\n[Step 4/6] Analyzing feedback quality...")
    result = execute_tool("analyze_feedback_quality", feedback_rows=state.feedback_rows)
    if result.get("status") == "error":
        state.error = result.get("message")
        print(f"ERROR: {result.get('message')}")
        return state.model_dump()
    state.feedback_quality = result.get("quality")
    print(f"✓ Feedback quality: {state.feedback_quality}")
    
    # === STEP 5: CALL OLLAMA FOR DECISION (ONLY ONCE) ===
    print("\n[Step 5/6] Agent reasoning (calling Ollama)...")
    decision_result = call_ollama_for_decision(state)
    if decision_result.get("status") == "error":
        state.error = decision_result.get("message")
        print(f"ERROR: {decision_result.get('message')}")
        return state.model_dump()
    
    state.final_decision = decision_result.get("decision")
    state.confidence = decision_result.get("confidence")
    state.final_reasoning = decision_result.get("reasoning")
    print(f"✓ Decision: {state.final_decision.upper()}")
    
    # === STEP 6: EXECUTE DECISION ===
    print("\n[Step 6/6] Executing decision...")
    
    if state.final_decision == "promote":
        # Get version number
        if os.path.exists('models/version_registry.json'):
            with open('models/version_registry.json') as f:
                registry = json.load(f)
                state.version_number = registry.get('current_version', 0) + 1
        else:
            state.version_number = 1
        
        execute_tool("promote_model", version_number=state.version_number)
        
        # Update registry
        if os.path.exists('models/version_registry.json'):
            with open('models/version_registry.json') as f:
                registry = json.load(f)
        else:
            registry = {'current_version': 0, 'versions': []}
        
        registry['current_version'] = state.version_number
        registry['versions'].append({
            'version': state.version_number,
            'timestamp': state.timestamp,
            'current_f1': state.current_f1,
            'candidate_f1': state.candidate_f1,
            'current_recall': state.current_recall,
            'candidate_recall': state.candidate_recall,
            'feedback_rows': state.feedback_rows,
            'feedback_quality': state.feedback_quality,
            'decision': 'promote',
            'confidence': state.confidence,
            'agent_reasoning': state.final_reasoning,
            'promoted': True
        })
        
        with open('models/version_registry.json', 'w') as f:
            json.dump(registry, f, indent=2)
        
        print(f"✓ PROMOTED to v{state.version_number}")
    
    else:
        execute_tool("reject_model")
        
        if os.path.exists('models/version_registry.json'):
            with open('models/version_registry.json') as f:
                registry = json.load(f)
        else:
            registry = {'current_version': 0, 'versions': []}
        
        registry['versions'].append({
            'version': None,
            'timestamp': state.timestamp,
            'current_f1': state.current_f1,
            'candidate_f1': state.candidate_f1,
            'current_recall': state.current_recall,
            'candidate_recall': state.candidate_recall,
            'feedback_rows': state.feedback_rows,
            'feedback_quality': state.feedback_quality,
            'decision': 'reject',
            'confidence': state.confidence,
            'agent_reasoning': state.final_reasoning,
            'promoted': False
        })
        
        with open('models/version_registry.json', 'w') as f:
            json.dump(registry, f, indent=2)
        
        print(f"✓ REJECTED. Current model retained.")
    
    print("\n" + "="*80)
    print("WORKFLOW COMPLETE")
    print("="*80 + "\n")
    
    return state.model_dump()