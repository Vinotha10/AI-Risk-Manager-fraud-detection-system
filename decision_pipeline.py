import json
from src.explain import explain_transaction
from src.audit_log import log_decision, init_db

with open('models/thresholds.json') as f:
    THRESHOLDS = json.load(f)

def make_decision(raw_transaction: dict):
    """
    Full pipeline: score -> rule check -> tier decision -> explanation -> audit log.
    Returns the decision result dict.
    """
    # Deterministic rule overrides the model decision
    if raw_transaction.get('amount', None) == 0:
        result = explain_transaction(raw_transaction)  # still explain for consistency
        decision = 'block'
        decision_reason = 'rule_zero_amount'
    else:
        result = explain_transaction(raw_transaction)
        score = result['risk_score']

        if score >= THRESHOLDS['block_threshold']:
            decision = 'block'
            decision_reason = 'model_threshold'
        else:
            decision = 'approve'
            decision_reason = 'model_threshold'

    # Log everything (approve decisions get logged too, but without SHAP detail needed for review)
    log_id = log_decision(
        transaction_data=raw_transaction,
        risk_score=result['risk_score'],
        decision=decision,
        decision_reason=decision_reason,
        explanation=result['explanation'],
        top_features=result['top_features']
    )

    return {
        'log_id': log_id,
        'decision': decision,
        'decision_reason': decision_reason,
        'risk_score': result['risk_score'],
        'explanation': result['explanation']
    }

if __name__ == '__main__':
    init_db()

    # Test with a few sample transactions
    samples = [
        {'amount': 181500.0, 'type': 'TRANSFER', 'oldbalanceOrg': 181500.0,
         'newbalanceOrig': 0.0, 'oldbalanceDest': 0.0, 'newbalanceDest': 0.0},
        {'amount': 500.0, 'type': 'CASH_OUT', 'oldbalanceOrg': 50000.0,
         'newbalanceOrig': 49500.0, 'oldbalanceDest': 100000.0, 'newbalanceDest': 100500.0},
        {'amount': 0.0, 'type': 'TRANSFER', 'oldbalanceOrg': 0.0,
         'newbalanceOrig': 0.0, 'oldbalanceDest': 5000.0, 'newbalanceDest': 5000.0},
    ]

    for i, txn in enumerate(samples):
        result = make_decision(txn)
        print(f"\nTransaction {i+1}: {txn['amount']} ({txn['type']})")
        print(f"  Decision: {result['decision']} (reason: {result['decision_reason']})")
        print(f"  Risk score: {result['risk_score']:.4f}")
        print(f"  Explanation: {result['explanation']}")

    from src.audit_log import get_all_logs

    logs = get_all_logs()
    for log in logs:
        print(log['id'], log['decision'], log['decision_reason'], log['risk_score'])