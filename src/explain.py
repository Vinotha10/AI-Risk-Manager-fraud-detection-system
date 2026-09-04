import pandas as pd
import numpy as np
import joblib
import json
import shap
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='shap')
# ---- Load model and feature list ----
model = joblib.load('models/base_model.pkl')
with open('models/feature_columns.json') as f:
    feature_columns = json.load(f)

# ---- Human-readable feature descriptions (for turning SHAP output into sentences) ----
FEATURE_DESCRIPTIONS = {
    'amount': "transaction amount",
    'balance_diff_orig': "mismatch in sender's balance change",
    'orig_balance_wiped': "sender's account was fully drained",
    'dest_balance_zero_before_after': "destination account had zero balance before and after (possible mule account)",
    'oldbalanceOrg': "sender's balance before the transaction",
    'oldbalanceDest': "destination's balance before the transaction",
    'amount_to_balance_ratio': "transaction amount relative to sender's balance",
    'amount_to_dest_balance_ratio': "transaction amount relative to destination's prior balance",
    'type_TRANSFER': "transaction type is TRANSFER"
}

# ---- Build the SHAP explainer once (reused across calls) ----
explainer = shap.TreeExplainer(model)

def prepare_features(raw_row: dict):
    """
    Takes a raw transaction dict (with original PaySim-style fields)
    and returns a properly formatted, aligned DataFrame ready for the model.
    """
    df = pd.DataFrame([raw_row])

    df['balance_diff_orig'] = df['oldbalanceOrg'] - df['amount'] - df['newbalanceOrig']
    df['orig_balance_wiped'] = (df['newbalanceOrig'] == 0).astype(int)
    df['dest_balance_zero_before_after'] = (
        (df['oldbalanceDest'] == 0) & (df['newbalanceDest'] == 0)
    ).astype(int)
    df['amount_to_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    df['amount_to_dest_balance_ratio'] = df['amount'] / (df['oldbalanceDest'] + 1)

    feature_cols_raw = ['amount', 'type', 'balance_diff_orig', 'orig_balance_wiped',
                         'dest_balance_zero_before_after', 'oldbalanceOrg', 'oldbalanceDest',
                         'amount_to_balance_ratio', 'amount_to_dest_balance_ratio']

    X = pd.get_dummies(df[feature_cols_raw].copy(), columns=['type'], drop_first=True)
    X = X.reindex(columns=feature_columns, fill_value=0)
    return X

def explain_transaction(raw_row: dict, top_k=3):
    X = prepare_features(raw_row)
    proba = model.predict_proba(X)[:, 1][0]
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    contributions = list(zip(X.columns, sv))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    top_features = contributions[:top_k]

    total_abs_impact = sum(abs(v) for _, v in contributions) or 1  # avoid div by zero

    reasons = []
    for feat_name, value in top_features:
        pct = abs(value) / total_abs_impact * 100
        direction = "increased" if value > 0 else "decreased"
        description = FEATURE_DESCRIPTIONS.get(feat_name, feat_name)
        reasons.append(f"{description} ({direction} risk, {pct:.0f}% of signal)")

    explanation_text = " | ".join(reasons)

    return {
        'risk_score': float(proba),
        'top_features': [{'feature': f, 'impact': float(v)} for f, v in top_features],
        'explanation': explanation_text
    }

if __name__ == '__main__':
    # Quick test using a sample transaction
    sample_transaction = {
        'amount': 181500.0,
        'type': 'TRANSFER',
        'oldbalanceOrg': 181500.0,
        'newbalanceOrig': 0.0,
        'oldbalanceDest': 0.0,
        'newbalanceDest': 0.0
    }

    result = explain_transaction(sample_transaction)
    print("Risk score:", result['risk_score'])
    print("Explanation:", result['explanation'])
    print("\nTop features (raw):")
    for feat in result['top_features']:
        print(f"  {feat['feature']}: {feat['impact']:.4f}")




    import pandas as pd

    test_final = pd.read_pickle('data/paysim_test_features.pkl')
    fraud_sample = test_final[test_final['isFraud']==1].iloc[0]

    raw_row = {
        'amount': fraud_sample['amount'],
        'type': fraud_sample['type'],
        'oldbalanceOrg': fraud_sample['oldbalanceOrg'],
        'newbalanceOrig': fraud_sample['newbalanceOrig'],
        'oldbalanceDest': fraud_sample['oldbalanceDest'],
        'newbalanceDest': fraud_sample['newbalanceDest']
    }

    #from src.explain import explain_transaction
    result = explain_transaction(raw_row)
    print("Actual fraud case:")
    print("Risk score:", result['risk_score'])
    print("Explanation:", result['explanation'])