# src/diagnose_blindspot.py
import pandas as pd
import joblib
import json

# ---- Load everything ----
model = joblib.load('models/base_model.pkl')
test_final = pd.read_pickle('data/paysim_test_features.pkl')

with open('models/feature_columns.json') as f:
    feature_columns = json.load(f)

# ---- Build X_test using the SAME source-of-truth as training ----
feature_cols_raw = ['amount', 'type', 'balance_diff_orig', 'orig_balance_wiped',
                     'dest_balance_zero_before_after', 'oldbalanceOrg', 'oldbalanceDest',
                     'amount_to_balance_ratio', 'amount_to_dest_balance_ratio']

X_test = pd.get_dummies(test_final[feature_cols_raw].copy(), columns=['type'], drop_first=True)
X_test = X_test.reindex(columns=feature_columns, fill_value=0)  # safety net against column mismatch

print("X_test columns:", X_test.columns.tolist())
print("X_test shape:", X_test.shape)

# ---- Predict ----
test_final['pred_proba'] = model.predict_proba(X_test)[:, 1]

# ---- Diagnostic 1: overall fraud score distribution ----
fraud_cases = test_final[test_final['isFraud'] == 1].copy()

print("\n=== Overall fraud probability distribution (NEW model) ===")
print(fraud_cases['pred_proba'].describe())

# ---- Diagnostic 2: how many fraud cases still scored very low ----
low_scored = fraud_cases[fraud_cases['pred_proba'] < 0.05]
print(f"\nFraud cases scoring below 0.05: {len(low_scored)}")
print(f"Total amount lost from those low-scored fraud cases: {low_scored['amount'].sum():,.2f}")

# ---- Diagnostic 3: specifically re-check the old blind-spot pattern ----
# (wiped origin account, but destination NOT empty before/after -- this was the exact pattern that failed before)
blind_spot_cases = fraud_cases[
    (fraud_cases['orig_balance_wiped'] == 1) & (fraud_cases['dest_balance_zero_before_after'] == 0)
]

print(f"\n=== Cases matching the OLD blind-spot pattern ===")
print(f"Count: {len(blind_spot_cases)}")
print("Their NEW score distribution:")
print(blind_spot_cases['pred_proba'].describe())

print(f"\nHow many of these blind-spot cases are STILL scoring below 0.05: "
      f"{(blind_spot_cases['pred_proba'] < 0.05).sum()} out of {len(blind_spot_cases)}")



still_missed = fraud_cases[fraud_cases['pred_proba'] < 0.05]
print(still_missed[['amount', 'oldbalanceOrg', 'oldbalanceDest', 
                      'amount_to_balance_ratio', 'amount_to_dest_balance_ratio',
                      'orig_balance_wiped', 'dest_balance_zero_before_after']].describe())
print(still_missed[['amount', 'oldbalanceOrg', 'oldbalanceDest',
                      'amount_to_balance_ratio', 'amount_to_dest_balance_ratio']].head(15))




zero_amount_fraud = fraud_cases[fraud_cases['amount'] == 0]
print(f"Total fraud cases with amount=0: {len(zero_amount_fraud)}")
print(f"Total legit cases with amount=0: {(test_final['amount']==0).sum() - len(zero_amount_fraud)}")