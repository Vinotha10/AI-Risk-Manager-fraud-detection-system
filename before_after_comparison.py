# src/before_after_comparison.py
import pandas as pd
import lightgbm as lgb
import joblib
import json
from src.threshold_optimizer import grid_search_thresholds, compute_cost_for_thresholds  # reuse your existing functions

train_final = pd.read_pickle('data/paysim_train_features.pkl')
test_final = pd.read_pickle('data/paysim_test_features.pkl')

y_train = train_final['isFraud']
y_test = test_final['isFraud']
amounts = test_final['amount'].values
scale_pos_weight = (y_train==0).sum() / (y_train==1).sum()

# ---- OLD MODEL: 7 features (no ratio features) ----
old_features = ['amount', 'type', 'balance_diff_orig', 'orig_balance_wiped',
                 'dest_balance_zero_before_after', 'oldbalanceOrg', 'oldbalanceDest']

X_train_old = pd.get_dummies(train_final[old_features].copy(), columns=['type'], drop_first=True)
X_test_old = pd.get_dummies(test_final[old_features].copy(), columns=['type'], drop_first=True)
X_train_old, X_test_old = X_train_old.align(X_test_old, join='left', axis=1, fill_value=0)

old_model = lgb.LGBMClassifier(
    n_estimators=200, learning_rate=0.05, num_leaves=31,
    scale_pos_weight=scale_pos_weight, random_state=42,
    deterministic=True,
    force_row_wise=True,
    verbose=-1
)
old_model.fit(X_train_old, y_train)
old_proba = old_model.predict_proba(X_test_old)[:, 1]

old_best, _ = grid_search_thresholds(y_test.values, old_proba, amounts, n_steps=50)
print("=== OLD MODEL (7 features) — Best threshold cost ===")
print(f"${old_best['total_cost']:,.2f}")

# ---- NEW MODEL: 9 features (with ratios) ----
new_model = joblib.load('models/base_model.pkl')
with open('models/feature_columns.json') as f:
    feature_columns = json.load(f)

new_features = ['amount', 'type', 'balance_diff_orig', 'orig_balance_wiped',
                 'dest_balance_zero_before_after', 'oldbalanceOrg', 'oldbalanceDest',
                 'amount_to_balance_ratio', 'amount_to_dest_balance_ratio']
X_test_new = pd.get_dummies(test_final[new_features].copy(), columns=['type'], drop_first=True)
X_test_new = X_test_new.reindex(columns=feature_columns, fill_value=0)
new_proba = new_model.predict_proba(X_test_new)[:, 1]

new_best, _ = grid_search_thresholds(y_test.values, new_proba, amounts, n_steps=50)
print("\n=== NEW MODEL (9 features + ratios) — Best threshold cost ===")
print(f"${new_best['total_cost']:,.2f}")

print(f"\n=== IMPROVEMENT ===")
improvement = old_best['total_cost'] - new_best['total_cost']
pct = (improvement / old_best['total_cost']) * 100
print(f"Cost reduced by: ${improvement:,.2f} ({pct:.1f}%)")


from sklearn.metrics import recall_score

old_pred = (old_proba >= 0.5).astype(int)
new_pred = (new_proba >= 0.5).astype(int)

print("OLD model recall @ 0.5:", recall_score(y_test, old_pred))
print("NEW model recall @ 0.5:", recall_score(y_test, new_pred))

print("\nOLD model missed fraud amount @ 0.5:", 
      test_final.loc[(old_pred==0) & (y_test==1), 'amount'].sum())
print("NEW model missed fraud amount @ 0.5:", 
      test_final.loc[(new_pred==0) & (y_test==1), 'amount'].sum())