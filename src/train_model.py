# src/train_model.py
import pandas as pd
import lightgbm as lgb
import joblib
import json

train_final = pd.read_pickle('data/paysim_train_features.pkl')
test_final = pd.read_pickle('data/paysim_test_features.pkl')

feature_cols_raw = ['amount', 'type', 'balance_diff_orig', 'orig_balance_wiped',
                     'dest_balance_zero_before_after', 'oldbalanceOrg', 'oldbalanceDest',
                     'amount_to_balance_ratio', 'amount_to_dest_balance_ratio']

X_train = pd.get_dummies(train_final[feature_cols_raw].copy(), columns=['type'], drop_first=True)
y_train = train_final['isFraud']
X_test = pd.get_dummies(test_final[feature_cols_raw].copy(), columns=['type'], drop_first=True)
y_test = test_final['isFraud']
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = lgb.LGBMClassifier(
    n_estimators=200, learning_rate=0.05, num_leaves=31,
    scale_pos_weight=scale_pos_weight, random_state=42,
    deterministic=True, force_row_wise=True, verbose=-1
)
model.fit(X_train, y_train)

joblib.dump(model, 'models/base_model.pkl')
with open('models/feature_columns.json', 'w') as f:
    json.dump(X_train.columns.tolist(), f)

print("Model trained and saved.")
print("Features:", X_train.columns.tolist())