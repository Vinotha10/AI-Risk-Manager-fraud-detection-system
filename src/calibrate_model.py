# src/calibrate_model.py
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import joblib
import json

# Load your existing training data
train_final = pd.read_pickle('data/paysim_train_features.pkl')

feature_cols_raw = ['amount', 'type', 'balance_diff_orig', 'orig_balance_wiped',
                     'dest_balance_zero_before_after', 'oldbalanceOrg', 'oldbalanceDest',
                     'amount_to_balance_ratio', 'amount_to_dest_balance_ratio']

X_train = pd.get_dummies(train_final[feature_cols_raw].copy(), columns=['type'], drop_first=True)
y_train = train_final['isFraud']

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

base_lgb = lgb.LGBMClassifier(
    n_estimators=200, learning_rate=0.05, num_leaves=31,
    scale_pos_weight=scale_pos_weight, random_state=42,
    deterministic=True, force_row_wise=True, verbose=-1
)

# Wrap with calibration (sigmoid = Platt scaling, works well for tree models)
calibrated_model = CalibratedClassifierCV(base_lgb, method='sigmoid', cv=3)
calibrated_model.fit(X_train, y_train)

joblib.dump(calibrated_model, 'models/base_model.pkl')
with open('models/feature_columns.json', 'w') as f:
    json.dump(X_train.columns.tolist(), f)

print("Calibrated model saved to models/base_model.pkl")
print("Feature columns:", X_train.columns.tolist())