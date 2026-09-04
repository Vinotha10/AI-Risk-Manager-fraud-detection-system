import pandas as pd
import pickle
import os

print("Preparing PaySim dataset features...")

# Load the raw filtered CSV (assumed to already exist)
csv_path = 'data/paysim_filtered.csv'
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found.")
    print("Download PaySim from: https://www.kaggle.com/datasets/ealaxi/paysim1")
    print("Then run: data/download_and_filter.py")
    exit(1)

df = pd.read_csv(csv_path)

# Feature engineering (same as Day 3)
df['balance_diff_orig'] = df['oldbalanceOrg'] - df['amount'] - df['newbalanceOrig']
df['orig_balance_wiped'] = (df['newbalanceOrig'] == 0).astype(int)
df['dest_balance_zero_before_after'] = (
    (df['oldbalanceDest'] == 0) & (df['newbalanceDest'] == 0)
).astype(int)
df['amount_to_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
df['amount_to_dest_balance_ratio'] = df['amount'] / (df['oldbalanceDest'] + 1)

# Time-based split (Day 2)
split_point = df['step'].quantile(0.80)
train_final = df[df['step'] <= split_point].copy()
test_final = df[df['step'] > split_point].copy()

os.makedirs('data', exist_ok=True)
pickle.dump(train_final, open('data/paysim_train_features.pkl', 'wb'))
pickle.dump(test_final, open('data/paysim_test_features.pkl', 'wb'))

print(f"✅ Train set: {len(train_final)} rows")
print(f"✅ Test set: {len(test_final)} rows")
print("Data ready in data/paysim_*.pkl")