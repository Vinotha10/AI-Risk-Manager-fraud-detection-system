import pandas as pd
from decision_pipeline import make_decision

test_final = pd.read_pickle('data/paysim_test_features.pkl')

# Take a manageable sample -- mix of fraud and legit for a realistic demo dataset
sample = pd.concat([
    test_final[test_final['isFraud']==1].sample(30, random_state=1),   # 30 fraud cases
    test_final[test_final['isFraud']==0].sample(200, random_state=1)   # 200 legit cases
]).sample(frac=1, random_state=1)  # shuffle

count = 0
for _, row in sample.iterrows():
    txn = {
        'amount': row['amount'],
        'type': row['type'],
        'oldbalanceOrg': row['oldbalanceOrg'],
        'newbalanceOrig': row['newbalanceOrig'],
        'oldbalanceDest': row['oldbalanceDest'],
        'newbalanceDest': row['newbalanceDest'],
        '_true_label': int(row['isFraud'])
    }
    make_decision(txn)
    count += 1
    if count % 50 == 0:
        print(f"Processed {count}/{len(sample)}...")

print(f"\nDone. Processed {len(sample)} transactions through the live pipeline.")