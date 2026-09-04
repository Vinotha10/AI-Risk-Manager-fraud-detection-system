import pandas as pd
import lightgbm as lgb
import joblib
import json
import shutil
import os
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score

from src.feedback_buffer import get_feedback_training_data

import os

VERSION_REGISTRY_PATH = 'models/version_registry.json'

def load_version_registry():
    if os.path.exists(VERSION_REGISTRY_PATH):
        with open(VERSION_REGISTRY_PATH) as f:
            return json.load(f)
    return {'current_version': 0, 'versions': []}

def save_version_registry(registry):
    with open(VERSION_REGISTRY_PATH, 'w') as f:
        json.dump(registry, f, indent=2)

FEATURE_COLS_RAW = ['amount', 'type', 'balance_diff_orig', 'orig_balance_wiped',
                     'dest_balance_zero_before_after', 'oldbalanceOrg', 'oldbalanceDest',
                     'amount_to_balance_ratio', 'amount_to_dest_balance_ratio']

FEEDBACK_WEIGHT = 50  # how much to trust each human feedback row vs bulk data


def add_derived_features(df):
    df = df.copy()
    df['balance_diff_orig'] = df['oldbalanceOrg'] - df['amount'] - df['newbalanceOrig']
    df['orig_balance_wiped'] = (df['newbalanceOrig'] == 0).astype(int)
    df['dest_balance_zero_before_after'] = (
        (df['oldbalanceDest'] == 0) & (df['newbalanceDest'] == 0)
    ).astype(int)
    df['amount_to_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    df['amount_to_dest_balance_ratio'] = df['amount'] / (df['oldbalanceDest'] + 1)
    return df


def build_X_y(df, feature_columns):
    X = pd.get_dummies(df[FEATURE_COLS_RAW].copy(), columns=['type'], drop_first=True)
    X = X.reindex(columns=feature_columns, fill_value=0)
    y = df['isFraud']
    return X, y


def evaluate_model(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        'precision': precision_score(y_test, pred, zero_division=0),
        'recall': recall_score(y_test, pred, zero_division=0),
        'f1': f1_score(y_test, pred, zero_division=0)
    }


def run_retraining_cycle():
    print("=== Starting guarded retraining cycle ===")

    # 1. Load base training data + current model + feature schema
    train_final = pd.read_pickle('data/paysim_train_features.pkl')
    test_final = pd.read_pickle('data/paysim_test_features.pkl')

    with open('models/feature_columns.json') as f:
        feature_columns = json.load(f)

    current_model = joblib.load('models/base_model.pkl')

    X_train_base, y_train_base = build_X_y(train_final, feature_columns)
    X_test, y_test = build_X_y(test_final, feature_columns)

    # 2. Get feedback data
    feedback_df = get_feedback_training_data()
    if feedback_df.empty:
        print("No feedback available yet. Nothing to retrain on.")
        return {'status': 'no_feedback'}

    feedback_df = add_derived_features(feedback_df)
    X_feedback, y_feedback = build_X_y(feedback_df, feature_columns)

    print(f"Base training rows: {len(X_train_base)} | Feedback rows: {len(X_feedback)}")

    # 3. Combine with sample weighting
    X_combined = pd.concat([X_train_base, X_feedback], ignore_index=True)
    y_combined = pd.concat([y_train_base, y_feedback], ignore_index=True)
    sample_weights = pd.concat([
        pd.Series([1] * len(X_train_base)),
        pd.Series([FEEDBACK_WEIGHT] * len(X_feedback))
    ], ignore_index=True)

    scale_pos_weight = (y_train_base == 0).sum() / (y_train_base == 1).sum()

    # 4. Train candidate model
    candidate_model = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        scale_pos_weight=scale_pos_weight, random_state=42,
        deterministic=True, force_row_wise=True, verbose=-1
    )
    candidate_model.fit(X_combined, y_combined, sample_weight=sample_weights)

    # 5. Validation gate — compare against current model on held-out test set
    current_scores = evaluate_model(current_model, X_test, y_test)
    candidate_scores = evaluate_model(candidate_model, X_test, y_test)

    print("\n=== Current model (held-out test) ===")
    print(current_scores)
    print("\n=== Candidate model (held-out test) ===")
    print(candidate_scores)

    # Decision rule: promote if candidate's F1 is not meaningfully worse (allow small tolerance)
    TOLERANCE = 0.01
    promote = candidate_scores['f1'] >= (current_scores['f1'] - TOLERANCE)

    result_log = {
        'timestamp': datetime.now().isoformat(),
        'feedback_rows_used': len(X_feedback),
        'current_scores': current_scores,
        'candidate_scores': candidate_scores,
        'promoted': promote
    }

    if promote:
            registry = load_version_registry()
            new_version = registry['current_version'] + 1

            # Backup current model with version number in filename
            os.makedirs('models/history', exist_ok=True)
            backup_name = f"models/history/model_v{new_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            shutil.copy('models/base_model.pkl', backup_name)  # backs up the OLD model being replaced

            joblib.dump(candidate_model, 'models/base_model.pkl')

            registry['versions'].append({
                'version': new_version,
                'timestamp': datetime.now().isoformat(),
                'feedback_rows_used': len(X_feedback),
                'scores': candidate_scores,
                'promoted': True,
                'backup_path': backup_name
            })
            registry['current_version'] = new_version
            save_version_registry(registry)

            print(f"\n✅ PROMOTED to v{new_version}. Previous model backed up to {backup_name}")
    else:
        registry = load_version_registry()
        registry['versions'].append({
            'version': None,
            'timestamp': datetime.now().isoformat(),
            'feedback_rows_used': len(X_feedback),
            'scores': candidate_scores,
            'promoted': False,
            'backup_path': None
        })
        save_version_registry(registry)
        print("\n❌ REJECTED. Candidate underperforms current model. Keeping existing model.")

    # Log the retraining event
    os.makedirs('data', exist_ok=True)
    log_path = 'data/retraining_log.jsonl'
    with open(log_path, 'a') as f:
        f.write(json.dumps(result_log) + '\n')

    return result_log

    


if __name__ == '__main__':
    run_retraining_cycle()