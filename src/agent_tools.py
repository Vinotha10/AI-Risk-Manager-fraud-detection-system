import pandas as pd
import lightgbm as lgb
import joblib
import json
import os
import numpy as np
from datetime import datetime
from sklearn.metrics import f1_score, recall_score, precision_score
from src.feedback_buffer import get_feedback_training_data

# Tool 1: Train candidate model
def train_candidate_model() -> dict:
    """Train candidate model on base data + feedback"""
    try:
        train_final = pd.read_pickle('data/paysim_train_features.pkl')
        feedback_df = get_feedback_training_data()
        
        if feedback_df.empty:
            return {"status": "error", "message": "No feedback available"}
        
        feature_cols_raw = ['amount', 'type', 'balance_diff_orig', 'orig_balance_wiped',
                            'dest_balance_zero_before_after', 'oldbalanceOrg', 'oldbalanceDest',
                            'amount_to_balance_ratio', 'amount_to_dest_balance_ratio']
        
        with open('models/feature_columns.json') as f:
            feature_columns = json.load(f)
        
        X_train = pd.get_dummies(train_final[feature_cols_raw].copy(), columns=['type'], drop_first=True)
        X_train = X_train.reindex(columns=feature_columns, fill_value=0)
        y_train = train_final['isFraud']
        
        def add_features(df):
            df = df.copy()
            df['balance_diff_orig'] = df['oldbalanceOrg'] - df['amount'] - df['newbalanceOrig']
            df['orig_balance_wiped'] = (df['newbalanceOrig'] == 0).astype(int)
            df['dest_balance_zero_before_after'] = ((df['oldbalanceDest'] == 0) & (df['newbalanceDest'] == 0)).astype(int)
            df['amount_to_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
            df['amount_to_dest_balance_ratio'] = df['amount'] / (df['oldbalanceDest'] + 1)
            return df
        
        feedback_df = add_features(feedback_df)
        X_feedback = pd.get_dummies(feedback_df[feature_cols_raw].copy(), columns=['type'], drop_first=True)
        X_feedback = X_feedback.reindex(columns=feature_columns, fill_value=0)
        y_feedback = feedback_df['isFraud']
        
        X_combined = pd.concat([X_train, X_feedback], ignore_index=True)
        y_combined = pd.concat([y_train, y_feedback], ignore_index=True)
        sample_weights = np.concatenate([np.ones(len(X_train)), np.full(len(X_feedback), 50)])
        
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        candidate = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            scale_pos_weight=scale_pos_weight, random_state=42,
            deterministic=True, force_row_wise=True, verbose=-1
        )
        candidate.fit(X_combined, y_combined, sample_weight=sample_weights)
        joblib.dump(candidate, 'models/candidate_model_temp.pkl')
        
        return {"status": "success", "feedback_rows": len(X_feedback), "base_rows": len(X_train)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# Tool 2: Evaluate current model
def evaluate_current_model() -> dict:
    """Evaluate current model on test set"""
    try:
        test_final = pd.read_pickle('data/paysim_test_features.pkl')
        with open('models/feature_columns.json') as f:
            feature_columns = json.load(f)
        
        feature_cols_raw = ['amount', 'type', 'balance_diff_orig', 'orig_balance_wiped',
                            'dest_balance_zero_before_after', 'oldbalanceOrg', 'oldbalanceDest',
                            'amount_to_balance_ratio', 'amount_to_dest_balance_ratio']
        
        X_test = pd.get_dummies(test_final[feature_cols_raw].copy(), columns=['type'], drop_first=True)
        X_test = X_test.reindex(columns=feature_columns, fill_value=0)
        y_test = test_final['isFraud']
        
        model = joblib.load('models/base_model.pkl')
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        
        return {
            "status": "success",
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "precision": float(precision_score(y_test, pred, zero_division=0))
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# Tool 3: Evaluate candidate model
def evaluate_candidate_model() -> dict:
    """Evaluate candidate model on test set"""
    try:
        test_final = pd.read_pickle('data/paysim_test_features.pkl')
        with open('models/feature_columns.json') as f:
            feature_columns = json.load(f)
        
        feature_cols_raw = ['amount', 'type', 'balance_diff_orig', 'orig_balance_wiped',
                            'dest_balance_zero_before_after', 'oldbalanceOrg', 'oldbalanceDest',
                            'amount_to_balance_ratio', 'amount_to_dest_balance_ratio']
        
        X_test = pd.get_dummies(test_final[feature_cols_raw].copy(), columns=['type'], drop_first=True)
        X_test = X_test.reindex(columns=feature_columns, fill_value=0)
        y_test = test_final['isFraud']
        
        model = joblib.load('models/candidate_model_temp.pkl')
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        
        return {
            "status": "success",
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "precision": float(precision_score(y_test, pred, zero_division=0))
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# Tool 4: Analyze feedback quality
def analyze_feedback_quality(feedback_rows: int) -> dict:
    """Analyze quality and diversity of feedback"""
    try:
        from src.audit_log import get_all_logs
        logs = get_all_logs()
        reviewed = [l for l in logs if l['human_reviewed'] == 1]
        
        agree = len([l for l in reviewed if l['human_feedback'] == 'agree'])
        disagree = len([l for l in reviewed if l['human_feedback'] in ['disagree_should_approve', 'disagree_should_block']])
        
        quality = "high" if feedback_rows > 50 and (agree > 0 and disagree > 0) else "medium" if feedback_rows > 10 else "low"
        diversity = "balanced" if abs(agree - disagree) < feedback_rows * 0.3 else "skewed"
        
        return {
            "status": "success",
            "feedback_rows": feedback_rows,
            "agrees": agree,
            "disagrees": disagree,
            "quality": quality,
            "diversity": diversity
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# Tool 5: Promote model
def promote_model(version_number: int) -> dict:
    """Execute promotion: backup current, save candidate as live"""
    try:
        os.makedirs('models/history', exist_ok=True)
        backup_name = f"models/history/model_v{version_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        if os.path.exists('models/base_model.pkl'):
            import shutil
            shutil.copy('models/base_model.pkl', backup_name)
        
        import shutil
        shutil.copy('models/candidate_model_temp.pkl', 'models/base_model.pkl')
        
        if os.path.exists('models/candidate_model_temp.pkl'):
            os.remove('models/candidate_model_temp.pkl')
        
        return {"status": "success", "backup": backup_name}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# Tool 6: Reject model
def reject_model() -> dict:
    """Reject candidate, keep current"""
    try:
        if os.path.exists('models/candidate_model_temp.pkl'):
            os.remove('models/candidate_model_temp.pkl')
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# Registry of available tools
TOOLS = {
    "train_candidate_model": train_candidate_model,
    "evaluate_current_model": evaluate_current_model,
    "evaluate_candidate_model": evaluate_candidate_model,
    "analyze_feedback_quality": analyze_feedback_quality,
    "promote_model": promote_model,
    "reject_model": reject_model
}

def execute_tool(tool_name: str, **kwargs) -> dict:
    """Execute a tool by name"""
    if tool_name not in TOOLS:
        return {"status": "error", "message": f"Unknown tool: {tool_name}"}
    try:
        return TOOLS[tool_name](**kwargs)
    except Exception as e:
        return {"status": "error", "message": str(e)}