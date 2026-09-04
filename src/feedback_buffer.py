import pandas as pd
import json
import sqlite3

DB_PATH = 'data/audit_trail.db'

def get_feedback_training_data():
    """
    Pulls all human-reviewed transactions and converts them into
    labeled training rows based on the feedback given.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM audit_log
        WHERE human_reviewed = 1
    ''')
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()

    records = []
    for row in rows:
        txn = json.loads(row['transaction_data'])

        # Determine the CORRECT label based on feedback
        feedback = row['human_feedback']
        original_decision = row['decision']

        if feedback == 'agree':
            # Human confirms the model's implied label was right
            correct_label = 1 if original_decision == 'block' else 0
        elif feedback == 'disagree_should_approve':
            # Model flagged it, human says it's actually legit
            correct_label = 0
        elif feedback == 'disagree_should_block':
            # Model didn't block it, human says it should have been
            correct_label = 1
        else:
            continue  # unknown feedback type, skip

        record = txn.copy()
        record['isFraud'] = correct_label
        records.append(record)

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)

if __name__ == '__main__':
    df = get_feedback_training_data()
    print(f"Feedback training examples: {len(df)}")
    if not df.empty:
        print(df[['amount', 'type', 'isFraud']])