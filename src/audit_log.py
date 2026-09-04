import sqlite3
import json
from datetime import datetime

DB_PATH = 'data/audit_trail.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            transaction_data TEXT NOT NULL,
            risk_score REAL NOT NULL,
            decision TEXT NOT NULL,
            decision_reason TEXT NOT NULL,
            explanation TEXT NOT NULL,
            top_features TEXT NOT NULL,
            human_reviewed INTEGER DEFAULT 0,
            human_feedback TEXT DEFAULT NULL,
            human_feedback_timestamp TEXT DEFAULT NULL
        )
    ''')
    conn.commit()
    conn.close()

def log_decision(transaction_data: dict, risk_score: float, decision: str,
                  decision_reason: str, explanation: str, top_features: list):
    """
    Logs a single transaction's decision to the audit trail.

    decision: 'approve' | 'hold' | 'block'
    decision_reason: 'model_threshold' | 'rule_zero_amount' | etc.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO audit_log
        (timestamp, transaction_data, risk_score, decision, decision_reason, explanation, top_features)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        json.dumps(transaction_data),
        risk_score,
        decision,
        decision_reason,
        explanation,
        json.dumps(top_features)
    ))
    conn.commit()
    log_id = cursor.lastrowid
    conn.close()
    return log_id

def get_pending_review(limit=50):
    """Fetch Block decisions that haven't been reviewed by a human yet."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM audit_log
        WHERE decision IN ('hold', 'block') AND human_reviewed = 0
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (limit,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def submit_feedback(log_id: int, feedback: str):
    """
    feedback: 'agree' | 'disagree_should_approve' | 'disagree_should_block' etc.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE audit_log
        SET human_reviewed = 1, human_feedback = ?, human_feedback_timestamp = ?
        WHERE id = ?
    ''', (feedback, datetime.now().isoformat(), log_id))
    conn.commit()
    conn.close()

def get_all_logs(limit=100):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?', (limit,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def get_feedback_stats():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT decision, human_feedback, COUNT(*) as count
        FROM audit_log
        WHERE human_reviewed = 1
        GROUP BY decision, human_feedback
    ''')
    rows = cursor.fetchall()
    conn.close()
    return rows

if __name__ == '__main__':
    init_db()
    print("Audit trail database initialized at", DB_PATH)