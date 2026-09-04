# src/reset_db.py
import os

db_path = 'data/audit_trail.db'
if os.path.exists(db_path):
    os.remove(db_path)
    print("Database wiped.")
else:
    print("No existing database found.")

from audit_log import init_db
init_db()
print("Fresh database initialized.")