import sqlite3
import json
import os

print("=== MEDINSIGHT AI PLATFORM AUDIT ===")
db_path = "backend/insurance_platform.db"
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    tables = cur.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    table_names = [t[0] for t in tables]
    print(f"Database ({db_path}) Tables found: {len(table_names)}")
    for name in table_names:
        count = cur.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        print(f"  - {name}: {count} records")
    conn.close()
else:
    print(f"DB not found at {db_path}")

print("\nSaved Models in 'models/':")
if os.path.exists("models"):
    for f in os.listdir("models"):
        p = os.path.join("models", f)
        print(f"  - {f} ({os.path.getsize(p)} bytes)")

print("\nChampion Metadata:")
meta_path = "models/champion_metadata.json"
if os.path.exists(meta_path):
    with open(meta_path) as mf:
        data = json.load(mf)
        print(f"  Model: {data.get('algorithm')}")
        print(f"  R2: {data.get('r2_score')}")
        print(f"  RMSE: {data.get('rmse')}")
        print(f"  MAE: {data.get('mae')}")
        print(f"  Trained On: {data.get('trained_at')}")

print("\nAudit Complete.")
