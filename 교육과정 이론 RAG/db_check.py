import sqlite3

conn = sqlite3.connect("chat_history.db")
cursor = conn.cursor()

cursor.execute("SELECT id, role, message FROM chat ORDER BY id")
rows = cursor.fetchall()

for row in rows:
    print(f"[{row[0]}] {row[1]}: {row[2]}")

conn.close()