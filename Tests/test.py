import psycopg2
import numpy as np

# 1. Connect to DB
conn = psycopg2.connect(
    host="localhost",
    dbname="velonix_db",
    user="postgres",
    password="nono4352"  # <-- your real password
)
cur = conn.cursor()

# 2. Fetch one embedding
cur.execute("""
SELECT embedding
FROM document_chunks
WHERE chunk_id = 'CCS Analysis - TIE212_chunk_4'
LIMIT 1;
""")

result = cur.fetchone()
db_embedding = result[0]  # This should be the array

cur.close()
conn.close()



print(db_embedding)
