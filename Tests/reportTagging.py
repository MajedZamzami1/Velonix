import argparse
import psycopg2
import pandas as pd
import numpy as np
from pgvector.psycopg2 import register_vector

def main():
    parser = argparse.ArgumentParser(description="Tag document chunks with departments")
    parser.add_argument('--chunkID', required=True, help='Chunk ID to tag')
    parser.add_argument('--department', required=True, help='Department to tag (Finance, IT, or HR)')
    args = parser.parse_args()

    department_files = {
        'HR': 'HRdata.csv',
        'IT': 'ITdata.csv',
        'Finance': 'financeData.csv'
    }
    
    if args.department not in department_files:
        print(f"Error: Department must be one of {list(department_files.keys())}")
        return

    try:
        conn = psycopg2.connect(
            host="localhost",
            dbname="velonix_db",
            user="postgres",
            password="nono4352"
        )
        conn.set_client_encoding('UTF8')
        register_vector(conn)
        cur = conn.cursor()

        cur.execute("""
            SELECT chunk_content, embedding 
            FROM document_chunks 
            WHERE chunk_id = %s
        """, (args.chunkID,))
        
        result = cur.fetchone()
        if not result:
            print(f"Error: Chunk ID {args.chunkID} not found in database")
            return

        chunk_content, embedding = result

        csv_file = department_files[args.department]
        
        try:
            df = pd.read_csv(csv_file)
        except FileNotFoundError:
            print(f"Error: CSV file {csv_file} not found")
            return

        clean_embedding = [float(x) for x in embedding]

        new_row = {
            'Chunktext': chunk_content, 
            'embedding': clean_embedding,  
            args.department: 'No' 
        }

        new_df = pd.DataFrame([new_row], columns=df.columns)
        
        df = pd.concat([df, new_df], axis=0, ignore_index=True, verify_integrity=True)

        df.to_csv(csv_file, index=False)
        print(f"Successfully added chunk {args.chunkID} to {csv_file}")

    except psycopg2.Error as e:
        print(f"Database error: {e}")
    finally:
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()
