import argparse
import os
from datetime import datetime
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
import time
import pandas as pd
import ast
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import psycopg2
import unicodedata
from pgvector.psycopg2 import register_vector


load_dotenv() 
api_key = os.getenv('OPENAI_API_KEY')

""" def ML():
    client = OpenAI(api_key=api_key)

    # 2. Load the CSV file
    df = pd.read_csv('financeData.csv')

    # 3. Parse embeddings (convert string to list)
    df['embedding'] = df['embedding'].apply(ast.literal_eval)
    embeddings = np.vstack(df['embedding'].values)

    # 4. KMeans clustering (k=2) directly on full 1536D embeddings
    kmeans = KMeans(n_clusters=2, random_state=0)
    labels = kmeans.fit_predict(embeddings)
    df['cluster'] = labels  # Assign clusters back to dataframe
    cluster_summary = df.groupby('cluster')['Finance'].value_counts()
    return kmeans

def ML(embedding, model):
    new_embedding = np.array(embedding).reshape(1, -1)
    predicted_cluster = kmeans.predict(new_embedding)
    majority_labels = df.groupby('cluster')['Finance'].agg(lambda x: x.value_counts().idxmax())
    real_label = majority_labels[predicted_cluster[0]]
    return real_label """

def ML2(file, Department):
    client = OpenAI(api_key=api_key)

    # 2. Load the CSV file
    df = pd.read_csv(file)

    # 3. Parse embeddings (convert string to list)
    df['embedding'] = df['embedding'].apply(ast.literal_eval)
    embeddings = np.vstack(df['embedding'].values)
    labels = df[Department].apply(lambda x: 1 if x == 'Yes' else 0).values
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

    # 5. Train a Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def ML2_predict(embedding, model):    
    new_embedding = np.array(embedding).reshape(1, -1)

    new_pred = model.predict(new_embedding)
    return new_pred

def main():
    parser = argparse.ArgumentParser(description="Velonix Document Ingestion")
    parser.add_argument('--filename', required=True, help='Path to the PDF file')
    parser.add_argument('--uploader', required=True, help='Uploader username or ID')
    parser.add_argument('--sensitivity', required=True, help='Document sensitivity level (e.g., public, confidential)')
    
    args = parser.parse_args()

    client = OpenAI(api_key=api_key)

    upload_date = datetime.utcnow().isoformat()

    document_metadata = {
        "filename": os.path.basename(args.filename),
        "uploader": args.uploader,
        "sensitivity": args.sensitivity,
        "upload_date": upload_date
    }

    print(f"\n[INFO] Extracting text from {args.filename}...")
    document_loader = PyMuPDFLoader(args.filename)
    documents = document_loader.load()
    text = "\n".join(doc.page_content for doc in documents)
    text = re.sub(r'\n', ' ', text) 
    text = unicodedata.normalize("NFKC", text) 


    text = re.sub(r'[^\n\x20-\x7E\u2013\u2014]', '', text)

    rec_text_splitter = RecursiveCharacterTextSplitter(separators=[
        "\n\n",
        ".",
        "\uff0e", 
        "\u3002",  
    ],
    chunk_size = 1000, chunk_overlap = 0, length_function = len, is_separator_regex=False)
    chunksContent = rec_text_splitter.split_text(text)

    cleaned_chunks = []
    for i, chunk in enumerate(chunksContent):
        chunk = chunk.strip()
            
        if not chunk.endswith(('.', "\uff0e","\u3002", '!', '?')):
            if i < len(chunksContent) - 1 and chunksContent[i + 1].strip():
                chunk += ' ' + chunksContent[i + 1].split('.')[0] + '.'
                chunksContent[i + 1] = '.'.join(chunksContent[i + 1].split('.')[1:]).strip()
            
        if chunk.strip():
            cleaned_chunks.append(chunk)

    chunksContent = cleaned_chunks

    chunks_with_metadata = []
    model1 = ML2('financeData.csv', 'Finance')
    model2 = ML2('ITData.csv', 'IT')
    for i, chunk_content in enumerate(chunksContent, 1):  

        doc_name = os.path.splitext(document_metadata["filename"])[0]
        

        try:
            response = client.embeddings.create(
                model="text-embedding-3-small", 
                input=chunk_content
            )
            embedding = response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding for chunk {i}: {e}")
            embedding = None

        ml_result = ML2_predict(embedding, model1)
        ml_result2 = ML2_predict(embedding, model2)

        department_tags = []
        #department_tag = "Finance" if ml_result == "Yes" else None
        if ml_result[0] == 1:
            department_tags.append("Finance")
        if ml_result2[0] == 1:
            department_tags.append("IT")

        chunk_metadata = {
            "ChunkID": f"{doc_name}_chunk_{i}",
            "ChunkContent": chunk_content,
            "Embedding": embedding, 
            "DepartmentTags": department_tags,
            "DocumentName": document_metadata["filename"],
            "documentChunkNumber": i,
            "Uploader": document_metadata["uploader"],
            "UploadDate": document_metadata["upload_date"],
            "Sensitivity": document_metadata["sensitivity"]
        }
        chunks_with_metadata.append(chunk_metadata)
        
        if i % 5 == 0:  
            print(f"Processed {i} chunks")

    print("\n[INFO] Document Metadata:")
    for key, value in document_metadata.items():
        print(f"  {key}: {value}")

    print("\n[INFO] Finance-Tagged Chunks:")
    print("-" * 80)
    finance_chunks = [chunk for chunk in chunks_with_metadata if "Finance" in chunk['DepartmentTags']]
    for chunk in finance_chunks:
        print(f"\nChunk ID: {chunk['ChunkID']}")
        print(f"Content Preview: {chunk['ChunkContent']}")
        print("-" * 80)
    
    print(f"\nTotal chunks: {len(chunks_with_metadata)}")
    print(f"Finance-tagged chunks: {len(finance_chunks)}")
    print(f"Percentage of finance chunks: {(len(finance_chunks)/len(chunks_with_metadata))*100:.2f}%")

    conn = psycopg2.connect(
    host="localhost",
    dbname="velonix_db",
    user="postgres",
    password="nono4352"
    )
    conn.set_client_encoding('UTF8')
    register_vector(conn)
    cur = conn.cursor()

    # 2. Insert chunks
    for chunk in chunks_with_metadata:
        
        try:
            cur.execute("""
            INSERT INTO document_chunks 
            (chunk_id, chunk_content, embedding, department_tags, document_name, document_chunk_number, uploader, upload_date, sensitivity)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                chunk["ChunkID"],
                chunk["ChunkContent"],
                chunk["Embedding"],
                chunk["DepartmentTags"],
                chunk["DocumentName"],
                chunk["documentChunkNumber"],
                chunk["Uploader"],
                chunk["UploadDate"],
                chunk["Sensitivity"]
            ))
        except psycopg2.Error as e:
            conn.rollback()
            print(f"Skipping chunk due to error: {e}")
            continue

    # 3. Commit and close
    conn.commit()
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
