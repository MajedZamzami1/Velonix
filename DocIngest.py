import argparse
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
import pandas as pd
import ast
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import psycopg2
import unicodedata
from pgvector.psycopg2 import register_vector

load_dotenv() 
api_key = os.getenv('OPENAI_API_KEY')

def ML2(file, Department):
    df = pd.read_csv(file)
    df['embedding'] = df['embedding'].apply(ast.literal_eval)
    embeddings = np.vstack(df['embedding'].values)
    labels = df[Department].apply(lambda x: 1 if x == 'Yes' else 0).values
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def ML2_predict(embedding, model):    
    new_embedding = np.array(embedding).reshape(1, -1)
    return model.predict(new_embedding)

def process_pdf_chunks(filepath):
    client = OpenAI(api_key=api_key)
    chunks_info = []

    print(f"\n[INFO] Extracting text from {filepath}...")
    document_loader = PyMuPDFLoader(filepath)
    documents = document_loader.load()
    text = "\n".join(doc.page_content for doc in documents)

    toc_patterns = [
        r'Table of Contents.*?(?=\n\n)',
        r'^\s*\d+\s*\.\s*.*?(?=\n\n)',
        r'^\s*\d+\.\d+\s*.*?(?=\n\n)',
        r'^\s*[A-Za-z]\.\s*.*?(?=\n\n)',
    ]
    for pattern in toc_patterns:
        text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)

    text = re.sub(r'\n', ' ', text) 
    text = unicodedata.normalize("NFKC", text) 
    text = re.sub(r'[^\n\x20-\x7E\u2013\u2014]', '', text)

    rec_text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", ".", "\uff0e", "\u3002"],
        chunk_size=1000,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False
    )
    chunksContent = rec_text_splitter.split_text(text)

    cleaned_chunks = []
    for i, chunk in enumerate(chunksContent):
        chunk = chunk.strip()
        if not chunk.endswith(('.', "\uff0e", "\u3002", '!', '?')):
            if i < len(chunksContent) - 1 and chunksContent[i + 1].strip():
                chunk += ' ' + chunksContent[i + 1].split('.')[0] + '.'
                chunksContent[i + 1] = '.'.join(chunksContent[i + 1].split('.')[1:]).strip()
        if chunk.strip():
            cleaned_chunks.append(chunk)

    chunksContent = cleaned_chunks

    chunks_with_metadata = []
    model1 = ML2('Finance2.csv', 'Finance')
    model2 = ML2('IT2.csv', 'IT')
    model3 = ML2('HR2.csv', 'HR')

    for i, chunk_content in enumerate(chunksContent, 1):  
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small", 
                input=chunk_content
            )
            embedding = response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding for chunk {i}: {e}")
            embedding = None

        is_finance = bool(ML2_predict(embedding, model1)[0] == 1)
        is_it = bool(ML2_predict(embedding, model2)[0] == 1)
        is_hr = bool(ML2_predict(embedding, model3)[0] == 1)

        chunk_metadata = {
            "chunk_content": chunk_content,
            "embedding": embedding,
            "is_finance": is_finance,
            "is_it": is_it,
            "is_hr": is_hr
        }
        chunks_with_metadata.append(chunk_metadata)
        
        # Add chunk info for frontend
        chunk_info = {
            "chunk_number": i,
            "content": chunk_content[:1000] + "..." if len(chunk_content) > 100 else chunk_content,
            "tags": {
                "finance": is_finance,
                "it": is_it,
                "hr": is_hr
            }
        }
        chunks_info.append(chunk_info)

        if i % 5 == 0:  
            print(f"Processed {i} chunks")

    print("\n[INFO] Chunk Tagging Summary:")
    print("-" * 80)
    for label, key in [("Finance", "is_finance"), ("IT", "is_it"), ("HR", "is_hr")]:
        count = sum(1 for c in chunks_with_metadata if c[key])
        print(f"{label}-tagged chunks: {count}")
        print(f"Percentage of {label} chunks: {(count/len(chunks_with_metadata))*100:.2f}%")

    return chunks_with_metadata, chunks_info

def save_chunks_to_db(chunks_with_metadata):
    conn = psycopg2.connect(
        host="localhost",
        dbname="velonix_db",
        user="postgres",
        password="nono4352"
    )
    conn.set_client_encoding('UTF8')
    register_vector(conn)
    cur = conn.cursor()

    for chunk in chunks_with_metadata:
        try:
            cur.execute("""
            INSERT INTO document_chunks 
            (chunk_content, embedding, is_finance, is_it, is_hr)
            VALUES (%s, %s, %s, %s, %s)
            """, (
                chunk["chunk_content"],
                chunk["embedding"],
                chunk["is_finance"],
                chunk["is_it"],
                chunk["is_hr"]
            ))
        except psycopg2.Error as e:
            conn.rollback()
            print(f"Skipping chunk due to error: {e}")
            continue

    conn.commit()
    cur.close()
    conn.close()

def process_pdf_file(filepath):
    chunks_with_metadata, chunks_info = process_pdf_chunks(filepath)
    save_chunks_to_db(chunks_with_metadata)
    return {
        "total_chunks": len(chunks_with_metadata),
        "chunks_info": chunks_info
    }

def process_pdf_chunks_stream(filepath):
    client = OpenAI(api_key=api_key)
    print(f"\n[INFO] Extracting text from {filepath}...")
    document_loader = PyMuPDFLoader(filepath)
    documents = document_loader.load()
    text = "\n".join(doc.page_content for doc in documents)

    toc_patterns = [
        r'Table of Contents.*?(?=\n\n)',
        r'^\s*\d+\s*\.\s*.*?(?=\n\n)',
        r'^\s*\d+\.\d+\s*.*?(?=\n\n)',
        r'^\s*[A-Za-z]\.\s*.*?(?=\n\n)',
    ]
    for pattern in toc_patterns:
        text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)

    text = re.sub(r'\n', ' ', text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'[^\n\x20-\x7E\u2013\u2014]', '', text)

    rec_text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", ".", "\uff0e", "\u3002"],
        chunk_size=1000,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False
    )
    chunksContent = rec_text_splitter.split_text(text)

    cleaned_chunks = []
    for i, chunk in enumerate(chunksContent):
        chunk = chunk.strip()
        if not chunk.endswith(('.', "\uff0e", "\u3002", '!', '?')):
            if i < len(chunksContent) - 1 and chunksContent[i + 1].strip():
                chunk += ' ' + chunksContent[i + 1].split('.')[0] + '.'
                chunksContent[i + 1] = '.'.join(chunksContent[i + 1].split('.')[1:]).strip()
        if chunk.strip():
            cleaned_chunks.append(chunk)
    chunksContent = cleaned_chunks

    model1 = ML2('Finance2.csv', 'Finance')
    model2 = ML2('IT2.csv', 'IT')
    model3 = ML2('HR2.csv', 'HR')

    for i, chunk_content in enumerate(chunksContent, 1):
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=chunk_content
            )
            embedding = response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding for chunk {i}: {e}")
            embedding = None

        is_finance = bool(ML2_predict(embedding, model1)[0] == 1)
        is_it = bool(ML2_predict(embedding, model2)[0] == 1)
        is_hr = bool(ML2_predict(embedding, model3)[0] == 1)

        chunk_info = {
            "chunk_number": i,
            "content": chunk_content,
            "embedding": embedding,
            "tags": {
                "finance": is_finance,
                "it": is_it,
                "hr": is_hr
            }
        }
        yield chunk_info

        if i % 5 == 0:
            print(f"Processed {i} chunks")

def main():
    parser = argparse.ArgumentParser(description="Velonix Document Ingestion")
    parser.add_argument('--filename', required=True, help='Path to the PDF file')
    args = parser.parse_args()
    process_pdf_file(args.filename)

if __name__ == "__main__":
    main()