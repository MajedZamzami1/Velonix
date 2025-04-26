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

load_dotenv() 
api_key = os.getenv('OPENAI_API_KEY')

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

        chunk_metadata = {
            "ChunkID": f"{doc_name}_chunk_{i}",
            "ChunkContent": chunk_content,
            "Embedding": embedding, 
            "DepartmentTag": None,
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

    print("\n[INFO] Extracted Text Content:")
    print("-" * 40)
    print(text[:1000]) 
    print("-" * 40)
    print("Sample chunk with metadata:")
    print(chunks_with_metadata[1])  
    print(f"Total chunks created: {len(chunks_with_metadata)}")

if __name__ == "__main__":
    main()
