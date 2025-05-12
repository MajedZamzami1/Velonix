import argparse
import os
from datetime import datetime
from pdfminer.high_level import extract_text as pdf_extract_text

import re
import unicodedata

def clean_text(text):

    
    # Remove table of contents
    # Common patterns for table of contents
    toc_patterns = [
        r'Table of Contents.*?(?=\n\n)',  # Matches "Table of Contents" and everything until next blank line
        r'Contents.*?(?=\n\n)',           # Matches "Contents" and everything until next blank line
        r'^\s*\d+\s*\.\s*.*?(?=\n\n)',   # Matches numbered entries (e.g., "1. Introduction")
        r'^\s*\d+\.\d+\s*.*?(?=\n\n)',   # Matches numbered entries with subnumbers (e.g., "1.1 Introduction")
        r'^\s*[A-Za-z]\.\s*.*?(?=\n\n)', # Matches lettered entries (e.g., "A. Introduction")
    ]
    
    for pattern in toc_patterns:
        text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    text = re.sub(r'\n', ' ', text) 
    text = unicodedata.normalize("NFKC", text) 


    text = re.sub(r'[^\n\x20-\x7E\u2013\u2014]', '', text)
    return text

def extract_text(file_path):
    """
    Extract text content from a PDF file using pdfminer.six.
    """
    try:
        text = pdf_extract_text(file_path)
        return text
    except Exception as e:
        print(f"[ERROR] Failed to extract text: {e}")
        return ""

def main():
    parser = argparse.ArgumentParser(description="Velonix Document Ingestion")
    parser.add_argument('--filename', required=True, help='Path to the PDF file')
    parser.add_argument('--uploader', required=True, help='Uploader username or ID')
    parser.add_argument('--sensitivity', required=True, help='Document sensitivity level (e.g., public, confidential)')
    
    args = parser.parse_args()

    upload_date = datetime.utcnow().isoformat()

    document_metadata = {
        "filename": os.path.basename(args.filename),
        "uploader": args.uploader,
        "sensitivity": args.sensitivity,
        "upload_date": upload_date
    }

    print(f"\n[INFO] Extracting text from {args.filename}...")
    extracted_text = extract_text(args.filename)
    extracted_text = clean_text(extracted_text)

    print("\n[INFO] Document Metadata:")
    for key, value in document_metadata.items():
        print(f"  {key}: {value}")

    print("\n[INFO] Extracted Text Content:")
    print("-" * 40)
    print(extracted_text[:5000])  
    print("-" * 40)
    

if __name__ == "__main__":
    main()
