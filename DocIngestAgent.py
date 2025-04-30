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
import psycopg2
import unicodedata
from pgvector.psycopg2 import register_vector

load_dotenv() 
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

def agent(text, department):
    if department == "Finance":
        examples = """
Examples:
Text: "Additionally, pricing  sensitivity surveys will help gauge enterprise expectations and willingness to pay for a robust AI  security platform.  A critical component of this validation will be offering paid Proof-of-Concept (PoC)  deployments to prospective customers. By tracking the conversion rate from PoC to full  adoption, we can measure tangible interest and assess whether enterprises recognize the  immediate value of Velonix. Benchmarking against competitors in adjacent cybersecurity  domains, such as AI-powered data loss prevention (DLP) and cloud security solutions, will  further refine our pricing strategy. If the PoC conversion rate exceeds 50%, along with strong  pricing validation, we can confidently confirm the hypothesis that enterprises are willing to pay  for AI security solutions. 5 ." → Yes
Text: "To  validate product-market fit, we aim for a Proof-of-Concept (PoC) to a paid conversion rate of at  least 50%, demonstrating that customers see immediate value in Velonix’s AI security solutions.  Additionally, a retention rate of 70% or higher post-pilot will indicate long-term adoption and  satisfaction, ensuring that enterprises integrate Velonix as a core security component rather than  a temporary solution.  6.2 Financial Viability  For Velonix to establish a sustainable business model, financial performance must align with its  growth strategy. In Year 1, we target at least $500K in revenue, driven by a combination of SaaS  subscriptions and high-value on-premises deployments. As adoption scales, annual revenue is  projected to surpass $1M, ensuring financial stability and reinvestment in product innovation ." → Yes
Text: "If enterprises find Velonix easy to integrate with minimal impact on performance, they’ll be  more likely to adopt it. However, if there’s friction, we may need to improve developer-friendly  integrations (APIs, SDKs) or provide low-code/no-code deployment options.  4.3 Willingness to Partner: Can We Secure Strategic Alliances?  Velonix is not just a standalone product—it needs to operate within an ecosystem of cloud  providers, AI platforms, and regulatory bodies. Our hypothesis is that:  1.​ Cloud providers (Azure, STC Cloud) will see value in integrating Velonix as an added  security layer for AI workloads.  2.​ Regulatory bodies (SDAIA, SAMA, NCA) will recognize Velonix as a compliance  enabler and may even recommend it as a best practice for AI security.  3.​ AI vendors (OpenAI, Anthropic, Hugging Face) will be open to partnerships, given the  growing concerns around data leakage in AI models ." → No
Text: "Additionally, it may reduce the chances of incomplete results, as the LLM  has access to a broader dataset before filtering is applied.    However, this approach comes with significant trade-offs. Since the full dataset is accessed  before any filtering takes place, there is a higher risk of unauthorized data exposure, especially  if the filtering logic is flawed or incomplete. It also increases vulnerability to prompt injection  attacks, where malicious input attempts to bypass access controls. From a technical standpoint,  this architecture demands robust internal safeguards and strict post-retrieval sanitization  mechanisms to prevent leakage. It also introduces greater processing overhead, as the system  spends resources retrieving and then filtering potentially large volumes of data. While it provides  flexibility and completeness in data context, this model is the least secure among the three if not  implemented with strong enforcement protocols ." → No
        """
        question = f"Text: \"{text}\"\n\nAnswer Yes ONLY if the text is explicitly about financial matters (revenue, profit, accounting, banking, investments, financial statements, budgets, costs, pricing, financial analysis). If the text mentions money or costs but is not relavant for someone in the Finance department, answer No."

    elif department == "IT":
        examples = """
Examples:
Text: "In  addition, this architecture may need a custom retrieval pipeline for filtering data before query  execution.    Architecture 2: Data Control Filtering After Prompt   ​ ​ In this architecture, data control filtering occurs after the user prompt is made but before data  retrieval. The system dynamically modifies the query to exclude unauthorized data, ensuring  only permitted information is retrieved and passed to the LLM. After the user prompt is made,  before retrieval, the system modifies the query to exclude unauthorized data before passing it to  the LLM. This approach offers several advantages. It allows for more flexible permissions, as  access policies can be enforced at query time, adapting to individual user roles. It also enables  better handling of ambiguous queries, allowing the system to reframe vague or broad requests  instead of rejecting them outright ." → Yes
Text: "Furthermore, the LLM can retrieve richer context than  pre-retrieval filtering, since the query operates on the full dataset with tailored access rules. However, this design has its drawbacks. There is a higher risk of data leakage if filtering is not  applied strictly enough, making it susceptible to prompt injection attacks. Additionally, this  method incurs increased processing overhead, as the retrieval system still searches the entire  dataset before filtering, which can lead to slower query response times.      Architecture 3: Data Control Filtering After Retrieval  In this architecture, the system retrieves data from the knowledge base first, then applies  filtering based on user permissions. This design enables the retrieval of the most complete and  relevant information before access restrictions are enforced. It can be beneficial in scenarios  where data relationships are complex, and the filtering process needs full context to make  accurate decisions ." → Yes
Text: "Beyond the core team, there is a network of advisors,  collaborators and partners that can fully fill eventual  competence gap that the core team has  5  We are working very closely with the AI director of Neom.  He has been a very supportive advisor and meets with us  very frequently to give us direction on how to work and  review our progress.        Desirability: Is there a market for the product ?  Do  customers want it ?  Score  (1-5)  Arguments and Documentation  The target customer segment is clearly defined (target  segment(s) and Buyer/User Personas)  2  We have an understanding of generally who we want to  target, however, we need more analysis of possible  architectures of our solution and its security level in order to  see what sector matches the level of security and speed of  use.  The target market is clearly sized (TAM and SAM)  5  Yes, we have a clear understanding of the target market, and  it is clearly sized ." → No
Text: "To reach profitability within the first year, Velonix must secure either 10 SaaS enterprise clients,  generating $500K in revenue, or four on-premises deployments at $125K each. This financial  model ensures a clear path to break-even while allowing room for growth. Expansion into  additional markets such as education, telecom, and retail will further enhance scalability,  particularly as AI security compliance becomes a regulatory mandate across industries.  Project & Team Members: Velonix, Mohammed Alghufauili, Majed Zamzami        Technical Feasibility: Can it be Done ?, Can The Team  Execute the Project ?  Score  (1-5)  Arguments and Documentation  The Product/Service Bundle is well defined (Specs,  design, features, materials, UI, UX...)  5  The customer will receive a preemptive cybersecurity tool  that prevents data leakages caused by LLMs and RAG  systems ." → No
        """
        question = f"Text: \"{text}\"\n\nAnswer Yes ONLY if the text is explicitly about IT/technical matters (software, hardware, cybersecurity, databases, networks, programming, IT infrastructure, technical support, system administration). If the text mentions technology but is not relavant for someone in the IT department, answer No."

    elif department == "HR":
        examples = """
Examples:
Text: "Inclusivity extends beyond meeting diversity  quotas and is a fundamental part of our core  values, aligned with Global Goal 5 on Gender  Equality. We actively seek and embrace  individuals from various backgrounds, ensuring  our candidate pool is as diverse as the society  we serve. By fostering an inclusive culture, we not only  empower our employees to realise their  full potential but also foster a dynamic and  collaborative atmosphere where everyone feels  valued and heard. Our commitment to diversity and inclusion  is reflected in our Code of Conduct, which  expects everyone to be collaborative,  supportive, and respectful, promoting a culture  that encourages participation and contribution  from all employees. In addition to our Code  of Conduct, our Human Resources Employee  Handbook outlines a Zero-Tolerance policy for  workplace discrimination or harassment. Our dedicated DEI Working Group takes the  lead in driving and promoting DEI initiatives  across all our offices ." → Yes
Text: "Diversity Diversity helps us better understand  our clients and enables us to embrace  the array of talents within each  individual, helping them to add value in  the workforce. We acknowledge the benefit of  different perspectives in decision- making and the valuable ideas and  contributions to the business diverse  individuals bring. Inclusion  We are committed to developing,  respecting, and promoting an inclusive  workplace, as diversity can play an  important role in addressing important  business issues. We strive to enable contributors to feel  their contribution, ideas, and values  matter, irrespective of their background,  identity, or circumstances ." → Yes
Text: "Buyer preferences, gain and pain points and unmet  needs are well understood and documented (a market  already exists, there is available market research that  indicate demand)  5  Buyers prevent data leakages and make sure they are  compliant with regulations.  The Pricing of the Product/Service Bundle is clearly  defined  3  Yes, it is clearly defined.  There are strong indications (market research,  analogies) that there is strong demand for our specific  product/service USPs  5  Yes, many companies around the world are searching for  solutions for these issues. In addition, we have received personal indications of willingness of the purchase of this  product from NEOM and some government agencies.         Viability (Can It Sustain & Scale?)   Score  1-5  Arguments and Documentation  The revenue model and bundle pricing is clearly  defined   3  We understand we want to make a yearly licensing revenue  model, however, we need more understanding of the details  of this model ." → No
Text: "The yearly volume break even point is a small  percentage (less than 5%) of the Serviceable Available  Market (SAM)  3  Yes, as shown before  The yearly volume break even point is coherent / is  supported by the company's resource base and cost  structure (realistic CAC, manufacturing capacity etc)  1  Yes, the yearly volume break-even point is coherent / is  supported by the company resource base and cost structure  (realistic CAC, manufacturing capacity, etc)    The business has a realistic plan for funding the early  stages of development  3  Yes we have a plan for funding the early stages of  development (found below)        Sustainability check   Score  1-5  Arguments and Documentation The start-up will be able to sustain its competitive  advantage because of network externalities  3  Yes, we have very good connections with Neom and  KAUST, which should give us a great starting network to  start with ." → No
        """
        question = f"Text: \"{text}\"\n\nAnswer Yes ONLY if the text is explicitly about HR matters (hiring, training, employee benefits, diversity, workplace policies, employee relations, performance management, HR compliance). If the text mentions people or employees but is NOT relevant to someone in HR answer No."

    else:
        raise ValueError("Unknown department")

    full_prompt = f"{examples}\n\n{question}"
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": full_prompt}]
    )

    answer = response.choices[0].message.content.strip()
    return answer.lower() == "yes"

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
    document_loader = PyMuPDFLoader(args.filename)
    documents = document_loader.load()
    text = "\n".join(doc.page_content for doc in documents)
    
    toc_patterns = [
        r'Table of Contents*?(?=\n\n)',  # Matches "Table of Contents" and everything until next blank line
        r'^\s*\d+\s*\.\s*.*?(?=\n\n)',   # Matches numbered entries (e.g., "1. Introduction")
        r'^\s*\d+\.\d+\s*.*?(?=\n\n)',   # Matches numbered entries with subnumbers (e.g., "1.1 Introduction")
        r'^\s*[A-Za-z]\.\s*.*?(?=\n\n)', # Matches lettered entries (e.g., "A. Introduction")
    ]
    
    for pattern in toc_patterns:
        text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
    
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

        # Use the agent to determine department tags
        department_tags = []
        if agent(chunk_content, "Finance"):
            department_tags.append("Finance")
        if agent(chunk_content, "IT"):
            department_tags.append("IT")
        if agent(chunk_content, "HR"):
            department_tags.append("HR")

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

    print("\n[INFO] Department-Tagged Chunks:")
    print("-" * 80)
    finance_chunks = [chunk for chunk in chunks_with_metadata if "Finance" in chunk['DepartmentTags']]
    IT_chunks = [chunk for chunk in chunks_with_metadata if "IT" in chunk['DepartmentTags']]
    HR_chunks = [chunk for chunk in chunks_with_metadata if "HR" in chunk['DepartmentTags']]
    
    for chunk in chunks_with_metadata:
        print(f"\nChunk ID: {chunk['ChunkID']}")
        print(f"Content Preview: {chunk['ChunkContent'][:100]}...")
        print(f"Tags: {chunk['DepartmentTags']}")
        print("-" * 80)
    
    print(f"\nTotal chunks: {len(chunks_with_metadata)}")
    print(f"Finance-tagged chunks: {len(finance_chunks)}")
    print(f"Percentage of finance chunks: {(len(finance_chunks)/len(chunks_with_metadata))*100:.2f}%")
    print(f"IT-tagged chunks: {len(IT_chunks)}")
    print(f"Percentage of IT chunks: {(len(IT_chunks)/len(chunks_with_metadata))*100:.2f}%")
    print(f"HR-tagged chunks: {len(HR_chunks)}")
    print(f"Percentage of HR chunks: {(len(HR_chunks)/len(chunks_with_metadata))*100:.2f}%")

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

    conn.commit()
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
