import os
import psycopg2
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        dbname="velonix_db",
        user="postgres",
        password="nono4352"
    )

def authenticate_user(conn):
    session = {
        'name': None,
        'finance': False,
        'hr': False,
        'it': False
    }
    
    name = input("Enter your name: ").strip()
    if not name:
        print("Error: Name cannot be empty")
        return session

    with conn.cursor() as cur:
        cur.execute(
            "SELECT finance, hr, it FROM roles WHERE name = %s",
            (name,)
        )
        user = cur.fetchone()
        
        if user:
            finance, hr, it = user
            session['name'] = name
            session['finance'] = finance
            session['hr'] = hr
            session['it'] = it
        else:
            print(f"Error: User '{name}' not found")
    
    return session

def get_user_prompt():
    prompt = input("\nEnter your question: ").strip()
    if not prompt:
        print("Error: Question cannot be empty")
        return None
    return prompt

def embed_prompt(prompt):
    client = OpenAI()
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=prompt
    )
    return response.data[0].embedding

def get_relevant_chunks(conn, session, embedding, limit=5):
    with conn.cursor() as cur:
        # Build the WHERE clause based on user's roles
        role_conditions = []
        if session['finance']:
            role_conditions.append("is_finance = TRUE")
        if session['hr']:
            role_conditions.append("is_hr = TRUE")
        if session['it']:
            role_conditions.append("is_it = TRUE")
        
        if not role_conditions:
            print("Warning: User has no roles assigned")
            return []
        print(role_conditions)    
        where_clause = " OR ".join(role_conditions)
        
        # Convert embedding to string representation for PostgreSQL
        # Keep the square brackets for vector format
        embedding_str = str(embedding)
        
        # Query to get relevant chunks
        cur.execute(
            f"""
            SELECT id, chunk_content, embedding
            FROM document_chunks
            WHERE {where_clause}
            ORDER BY embedding <#> %s::vector
            LIMIT %s;
            """,
            (embedding_str, limit)
        )
        
        chunks = cur.fetchall()
        return chunks

def build_final_prompt(chunks, user_question):
    # System instructions
    system_instruction = """You are an AI assistant helping answer questions using internal company knowledge. 
Only use the context provided. If no answer can be found, say: "I'm sorry, I could not find an answer."

"""
    
    # Build context section
    context_section = "Context:\n"
    for i, (_, content, _) in enumerate(chunks, 1):
        context_section += f"[{i}] {content.strip()}\n"
    
    # Add user question
    user_question_section = f"\nQuestion: {user_question}"
    
    # Combine all parts
    final_prompt = system_instruction + context_section + user_question_section
    
    return final_prompt

def generate_answer(prompt):
    client = OpenAI()
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating answer: {str(e)}")
        return None

def main():
    print("\n=== Velonix RAG System ===")
    
    conn = get_db_connection()
    try:
        session = authenticate_user(conn)
        if not session['name']:
            print("Authentication failed. Exiting...")
            return

        prompt = get_user_prompt()
        if prompt:
            embedding = embed_prompt(prompt)
            print("\nPrompt embedded successfully!")
            
            # Get relevant chunks
            chunks = get_relevant_chunks(conn, session, embedding)
            if chunks:
                print(f"\nFound {len(chunks)} relevant chunks")
                
                # Build and display final prompt
                final_prompt = build_final_prompt(chunks, prompt)
                print("\nFinal Prompt:")
                print("-" * 50)
                print(final_prompt)
                print("-" * 50)
                
                # Generate and display answer
                print("\nGenerating answer...")
                answer = generate_answer(final_prompt)
                if answer:
                    print("\nAnswer:")
                    print("-" * 50)
                    print(answer)
                    print("-" * 50)
            else:
                print("\nNo relevant chunks found.")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
