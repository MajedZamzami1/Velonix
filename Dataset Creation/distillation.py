import openai
import csv
import time
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

PROMPT = (
    "You are a generator of Technology document snippets from a company's documents. Generate 5 different, realistic, multi-sentence document chunks about Technology (each over 900 characters and under 1000 characters). Each chunk should be a paragraph. If a chunk is less than 500 characters, expand it with more detail. Return the 5 chunks as a numbered list."
)

def generate_chunks():
    client = openai.OpenAI(api_key=openai.api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=3000,
        temperature=0.8,
    )
    content = response.choices[0].message.content.strip()
    # Parse the 5 chunks from the numbered list
    chunks = []
    for line in content.split('\n'):
        line = line.strip()
        if line and (line[0].isdigit() and line[1] in ['.', ')']):
            # Remove the number and dot/parenthesis
            chunk = line[2:].strip()
            if chunk:
                chunks.append(chunk)
        elif chunks and line:
            # Handle multi-line paragraphs
            chunks[-1] += ' ' + line
    # Fallback: if not parsed, just split by double newlines
    if len(chunks) < 5:
        chunks = [c.strip() for c in content.split('\n\n') if c.strip()]
    return chunks[:5]

def embed_snippet(snippet):
    client = openai.OpenAI(api_key=openai.api_key)
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=snippet
    )
    return response.data[0].embedding

snippets = []
embeddings = []

target_count = 100
while len(snippets) < target_count:
    print(f"Generating chunks {len(snippets)+1} to {min(len(snippets)+5, target_count)}...")
    new_chunks = generate_chunks()
    for chunk in new_chunks:
        if len(snippets) >= target_count:
            break
        embedding = embed_snippet(chunk)
        snippets.append(chunk)
        embeddings.append(embedding)

with open('Technology_snippets.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Chunktext', 'embedding', 'Technology'])
    for snippet, embedding in zip(snippets, embeddings):
        writer.writerow([snippet, str(embedding), "Yes"])

print("Done! Saved to Technology_snippets.csv") 