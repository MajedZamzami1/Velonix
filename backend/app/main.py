import sys
import os
from pathlib import Path

# Add the parent directory to Python path to import existing modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import psycopg2
from dotenv import load_dotenv
from Roles import get_db_connection, add_user, edit_user_role, delete_user
from RAG import authenticate_user, get_relevant_chunks, build_final_prompt, generate_answer, embed_prompt_from_text
from DocIngest import process_pdf_file, process_pdf_chunks, save_chunks_to_db, process_pdf_chunks_stream
from fastapi.responses import StreamingResponse
import json
import tempfile

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Velonix API",
    description="API for Velonix Knowledge Base System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserAuth(BaseModel):
    name: str

class Question(BaseModel):
    text: str
    user: UserAuth

class RoleUpdate(BaseModel):
    name: str
    finance: bool
    hr: bool
    it: bool

class NewUser(BaseModel):
    name: str
    finance: bool = False
    hr: bool = False
    it: bool = False

@app.post("/auth")
async def authenticate(user: UserAuth):
    conn = get_db_connection()
    try:
        session = authenticate_user(conn, user.name)
        if not session['name']:
            raise HTTPException(status_code=401, detail="Authentication failed")
        return session
    finally:
        conn.close()

@app.post("/ask")
async def ask_question(question: Question):
    conn = get_db_connection()
    try:
        session = authenticate_user(conn, question.user.name)
        if not session['name']:
            raise HTTPException(status_code=401, detail="Authentication failed")
        
        embedding = embed_prompt_from_text(question.text)
        chunks = get_relevant_chunks(conn, session, embedding)
        if not chunks:
            return {"answer": "No relevant information found.", "chunks": []}
        
        final_prompt = build_final_prompt(chunks, question.text)
        answer = generate_answer(final_prompt)
        # Prepare chunk info for frontend
        chunk_infos = [
            {
                "content": chunk[1],
                "is_finance": chunk[2],
                "is_hr": chunk[3],
                "is_it": chunk[4]
            }
            for chunk in chunks
        ]
        return {"answer": answer, "chunks": chunk_infos}
    finally:
        conn.close()

@app.get("/roles")
async def list_roles():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT name, finance, hr, it FROM roles ORDER BY name")
            roles = cur.fetchall()
            return [{"name": name, "finance": finance, "hr": hr, "it": it} 
                   for name, finance, hr, it in roles]
    finally:
        conn.close()

@app.post("/roles")
async def update_role(role: RoleUpdate):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE roles 
                SET finance = %s, hr = %s, it = %s 
                WHERE name = %s
                """,
                (role.finance, role.hr, role.it, role.name)
            )
            if cur.rowcount == 0:
                raise HTTPException(status_code=404, detail="User not found")
            conn.commit()
            return {"message": "Role updated successfully"}
    finally:
        conn.close()

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    import traceback
    try:
        contents = await file.read()
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(contents)
            
        # Process the file and get chunk information
        result = process_pdf_file(temp_path)
        os.remove(temp_path)
        return result
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return {"error": str(e), "traceback": tb}

@app.post("/upload-stream")
async def upload_document_stream(file: UploadFile = File(...)):
    import tempfile
    import os
    import json

    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    def stream_chunks():
        try:
            # Process and stream chunks one by one
            for chunk in process_pdf_chunks_stream(tmp_path):
                # Save chunk to database
                save_chunks_to_db([{
                    "chunk_content": chunk["content"],
                    "embedding": chunk["embedding"],
                    "is_finance": chunk["tags"]["finance"],
                    "is_it": chunk["tags"]["it"],
                    "is_hr": chunk["tags"]["hr"]
                }])
                # Send a frontend-friendly version of the chunk
                frontend_chunk = {
                    "chunk_number": chunk["chunk_number"],
                    "content": chunk["content"][:1000] + "..." if len(chunk["content"]) > 100 else chunk["content"],
                    "tags": chunk["tags"]
                }
                yield f"data: {json.dumps(frontend_chunk)}\n\n"
        finally:
            os.remove(tmp_path)

    return StreamingResponse(stream_chunks(), media_type="text/event-stream")

@app.post("/users")
async def add_user_api(user: NewUser):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT name FROM roles WHERE name = %s", (user.name,))
            if cur.fetchone():
                raise HTTPException(status_code=400, detail="User already exists")
            cur.execute(
                "INSERT INTO roles (name, finance, hr, it) VALUES (%s, %s, %s, %s)",
                (user.name, user.finance, user.hr, user.it)
            )
            conn.commit()
            return {"message": "User added successfully"}
    finally:
        conn.close()

@app.delete("/users/{name}")
async def delete_user_api(name: str):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM roles WHERE name = %s RETURNING name", (name,))
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="User not found")
            conn.commit()
            return {"message": "User deleted successfully"}
    finally:
        conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 