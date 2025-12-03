# main.py
import os
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from dotenv import load_dotenv
from huggingface_hub import InferenceClient, InferenceTimeoutError
import time


# 1. Load Environment Variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
HF_TOKEN = os.getenv("HF_TOKEN")


if not MONGO_URI or not HF_TOKEN:
    raise ValueError("Missing environment variables. Check your .env file.")


# 2. Initialize App & Database
app = FastAPI(
    title="AI Summarizer Service",
    description="A REST API that summarizes text using Hugging Face and saves history to MongoDB.",
    version="1.0"
)


# Connect to MongoDB
try:
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client['ai_project_db']
    collection = db['summaries']
    mongo_client.admin.command('ping')
    print("✅ Successfully connected to MongoDB")
except Exception as e:
    print(f"❌ MongoDB Connection Error: {e}")


# 3. AI Configuration - Using smaller, faster model
hf_client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN,
    timeout=60  # 60 seconds timeout
)


# 4. Data Models
class SummaryRequest(BaseModel):
    text: str


# --- ENDPOINTS ---

@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "running", "message": "Go to /docs for Swagger UI"}


@app.post("/summarize")
def summarize_text(request: SummaryRequest):
    """
    Receives text, sends it to AI for summarization, and saves to DB.
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Limit text length
    max_length = 1024
    if len(request.text) > max_length:
        raise HTTPException(
            status_code=400, 
            detail=f"Text too long. Maximum {max_length} characters allowed."
        )

    # Try with retry logic - using faster DistilBART model
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = hf_client.summarization(
                request.text,
                model="sshleifer/distilbart-cnn-12-6"  # Faster model!
            )
            summary_result = result.summary_text
            break
            
        except InferenceTimeoutError:
            if attempt < max_retries - 1:
                wait_time = 10 * (attempt + 1)
                print(f"Timeout, retrying in {wait_time}s... (attempt {attempt + 2}/{max_retries})")
                time.sleep(wait_time)
                continue
            else:
                raise HTTPException(
                    status_code=504, 
                    detail="Model timeout. Please try again."
                )
        except Exception as e:
            error_msg = str(e)
            if "timed out" in error_msg.lower():
                if attempt < max_retries - 1:
                    wait_time = 10 * (attempt + 1)
                    print(f"Read timeout, retrying in {wait_time}s... (attempt {attempt + 2}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    raise HTTPException(
                        status_code=504, 
                        detail="Connection timeout. The model is warming up. Please try again in a moment."
                    )
            else:
                raise HTTPException(status_code=500, detail=f"AI Error: {error_msg}")

    # Save to MongoDB
    record = {
        "original_text": request.text,
        "summary_text": summary_result,
        "created_at": datetime.utcnow()
    }
    insert_result = collection.insert_one(record)

    return {
        "id": str(insert_result.inserted_id),
        "summary": summary_result,
        "timestamp": record["created_at"]
    }


@app.get("/history")
def get_history(limit: int = 10):
    """Retrieves the last 10 summaries from MongoDB."""
    cursor = collection.find({}, {"_id": 0}).sort("created_at", -1).limit(limit)
    history = list(cursor)
    return {"count": len(history), "data": history}
