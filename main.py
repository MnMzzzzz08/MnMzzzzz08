import base64
from dotenv import load_dotenv
import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()


print(os.getenv("OPENAI_API_KEY"))

# Initialize ChatOpenAI model
llm = ChatOpenAI(
    model="gpt-4o-audio-preview",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7
)

# Initialize FastAPI app
app = FastAPI(title="Audio Transcription API")

def encode_audio_to_base64(audio_file_path: str) -> str:
    """Convert audio file to base64 string for API transmission."""
    with open(audio_file_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")

def get_audio_media_type(file_path: str) -> str:
    """Determine media type from file extension."""
    extension = Path(file_path).suffix.lower()
    media_types = {
        ".mp3": "audio/mp3",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg"
    }
    return media_types.get(extension, "audio/mpeg")

def process_audio_with_chat(audio_file_path: str, user_question: str) -> str:
    """Process audio file with ChatOpenAI and return the response."""
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
    
    audio_data = encode_audio_to_base64(audio_file_path)
    media_type = get_audio_media_type(audio_file_path)
    
    message = HumanMessage(
        content=[
            {"type": "text", "text": user_question},
            {"type": "audio", "base64": audio_data, "mime_type": media_type}
        ]
    )
    
    response = llm.invoke([message])
    return response.content

# Root endpoint for health check
@app.get("/")
async def root():
    return {"status": "ok", "message": "Audio Transcription API is running"}

# FastAPI endpoint
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), prompt: str = None):
    temp_path = f"temp_{file.filename}"
    
    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        # Use default prompt if none provided
        if not prompt:
            prompt = (
                "You are a transcription assistant. You are given an audio file. "
                "Please transcribe the speech and identify speakers clearly. "
                "Return the transcription in the original language, "
                "with clean line breaks and readable formatting. "
                "Do NOT translate the text."
            )
        
        transcription = process_audio_with_chat(temp_path, prompt)

        # Clean formatting: replace escaped newlines/tabs with real characters
        clean_transcription = (
            transcription.replace("\\n", "\n")
                         .replace("\\t", "\t")
                         .strip()
        )
        
        return {
            "transcription": clean_transcription,
            "status": "success"
        }
    
    finally:
        # Delete temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
