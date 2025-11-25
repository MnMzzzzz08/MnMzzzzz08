import base64
from dotenv import load_dotenv
import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, PlainTextResponse
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
        # Save uploaded file
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        # Use default prompt if none provided
        if not prompt:
            prompt = (
                "You are a transcription assistant. Transcribe the audio file and identify each speaker. "
                "Format the output like a movie script with clear speaker labels:\n\n"
                "Speaker 1: [their complete dialogue]\n\n"
                "Speaker 2: [their complete dialogue]\n\n"
                "Speaker 1: [their dialogue when they speak again]\n\n"
                "Rules:\n"
                "- Each speaker gets their own line starting with 'Speaker X:' followed by their dialogue\n"
                "- Use consistent numbering for the same speaker throughout\n"
                "- Add a blank line between each speaker's turn\n"
                "- Transcribe in the original language without translation\n"
                "- Make it clean and readable like a script"
            )
        
        transcription = process_audio_with_chat(temp_path, prompt)

        # The transcription already comes with \n characters
        # Split by \n to get individual speaker lines
        lines = transcription.split('\\n')
        
        # Clean and format each line
        formatted_lines = []
        for line in lines:
            line = line.strip()
            if line:  # Only keep non-empty lines
                formatted_lines.append(line)
        
        # Join with actual newlines (double newline for spacing between speakers)
        formatted_transcription = '\n\n'.join(formatted_lines)
        
        # Return as plain text for easy PDF conversion
        return JSONResponse(
            content={
                "transcription": formatted_transcription,
                "status": "success"
            },
            media_type="application/json; charset=utf-8"
        )
    
    finally:
        # Delete temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

# New endpoint that returns plain text (perfect for PDF conversion)
@app.post("/transcribe-text", response_class=PlainTextResponse)
async def transcribe_text(file: UploadFile = File(...), prompt: str = None):
    temp_path = f"temp_{file.filename}"
    
    try:
        # Save uploaded file
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        # Use default prompt if none provided
        if not prompt:
            prompt = (
                "You are a transcription assistant. Transcribe the audio file and identify each speaker. "
                "Format the output like a movie script with clear speaker labels:\n\n"
                "Speaker 1: [their complete dialogue]\n\n"
                "Speaker 2: [their complete dialogue]\n\n"
                "Speaker 1: [their dialogue when they speak again]\n\n"
                "Rules:\n"
                "- Each speaker gets their own line starting with 'Speaker X:' followed by their dialogue\n"
                "- Use consistent numbering for the same speaker throughout\n"
                "- Add a blank line between each speaker's turn\n"
                "- Transcribe in the original language without translation\n"
                "- Make it clean and readable like a script"
            )
        
        transcription = process_audio_with_chat(temp_path, prompt)

        # The transcription already comes with \n characters
        # Split by \n to get individual speaker lines
        lines = transcription.split('\\n')
        
        # Clean and format each line
        formatted_lines = []
        for line in lines:
            line = line.strip()
            if line:  # Only keep non-empty lines
                formatted_lines.append(line)
        
        # Join with actual newlines (double newline for spacing between speakers)
        formatted_transcription = '\n\n'.join(formatted_lines)
        
        # Return as plain text with UTF-8 encoding
        return PlainTextResponse(
            content=formatted_transcription,
            media_type="text/plain; charset=utf-8"
        )
    
    finally:
        # Delete temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)