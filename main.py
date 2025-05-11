from typing import Union
import whisper
from fastapi import FastAPI, File, UploadFile
import uvicorn  

app = FastAPI()

@app.post("/")
async def transcribe_audio(file: UploadFile = File(...)):
    # Save the uploaded file
    with open("temp_audio_file.m4a", "wb") as f:
        f.write(await file.read())
    
    # Load Whisper model
    model = whisper.load_model("tiny")
    
    # Transcribe the saved audio file
    result = model.transcribe("temp_audio_file.m4a")
    print(result["text"])
    
    # Return the transcription result
    return {"transcription": result["text"]}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)