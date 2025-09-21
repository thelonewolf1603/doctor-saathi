import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

# Ollama API endpoint (default: http://localhost:11434)
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")

app = FastAPI()

# Pydantic model for /diagnose request
class DiagnoseRequest(BaseModel):
    patient_id: str
    age: int
    sex: str
    symptoms: list[str]
    past_history: str
    medications: list[str]

@app.post("/diagnose")
async def diagnose(data: DiagnoseRequest):
    try:
        prompt = f"""
        Patient ID: {data.patient_id}
        Age: {data.age}
        Sex: {data.sex}
        Symptoms: {', '.join(data.symptoms)}
        Past history: {data.past_history}
        Current medications: {', '.join(data.medications)}

        Based on this, provide possible diagnoses and recommended next steps.
        """

        payload = {
            "model": "llama3",
            "prompt": f"You are a helpful AI medical assistant.\n{prompt}",
            "stream": False  # ensures we get full response at once
        }

        response = requests.post(OLLAMA_API_URL, json=payload)

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        data = response.json()
        diagnosis = data.get("response", "")

        return {"diagnosis": diagnosis}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))