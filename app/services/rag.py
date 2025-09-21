# app/services/rag.py
import os
from typing import List
from app.services.retriever import retrieve_top_k
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def assemble_prompt(patient_info: dict, retrieved_ctx: List[dict]) -> str:
    header = (
        "You are an evidence-aware medical assistant. Use the provided snippets (with sources) "
        "to propose likely diagnoses, differential diagnoses, and recommended next steps (tests/urgent referral vs home care). "
        "Always indicate uncertainty and cite sources by id.\n\n"
    )
    patient_block = f"Patient: age={patient_info.get('age')}, sex={patient_info.get('sex')}\n"
    patient_block += "Symptoms:\n" + "\n".join(f"- {s}" for s in patient_info.get("symptoms", [])) + "\n"
    if patient_info.get("history"):
        patient_block += "History: " + patient_info["history"] + "\n"
    if patient_info.get("medications"):
        patient_block += "Medications: " + ", ".join(patient_info["medications"]) + "\n"
    ctx_block = "\nTop evidence passages:\n"
    for i, r in enumerate(retrieved_ctx):
        md = r["metadata"]
        ctx_block += f"[SRC{i}] id={md.get('id')} title={md.get('title', 'NA')} score={r['score']}\n{md.get('text')}\n\n"

    prompt = header + patient_block + ctx_block + "\nNow give:\n1) concise summary\n2) top 3 possible diagnoses (ranked)\n3) what to test or recommended next steps\n4) sources references mapping (use [SRC#] tags)\n"
    return prompt

def call_llm(prompt: str) -> str:
    # Example: call OpenAI ChatCompletion (you can replace with local LLM)
    res = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # replace with a model you have access to
        messages=[{"role":"user","content":prompt}],
        temperature=0.0,
        max_tokens=700
    )
    return res.choices[0].message.content

def diagnose_patient(symptoms: List[str], age:int=None, sex:str=None, history:str=None, medications:List[str]=[]):
    patient = {"age": age, "sex": sex, "symptoms": symptoms, "history": history, "medications": medications}
    # Build a query string for retrieval
    query = " ; ".join(symptoms) + (f" history: {history}" if history else "")
    retrieved = retrieve_top_k(query, k=6)  # list of dicts {"score":..., "metadata": {...}}
    prompt = assemble_prompt(patient, retrieved)
    llm_out = call_llm(prompt)

    # Basic parsing — you should robustly parse structured output in production
    return {
        "summary": llm_out.split("\n")[0],
        "differential": ["(see LLM output)"],
        "recommendations": ["(see LLM output)"],
        "sources": [{"score": r["score"], "metadata": r["metadata"]} for r in retrieved],
        "disclaimers": [
            "This is an AI-based suggestion for prototyping only. Not medical advice. Validate with a clinician."
        ]
    }