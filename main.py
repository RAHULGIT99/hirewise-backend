# backend/main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import tempfile
import os
from dotenv import load_dotenv
from typing import Optional
import json

# text extraction libs
import PyPDF2
import docx2txt

# google gen ai
from google import genai

app = FastAPI(title="ATS Evaluator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY environment variable not set.")
genai_client = genai.Client(api_key=GOOGLE_API_KEY)

def extract_text_from_pdf(file_path: str) -> str:
    text = []
    with open(file_path, "rb") as fh:
        reader = PyPDF2.PdfReader(fh)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text.append(page_text)
    return "\n".join(text).strip()

def extract_text_from_docx(file_path: str) -> str:
    return docx2txt.process(file_path) or ""

def extract_text_from_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def extract_text(upload: UploadFile) -> str:
    suffix = (upload.filename or "").lower().split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix="." + suffix) as tmp:
        tmp.write(upload.file.read())
        tmp_path = tmp.name

    try:
        if suffix == "pdf":
            return extract_text_from_pdf(tmp_path)
        elif suffix in ("docx", "doc"):
            return extract_text_from_docx(tmp_path)
        elif suffix in ("txt", "md"):
            return extract_text_from_txt(tmp_path)
        else:
            return ""  # unsupported format - return empty; Gemini will still be asked to reason
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

class EvalResponse(BaseModel):
    ats_score: Optional[int]
    matching_keywords: Optional[list]
    missing_keywords: Optional[list]
    strengths: Optional[list]
    weaknesses: Optional[list]
    suggestions: Optional[list]
    summary: Optional[str]

@app.post("/evaluate", response_model=EvalResponse)
async def evaluate_resume(
    role: Optional[str] = Form(None),
    job_description: Optional[str] = Form(None),
    resume: UploadFile = File(...),
):
    text = extract_text(resume)
    if not text:
        # allow evaluating even if extraction fails; but inform user
        text = "(Resume text could not be reliably extracted; proceed with available content.)"

    # Build a single, structured prompt asking Gemini for JSON output.
    prompt = f"""
You are an expert hiring assistant and ATS system. Evaluate the candidate resume below against the job information provided.
Return ONLY valid JSON (no extra commentary) with the following fields:
- ats_score: integer 0-100 (calculated for candidate->job fit, consider keywords, role match, experience, formatting)
- matching_keywords: array of key keywords that appear in the resume and match the JD/role
- missing_keywords: array of important keywords from JD/role that are NOT present in resume
- strengths: array of concise strengths observed in resume
- weaknesses: array of concise weaknesses or risks for ATS rejection
- suggestions: array of concrete, prioritized suggestions to improve the resume for this JD (formatting, keywords, bullets)
- summary: one-paragraph summary (1-2 sentences)

Job info:
Role: {role or "(not provided)"}
Job description: {job_description or "(not provided)"}

Resume text:
\"\"\"{text}\"\"\"

IMPORTANT: produce a valid JSON only. If candidate appears suitable for fresher/intern roles, mention that in summary and reflect in ats_score. Keep arrays short (max 8 items each). Give numeric ats_score and ensure it's plausible and explainable by keywords & experience.
"""

    try:
        response = genai_client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GenAI error: {str(e)}")

    # attempt to parse model output as JSON
    out_text = getattr(response, "text", None) or str(response)
    # The model might return JSON in the text — try to find JSON substring.
    parsed = None
    try:
        parsed = json.loads(out_text)
    except Exception:
        # try to extract first JSON object in text
        import re
        m = re.search(r"(\{[\s\S]*\})", out_text)
        if m:
            try:
                parsed = json.loads(m.group(1))
            except Exception:
                parsed = None

    if not parsed:
        # fallback: return partial structured response
        raise HTTPException(status_code=500, detail=f"Failed to parse model JSON. Raw output: {out_text[:1000]}")

    # sanitize fields
    def ensure_list(v):
        if v is None: return []
        if isinstance(v, list): return v
        if isinstance(v, str): return [v]
        return list(v)

    resp = {
        "ats_score": int(parsed.get("ats_score")) if parsed.get("ats_score") is not None else None,
        "matching_keywords": ensure_list(parsed.get("matching_keywords")),
        "missing_keywords": ensure_list(parsed.get("missing_keywords")),
        "strengths": ensure_list(parsed.get("strengths")),
        "weaknesses": ensure_list(parsed.get("weaknesses")),
        "suggestions": ensure_list(parsed.get("suggestions")),
        "summary": parsed.get("summary", ""),
    }

    return resp

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
