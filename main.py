import os
import fitz  # PyMuPDF
from gtts import gTTS
from openai import OpenAI
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from tempfile import NamedTemporaryFile
from fastapi.middleware.cors import CORSMiddleware
# Setup Groq API
os.environ["GROQ_API_KEY"] = "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"
client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔴 For production, replace "*" with frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    doc.close()
    return text

def split_text_into_chunks(text, max_chunk_size=5000):
    chunks, current_chunk = [], ""
    for line in text.split("\n"):
        if len(current_chunk) + len(line) <= max_chunk_size:
            current_chunk += line + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = line + "\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def summarize_chunk(chunk):
    summary_response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": (
                "Summarize the following text in a paragraph understandable by a non-medical person. "
                "Use this exact format:\n"
                "Objective: [objective]\n"
                "Drug: [drug name or 'not specified']\n"
                "How it works: [mechanism]\n"
                "Duration: [timeframe]\n"
                "Number of visits: [visit count]\n"
                "What happens in each visit: [visit details]\n"
                "Side effects: [side effects]\n"
                "Benefits: [benefits]\n"
                "Patient rights and responsibilities: [rights and duties]\n"
                "Contact details: [contact info]\n\n"
                "Text to summarize: " + chunk
            )}
        ],
        model="llama-3.1-8b-instant",
    )
    return summary_response.choices[0].message.content

def combine_summaries(chunk_summaries):
    final_summary = "\n".join(chunk_summaries)
    return final_summary

def refine_english_text(text):
    prompt = (
        "Remove grammar errors from this text. Use simple English and no symbols:\n\n" + text
    )
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant skilled in English."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.1-8b-instant",
    )
    return response.choices[0].message.content.strip()

def translate_to_tamil(text):
    words = text.split()
    chunks = [' '.join(words[i:i + 100]) for i in range(0, len(words), 100)]
    tamil_text = ""
    for chunk in chunks:
        prompt = (
            "இந்த ஆங்கில உரையை தமிழுக்கு மொழிபெயர்க்கவும். எளிமையான உள்ளூர் தமிழில்:\n\n" + chunk
        )
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "நீங்கள் தமிழில் திறமையான உதவியாளர்."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=4000
        )
        tamil_text += response.choices[0].message.content.strip() + " "
    return tamil_text.strip()

def text_to_speech(tamil_text):
    filename = "summary_tamil.mp3"
    tts = gTTS(text=tamil_text, lang="ta", slow=False)
    tts.save(filename)
    return filename

@app.post("/convert-pdf")
async def convert_pdf(file: UploadFile = File(...)):
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        text = extract_text_from_pdf(tmp_path)
        chunks = split_text_into_chunks(text)
        summaries = [summarize_chunk(chunk) for chunk in chunks]
        summary = combine_summaries(summaries)
        refined = refine_english_text(summary)
        tamil = translate_to_tamil(refined)
        audio_file = text_to_speech(tamil)

        return {
            "english_summary": summary,
            "refined_english": refined,
            "tamil_translation": tamil,
            "audio_url": f"/download-audio"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        os.remove(tmp_path)

@app.get("/download-audio")
def download_audio():
    return FileResponse("summary_tamil.mp3", media_type="audio/mpeg", filename="summary_tamil.mp3")
