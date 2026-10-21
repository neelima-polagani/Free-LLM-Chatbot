from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import requests

app = FastAPI()
templates = Jinja2Templates(directory="templates")

MODEL = "llama3.2"
OLLAMA_URL = "http://localhost:11434/api/chat"


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}


@app.post("/api/chat")
async def chat(data: dict):
    message = data.get("message", "").strip()
    history = data.get("history", [])
    temperature = float(data.get("temperature", 0.7))
    temperature = max(0.0, min(2.0, temperature))

    if not message:
        return {"reply": "Please enter a message.", "model_used": MODEL}

    messages = [{
        "role": "system",
        "content": "You are a helpful assistant. Answer simply and clearly."
    }]

    for item in history[-5:]:
        if item.get("role") in ("user", "assistant") and item.get("content"):
            messages.append({
                "role": item["role"],
                "content": item["content"]
            })

    messages.append({"role": "user", "content": message})

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature}
            },
            timeout=180
        )
        response.raise_for_status()
        result = response.json()
        reply = result.get("message", {}).get("content", "")

        return {"reply": reply, "model_used": MODEL}

    except requests.RequestException:
        return {
            "reply": "Cannot connect to Ollama. Check that Ollama is running and llama3.2 is installed.",
            "model_used": MODEL
        }