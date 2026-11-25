import json
import os
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent.parent
HTML_FILE = APP_DIR / "tripmood.html"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

app = FastAPI(title="TripMood AI", version="2.0")

SYSTEM_PROMPT = """
Ты — TripMood AI, дружелюбный помощник по путешествиям.
Отвечай на русском языке, кратко и практично.
Помогай выбирать районы, достопримечательности, необычные места,
составлять маршруты и учитывать интересы и бюджет пользователя.
Не выдумывай точные адреса, цены, часы работы, наличие билетов или
актуальные события. Если данные могут меняться, попроси проверить их
в официальном источнике. Фактический поиск мест на сайте выполняется
отдельно по данным OpenStreetMap.
"""

ALLOWED_CATEGORIES = {
    "views", "hidden", "party", "food", "culture", "nature", "art", "active"
}


def require_api_key():
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Облачный ИИ ещё не настроен: добавьте GEMINI_API_KEY в переменные окружения сервера."
        )


def gemini_generate(system_instruction: str, user_text: str, *, json_mode=False) -> str:
    require_api_key()
    url = GEMINI_URL.format(model=GEMINI_MODEL)
    payload = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": {"temperature": 0.2 if json_mode else 0.7},
    }
    if json_mode:
        payload["generationConfig"]["responseMimeType"] = "application/json"
    try:
        response = requests.post(
            url,
            headers={"x-goog-api-key": GEMINI_API_KEY},
            json=payload,
            timeout=45,
        )
        if response.status_code == 429:
            raise HTTPException(status_code=429, detail="Превышен временный лимит бесплатных AI-запросов. Подождите и попробуйте снова.")
        if response.status_code in (401, 403):
            raise HTTPException(status_code=503, detail="Ключ Gemini API недействителен или не имеет доступа. Проверьте настройки сервера.")
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates") or []
        parts = (candidates[0].get("content", {}).get("parts", []) if candidates else [])
        answer = "".join(p.get("text", "") for p in parts).strip()
        if not answer:
            raise HTTPException(status_code=502, detail="ИИ вернул пустой ответ. Попробуйте сформулировать запрос иначе.")
        return answer
    except HTTPException:
        raise
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="Облачный ИИ отвечает слишком долго. Повторите запрос.")
    except requests.RequestException:
        raise HTTPException(status_code=502, detail="Не удалось связаться с облачным ИИ. Проверьте подключение сервера.")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=3000)


class SearchPlanRequest(BaseModel):
    city: str = Field(min_length=1, max_length=100)
    country: str = Field(min_length=1, max_length=100)
    prompt: str = Field(default="", max_length=1500)
    selected_categories: list[str] = Field(default_factory=list, max_length=8)


@app.get("/")
def home():
    if not HTML_FILE.exists():
        raise HTTPException(status_code=404, detail="Файл сайта не найден.")
    return FileResponse(HTML_FILE, media_type="text/html")


@app.get("/api/health")
def health():
    return {
        "status": "online",
        "ai_provider": "Gemini API",
        "model": GEMINI_MODEL,
        "api_key_configured": bool(GEMINI_API_KEY),
    }


@app.post("/api/chat")
def chat(data: ChatRequest):
    message = data.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Введите сообщение.")
    answer = gemini_generate(SYSTEM_PROMPT, message)
    return {"answer": answer}


@app.post("/api/search-plan")
def search_plan(data: SearchPlanRequest):
    selected = [c for c in data.selected_categories if c in ALLOWED_CATEGORIES]
    user_request = (
        f"Город: {data.city}. Страна: {data.country}. "
        f"Категории, отмеченные пользователем: {', '.join(selected)}. "
        f"Дополнительные пожелания: {data.prompt or 'не указаны'}."
    )
    instruction = """
Ты модуль планирования поиска туристических мест.
На основании запроса выбери подходящие категории из строго заданного списка:
views, hidden, party, food, culture, nature, art, active.
Верни только JSON формата:
{"categories":["views","culture"],"reason":"короткое объяснение по-русски"}.
Сохраняй категории, отмеченные пользователем, и добавляй категории,
которые явно следуют из пожеланий. Если пожелания пустые, используй отмеченные.
Не придумывай названия конкретных мест. Не добавляй категории вне списка.
"""
    content = gemini_generate(instruction, user_request, json_mode=True)
    try:
        parsed = json.loads(content)
    except (ValueError, TypeError):
        raise HTTPException(status_code=502, detail="ИИ вернул некорректный план поиска. Попробуйте ещё раз.")
    categories = [
        item for item in parsed.get("categories", [])
        if isinstance(item, str) and item in ALLOWED_CATEGORIES
    ]
    categories = list(dict.fromkeys(selected + categories))
    if not categories:
        categories = ["views", "hidden"]
    return {
        "categories": categories,
        "reason": str(parsed.get("reason", ""))[:300],
        "city": data.city,
        "country": data.country,
    }
