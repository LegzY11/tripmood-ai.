import json
import os
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent.parent
HTML_FILE = APP_DIR / "tripmood.html"

CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()

AI_MODEL = os.getenv(
  "CLOUDFLARE_AI_MODEL",
  "@cf/meta/llama-3.1-8b-instruct"
).strip()

CLOUDFLARE_URL = (
    "https://api.cloudflare.com/client/v4/accounts/"
    "{account_id}/ai/v1/chat/completions"
)

app = FastAPI(title="TripMood AI", version="3.0")

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
    "views", "hidden", "party", "food",
    "culture", "nature", "art", "active"
}


def require_api_key():
    if not CLOUDFLARE_ACCOUNT_ID or not CLOUDFLARE_API_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="Cloudflare Workers AI не настроен. Проверьте переменные окружения."
        )


def cloudflare_generate(
    system_instruction: str,
    user_text: str,
    *,
    json_mode=False
) -> str:

    require_api_key()

    url = CLOUDFLARE_URL.format(
        account_id=CLOUDFLARE_ACCOUNT_ID
    )

    payload = {
        "model": AI_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_instruction
            },
            {
                "role": "user",
                "content": user_text
            }
        ],
        "temperature": 0.2 if json_mode else 0.7,
        "max_tokens": 1200
    }

    if json_mode:
        payload["response_format"] = {
            "type": "json_object"
        }

    try:
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=60
        )

        if response.status_code == 429:
            raise HTTPException(
                status_code=429,
                detail="Превышен лимит запросов Cloudflare AI. Попробуйте позже."
            )

        if response.status_code in (401, 403):
            raise HTTPException(
                status_code=503,
                detail="Cloudflare API Token недействителен или не имеет нужных разрешений."
            )

        if response.status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail="Ошибка Cloudflare Workers AI. Проверьте модель и настройки аккаунта."
            )

        data = response.json()

        choices = data.get("choices") or []
        answer = ""

        if choices:
            answer = (
                choices[0]
                .get("message", {})
                .get("content", "")
            )

        if not isinstance(answer, str) or not answer.strip():
            raise HTTPException(
                status_code=502,
                detail="Cloudflare AI вернул пустой ответ."
            )

        return answer.strip()

    except HTTPException:
        raise

    except requests.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Cloudflare AI отвечает слишком долго. Повторите запрос."
        )

    except requests.RequestException:
        raise HTTPException(
            status_code=502,
            detail="Не удалось связаться с Cloudflare Workers AI."
        )

    except (ValueError, KeyError, TypeError):
        raise HTTPException(
            status_code=502,
            detail="Не удалось обработать ответ Cloudflare AI."
        )


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=3000)


class SearchPlanRequest(BaseModel):
    city: str = Field(min_length=1, max_length=100)
    country: str = Field(min_length=1, max_length=100)
    prompt: str = Field(default="", max_length=1500)
    selected_categories: list[str] = Field(
        default_factory=list,
        max_length=8
    )


@app.get("/")
def home():
    if not HTML_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Файл сайта не найден."
        )

    return FileResponse(
        HTML_FILE,
        media_type="text/html"
    )


@app.get("/api/health")
def health():
    return {
        "status": "online",
        "ai_provider": "Cloudflare Workers AI",
        "model": AI_MODEL,
        "api_key_configured": bool(
            CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN
        )
    }


@app.post("/api/chat")
def chat(data: ChatRequest):
    message = data.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Введите сообщение."
        )

    answer = cloudflare_generate(
        SYSTEM_PROMPT,
        message
    )

    return {"answer": answer}


@app.post("/api/search-plan")
def search_plan(data: SearchPlanRequest):

    selected = [
        c for c in data.selected_categories
        if c in ALLOWED_CATEGORIES
    ]

    user_request = (
        f"Город: {data.city}. Страна: {data.country}. "
        f"Категории, отмеченные пользователем: {', '.join(selected)}. "
        f"Дополнительные пожелания: {data.prompt or 'не указаны'}."
    )

    instruction = """
Ты модуль планирования поиска туристических мест.

Выбери подходящие категории из строго заданного списка:
views, hidden, party, food, culture, nature, art, active.

Верни только JSON формата:
{"categories":["views","culture"],"reason":"короткое объяснение по-русски"}.

Сохраняй категории, отмеченные пользователем, и добавляй категории,
которые явно следуют из пожеланий.
Если пожелания пустые, используй отмеченные.
Не придумывай названия конкретных мест.
Не добавляй категории вне списка.
"""

    content = cloudflare_generate(
        instruction,
        user_request,
        json_mode=True
    )

    try:
        parsed = json.loads(content)

        if not isinstance(parsed, dict):
            raise ValueError("Expected JSON object")

    except (ValueError, TypeError):
        raise HTTPException(
            status_code=502,
            detail="ИИ вернул некорректный план поиска. Попробуйте ещё раз."
        )

    result_categories = parsed.get("categories", [])

    if not isinstance(result_categories, list):
        result_categories = []

    categories = [
        item for item in result_categories
        if isinstance(item, str)
        and item in ALLOWED_CATEGORIES
    ]

    categories = list(dict.fromkeys(selected + categories))

    if not categories:
        categories = ["views", "hidden"]

    reason = parsed.get("reason", "")

    return {
        "categories": categories,
        "reason": str(reason)[:300],
        "city": data.city,
        "country": data.country,
    }
