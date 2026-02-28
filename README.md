# Sobi Sentiment Analyzer

Instant product sentiment analysis from live web reviews. Type a product name → get average rating, sentiment breakdown, and side-by-side pros/cons in seconds.

<img width="767" height="958" alt="image" src="https://github.com/user-attachments/assets/2bd5b0df-7c2a-45d0-a8e8-95c41948537f" />

## Stack

- **FastAPI** — backend + serves the UI
- **Jinja2** — single-page HTML template
- **Tavily** — live web search for product reviews
- **OpenRouter** (`arcee-ai/trinity-large-preview:free`) — LLM analysis
- **Tailwind CSS + anime.js** — UI + animations (CDN)

## Agentic Workflow

Each search triggers a 2-step agent pipeline in `app/services/llm.py`:

```
User query
   │
   ▼
[Step 1] Tavily Search
   • Query: "{product} review pros cons rating"
   • Returns: 5 web snippets (300 chars each)
   │
   ▼
[Step 2] OpenRouter LLM (arcee-ai/trinity-large-preview:free)
   • Input: search snippets + structured prompt
   • Output: JSON { avg_rating, review_count, positive, neutral,
                    negative, pros[], cons[] }
   │
   ▼
API response → animated UI
```

## Setup

```bash
uv sync
```

**.env**
```
TAVILY_API_KEY=your_tavily_key
OPENROUTER_API_KEY=your_openrouter_key
```

Get keys:
- Tavily: https://tavily.com
- OpenRouter: https://openrouter.ai

## Run

```bash
# Terminal 1 — web server
make dev

Open http://localhost:8000

## API

```
POST /api/analyze
Content-Type: application/json

{ "product_name": "Samsung Galaxy S21" }
```

Response fields: `avg_rating`, `review_count`, `pos_pct`, `neu_pct`, `neg_pct`, `pros[]`, `cons[]`, `shopee_url`, `token_usage`.

## Token Usage

Logged per request and shown in the UI under "LLM Prompt Preview":

```
[tavily] got 5 results for 'Samsung Galaxy S21'
[openrouter] tokens — input: 612, output: 87, total: 699
```
