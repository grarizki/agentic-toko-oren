import os
import json
import logging

logger = logging.getLogger(__name__)

MODEL = "arcee-ai/trinity-large-preview:free"

PROMPT_TEMPLATE = """You are a product review analyst. Analyze these search results about "{product_name}" reviews and return ONLY valid JSON with no markdown, no explanation:
{{"avg_rating": float, "review_count": int, "positive": int, "neutral": int, "negative": int, "pros": ["string", "string", "string"], "cons": ["string", "string", "string"]}}

Rules:
- avg_rating: 1.0-5.0 (estimate from review sentiment in the search results)
- positive + neutral + negative should sum to review_count
- pros: 3 concise bullet points (max 8 words each) extracted from positive mentions
- cons: 3 concise bullet points (max 8 words each) extracted from negative mentions
- JSON only, no other text

Search results:
{search_results}"""


def get_prompt(product_name: str) -> str:
    return PROMPT_TEMPLATE.format(product_name=product_name, search_results="<live search results>")


async def analyze(product_name: str) -> dict:
    from tavily import TavilyClient
    from openai import AsyncOpenAI

    tavily_key = os.getenv("TAVILY_API_KEY", "")
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")

    if not tavily_key or not openrouter_key:
        raise RuntimeError("TAVILY_API_KEY and OPENROUTER_API_KEY must be set in .env")

    # Step 1: Tavily search
    logger.info("[tavily] searching: %r", product_name)
    client = TavilyClient(api_key=tavily_key)
    search_response = client.search(
        query=f"{product_name} review pros cons rating",
        search_depth="basic",
        max_results=5,
    )
    result_count = len(search_response.get("results", []))
    logger.info("[tavily] got %d results for %r", result_count, product_name)

    snippets = []
    for r in search_response.get("results", []):
        content = r.get("content", "")
        if content:
            snippets.append(content[:300])

    search_text = "\n---\n".join(snippets) if snippets else "No review data found."

    # Step 2: OpenRouter LLM analyzes search results
    prompt = PROMPT_TEMPLATE.format(product_name=product_name, search_results=search_text)
    logger.info("[openrouter] sending request for %r (model=%s, prompt_len=%d)", product_name, MODEL, len(prompt))

    llm = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_key,
    )
    completion = await llm.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
    )

    usage = completion.usage
    input_tokens = usage.prompt_tokens if usage else "?"
    output_tokens = usage.completion_tokens if usage else "?"
    total_tokens = usage.total_tokens if usage else "?"
    logger.info("[openrouter] tokens — input: %s, output: %s, total: %s", input_tokens, output_tokens, total_tokens)

    raw = (completion.choices[0].message.content or "").strip()

    # Extract JSON even if model wraps it in markdown
    if "```" in raw:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        raw = raw[start:end]
    elif not raw.startswith("{"):
        start = raw.find("{")
        if start != -1:
            raw = raw[start:]

    data = json.loads(raw)
    data["matched"] = True
    data["sample_snippets"] = {}
    data["token_usage"] = {"input": input_tokens, "output": output_tokens}
    return data
