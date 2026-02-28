import os
import json
import logging
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolUnionParam

logger = logging.getLogger(__name__)

MODEL = "arcee-ai/trinity-large-preview:free"
MAX_ITERATIONS = 5

SYSTEM_PROMPT = """You are a product review analyst agent. Your goal is to gather enough review data about a product and produce a structured sentiment analysis.

You have access to a `search_web` tool. Use it to search for product reviews. You may call it multiple times with different queries if the first search lacks sufficient data.

When you have enough information, respond with ONLY valid JSON (no markdown, no explanation):
{"avg_rating": float, "review_count": int, "positive": int, "neutral": int, "negative": int, "pros": ["string", "string", "string"], "cons": ["string", "string", "string"]}

Rules:
- avg_rating: 1.0-5.0 (estimate from review sentiment)
- positive + neutral + negative should sum to review_count
- pros: 3 concise bullet points (max 8 words each)
- cons: 3 concise bullet points (max 8 words each)
- Final response must be JSON only, no other text"""

TOOLS: list[ChatCompletionToolUnionParam] = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for product reviews, ratings, and user feedback. Returns snippets from review pages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, e.g. 'Samsung Galaxy S21 review pros cons rating'",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


def get_prompt(product_name: str) -> str:
    return f"Analyze sentiment for: {product_name}\n\nUse the search_web tool to find reviews, then return structured JSON."


def _extract_json(raw: str) -> dict:
    if "```" in raw:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        raw = raw[start:end]
    elif not raw.startswith("{"):
        start = raw.find("{")
        if start != -1:
            raw = raw[start:]
    return json.loads(raw)


async def analyze(product_name: str) -> dict:
    from tavily import TavilyClient
    from openai import AsyncOpenAI

    tavily_key = os.getenv("TAVILY_API_KEY", "")
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")

    if not tavily_key or not openrouter_key:
        raise RuntimeError("TAVILY_API_KEY and OPENROUTER_API_KEY must be set in .env")

    tavily = TavilyClient(api_key=tavily_key)
    llm = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_key,
    )

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Analyze product reviews for: {product_name}"},
    ]

    total_input_tokens = 0
    total_output_tokens = 0
    iteration = 0

    logger.info("[agent] starting agentic loop for %r", product_name)

    while iteration < MAX_ITERATIONS:
        iteration += 1
        logger.info("[agent] iteration %d/%d", iteration, MAX_ITERATIONS)

        completion = await llm.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=600,
        )

        usage = completion.usage
        if usage:
            total_input_tokens += usage.prompt_tokens or 0
            total_output_tokens += usage.completion_tokens or 0
            logger.info(
                "[openrouter] iteration %d tokens — input: %s, output: %s",
                iteration,
                usage.prompt_tokens,
                usage.completion_tokens,
            )

        choice = completion.choices[0]
        assistant_message = choice.message

        # Append assistant message to history
        messages.append(assistant_message.model_dump(exclude_none=True))  # type: ignore

        # No tool calls → LLM is done, extract final JSON
        if not assistant_message.tool_calls:
            raw = (assistant_message.content or "").strip()
            logger.info("[agent] done after %d iterations, parsing final JSON", iteration)
            data = _extract_json(raw)
            data["matched"] = True
            data["sample_snippets"] = {}
            data["token_usage"] = {
                "input": total_input_tokens,
                "output": total_output_tokens,
            }
            logger.info(
                "[agent] total tokens — input: %d, output: %d",
                total_input_tokens,
                total_output_tokens,
            )
            return data

        # Execute each tool call
        for tool_call in assistant_message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            if fn_name == "search_web":
                query = fn_args.get("query", f"{product_name} review")
                logger.info("[tavily] agent searching: %r", query)

                search_response = tavily.search(
                    query=query,
                    search_depth="basic",
                    max_results=5,
                )
                results = search_response.get("results", [])
                logger.info("[tavily] got %d results for %r", len(results), query)

                snippets = []
                for r in results:
                    content = r.get("content", "")
                    if content:
                        snippets.append(content[:300])

                tool_result = "\n---\n".join(snippets) if snippets else "No results found."
            else:
                tool_result = f"Unknown tool: {fn_name}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            })

    raise RuntimeError(f"Agent did not produce a result after {MAX_ITERATIONS} iterations")
