import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services import llm

logger = logging.getLogger(__name__)

router = APIRouter()


class AnalyzeRequest(BaseModel):
    product_name: str


@router.post("/analyze")
async def analyze(body: AnalyzeRequest):
    if not body.product_name.strip():
        raise HTTPException(status_code=400, detail="product_name is required")

    product_name = body.product_name.strip()
    logger.info("[api] analyzing %r", product_name)

    try:
        result = await llm.analyze(product_name)
    except Exception as e:
        logger.error("[api] failed for %r: %s", product_name, e)
        raise HTTPException(status_code=500, detail=str(e))

    token_usage = result.get("token_usage", {})
    logger.info(
        "[api] done for %r — tokens in=%s out=%s",
        product_name,
        token_usage.get("input", "?"),
        token_usage.get("output", "?"),
    )

    pos = result.get("positive", 0)
    neu = result.get("neutral", 0)
    neg = result.get("negative", 0)

    if pos + neu + neg == 0:
        pos_pct, neu_pct, neg_pct = 34, 33, 33
    else:
        total_sentiment = pos + neu + neg
        pos_pct = round(pos / total_sentiment * 100)
        neg_pct = round(neg / total_sentiment * 100)
        neu_pct = 100 - pos_pct - neg_pct

    review_count = result.get("review_count", 0)
    if review_count >= 10:
        confidence = "High"
    elif review_count >= 5:
        confidence = "Medium"
    else:
        confidence = "Low"

    shopee_keyword = product_name.replace(" ", "+")
    shopee_url = f"https://shopee.co.id/search?keyword={shopee_keyword}"

    return {
        "product_name": product_name,
        "avg_rating": result.get("avg_rating", 0),
        "review_count": review_count,
        "positive": pos,
        "neutral": neu,
        "negative": neg,
        "pos_pct": pos_pct,
        "neu_pct": neu_pct,
        "neg_pct": neg_pct,
        "pros": result.get("pros", []),
        "cons": result.get("cons", []),
        "sample_snippets": result.get("sample_snippets", {}),
        "matched": result.get("matched", False),
        "confidence": confidence,
        "shopee_url": shopee_url,
        "prompt_preview": llm.get_prompt(product_name),
        "token_usage": token_usage,
    }
