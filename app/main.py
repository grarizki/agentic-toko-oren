from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import logging
import os

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from app.routes import pages, api

app = FastAPI(title="Product Sentiment Analyzer")

templates = Jinja2Templates(directory="app/templates")

app.include_router(pages.router)
app.include_router(api.router, prefix="/api")
