"""Application entry point."""

from __future__ import annotations

from fastapi import FastAPI

from app.api import create_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Conversational SHL Assessment Recommender",
        version="0.1.0",
        description="Catalog-grounded SHL assessment recommender API.",
    )
    app.include_router(create_router())
    return app


app = create_app()
