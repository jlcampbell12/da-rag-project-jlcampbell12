from contextlib import asynccontextmanager

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialise shared resources here in later phases
    yield
    # Shutdown: release shared resources here in later phases


app = FastAPI(
    title="Wikipedia RAG System",
    description="RAG system using LlamaIndex with Wikipedia articles and Azure OpenAI",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "wikipedia-rag",
        "version": "0.1.0",
    }
