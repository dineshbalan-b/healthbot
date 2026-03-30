import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv

from app.api.routes import router as api_router
from app.services.rag_service import init_vector_store

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(" Starting AI Medicine Assistant...")
    init_vector_store()
    yield
    print(" Shutting down...")

app = FastAPI(title="AI Document Assistant", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(api_router, prefix="/api")

@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    # Make sure to run the application with python -m app.main
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
