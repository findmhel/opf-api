from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.predict import router as predict_router
from app.routes.data import router as data_router

app = FastAPI(
    title="OpenFinance ML API",
    description="API para predição de adesão ao Open Finance usando Random Forest",
    version="1.0.0"
)

# Configurar CORS para permitir requisições do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Alternativa
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar rotas
app.include_router(predict_router, prefix="/predict", tags=["Predictions"])
app.include_router(data_router, prefix="/data", tags=["Data"])

@app.get("/", tags=["Health"])
def root():
    return {
        "message": "OpenFinance ML API",
        "status": "online",
        "docs": "/docs"
    }

