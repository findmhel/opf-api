from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.predict import router as predict_router
from app.routes.data import router as data_router
from app.routes.predict_sklearn import router as predict_sklearn_router

app = FastAPI(
    title="OpenFinance ML API",
    description="API para predição de adesão ao Open Finance usando Random Forest",
    version="1.0.0"
)

# Configurar CORS para permitir requisições do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust for production)
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar rotas
# rota principal de predição (Spark pipeline or sklearn via service chooser)
app.include_router(predict_router, prefix="/predict", tags=["Predictions"])
# rota sklearn específica (acessível em /predict/sklearn)
app.include_router(predict_sklearn_router, prefix="/predict", tags=["Predictions"])
app.include_router(data_router, prefix="/data", tags=["Data"])

@app.get("/", tags=["Health"])
def root():
    return {
        "message": "OpenFinance ML API",
        "status": "online",
        "docs": "/docs"
    }


# Log environment on startup to help diagnose production issues
@app.on_event("startup")
def log_startup():
    import os
    logger = __import__('logging').getLogger(__name__)
    logger.info("Starting OpenFinance ML API")
    logger.info("ENV: SKLEARN_MODEL_PATH=%s", os.environ.get('SKLEARN_MODEL_PATH'))
    logger.info("ENV: MODEL_PATH=%s", os.environ.get('MODEL_PATH'))
    logger.info("ENV: PARQUET_PATH=%s", os.environ.get('PARQUET_PATH'))

