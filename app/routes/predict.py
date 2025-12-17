from fastapi import APIRouter, HTTPException
from app.schemas.user_data import UserData
from app.services.prediction_service import run_prediction, run_explanations
import logging
import traceback

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/")
def predict(user: UserData):
    data = user.dict()
    try:
        pred = run_prediction(data)
        exp = run_explanations(data)
        return {**pred, **exp}
    except RuntimeError as e:
        # Expected runtime error when no model is available; return 503 Service Unavailable
        tb = traceback.format_exc()
        logger.error("Prediction runtime error: %s", tb)
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Unexpected error in /predict: %s", tb)
        raise HTTPException(status_code=500, detail="Erro interno no servidor ao processar predição")
