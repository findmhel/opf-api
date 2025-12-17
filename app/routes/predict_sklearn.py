from fastapi import APIRouter, HTTPException, Body
from typing import Any, Dict
from app.model_loader import load_sklearn_model, predict_sklearn
import os

router = APIRouter()


@router.post('/sklearn')
def predict(payload: Dict[str, Any] = Body(...)):
    path = os.environ.get('SKLEARN_MODEL_PATH')
    if not path:
        raise HTTPException(status_code=400, detail='SKLEARN_MODEL_PATH not configured')

    try:
        model_obj = load_sklearn_model(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    client = payload
    if not isinstance(client, dict):
        raise HTTPException(status_code=400, detail='Payload must be a JSON object')

    try:
        result = predict_sklearn(model_obj, client)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result
