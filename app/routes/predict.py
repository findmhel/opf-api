from fastapi import APIRouter
from app.schemas.user_data import UserData
from app.services.prediction_service import run_prediction, run_explanations

router = APIRouter()

@router.post("/")
def predict(user: UserData):
    data = user.dict()

    pred = run_prediction(data)
    exp = run_explanations(data)

    return {
        **pred,
        **exp
    }
