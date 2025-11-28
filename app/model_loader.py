import json
from pyspark.ml import PipelineModel
from app.config import MODEL_PATH, METADATA_PATH

def load_metadata():
    try:
        with open(METADATA_PATH, "r") as f:
            return json.load(f)
    except:
        return {}

def load_model():
    return PipelineModel.load(MODEL_PATH)

metadata = load_metadata()
model = load_model()
