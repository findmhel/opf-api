import json
from typing import Any, Dict, List, Optional
from pathlib import Path
from app.config import MODEL_PATH, METADATA_PATH


# Lazy caches
_metadata_cache: Optional[Dict[str, Any]] = None
_spark_model_cache: Optional[Any] = None


def load_metadata() -> Dict[str, Any]:
    global _metadata_cache
    if _metadata_cache is not None:
        return _metadata_cache
    try:
        with open(METADATA_PATH, "r") as f:
            _metadata_cache = json.load(f)
    except Exception:
        _metadata_cache = {}
    return _metadata_cache


def _load_spark_model_internal(path: str):
    # internal helper to avoid importing pyspark at module import time
    from pyspark.ml import PipelineModel
    return PipelineModel.load(path)


def get_spark_model() -> Optional[Any]:
    """Lazily carrega o PipelineModel do Spark apenas se for necessário.

    Retorna None se o modelo não existir no caminho configurado.
    """
    global _spark_model_cache
    if _spark_model_cache is not None:
        return _spark_model_cache

    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        return None

    try:
        _spark_model_cache = _load_spark_model_internal(str(model_path))
    except Exception:
        # falha ao carregar Spark model — propague None para que o serviço escolha sklearn
        _spark_model_cache = None
    return _spark_model_cache


def load_model_from(path: str):
    """Carrega um PipelineModel de um caminho arbitrário (lazy import)."""
    from pyspark.ml import PipelineModel
    return PipelineModel.load(path)


def predict_with_model(model_obj, df):
    """Executa a transformação usando um PipelineModel já carregado.

    `df` deve ser um pyspark DataFrame com as colunas esperadas pelo pipeline.
    Retorna o DataFrame transformado.
    """
    return model_obj.transform(df)


def load_sklearn_model(path: str) -> Any:
    """Load a scikit-learn model saved with joblib.

    Expected object is a dict with keys:
      - 'model': sklearn estimator
      - 'feature_columns': list[str]
    """
    import joblib
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Sklearn model not found at {path}')
    obj = joblib.load(path)
    return obj


def predict_sklearn(model_obj: Any, client: Dict[str, Any]) -> Dict[str, Any]:
    """Predict single client dict using sklearn model object produced by load_sklearn_model.

    `client` is a mapping with keys for raw features (human-readable). We build a feature vector
    matching `model_obj['feature_columns']` by selecting values or using 0 for missing dummies.
    Returns a dict with 'prediction' and optional 'probability'.
    """
    import pandas as pd
    feature_cols: List[str] = model_obj.get('feature_columns') or []
    model_skl = model_obj.get('model')
    if model_skl is None or not feature_cols:
        raise RuntimeError('Invalid sklearn model object')

    row = pd.DataFrame([client])
    row_enc = pd.get_dummies(row)

    for c in feature_cols:
        if c not in row_enc.columns:
            row_enc[c] = 0

    X = row_enc[feature_cols].astype(float)
    proba = None
    if hasattr(model_skl, 'predict_proba'):
        proba = model_skl.predict_proba(X)[0].tolist()
    pred = model_skl.predict(X)[0].item()
    return {'prediction': pred, 'probability': proba}
