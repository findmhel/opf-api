import json
from typing import Any, Dict, List, Optional
from pathlib import Path
from app.config import MODEL_PATH, METADATA_PATH
import os
import logging
import urllib.request
import shutil


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
    logger = logging.getLogger(__name__)
    p = Path(path)

    # If the file doesn't exist but a SKLEARN_MODEL_URL is configured, try to download it.
    if not p.exists():
        url = os.environ.get('SKLEARN_MODEL_URL')
        if url:
            try:
                logger.info('SKLEARN_MODEL_PATH not found; attempting download from SKLEARN_MODEL_URL')
                # ensure parent dir
                p.parent.mkdir(parents=True, exist_ok=True)
                # stream download to file
                with urllib.request.urlopen(url) as resp, open(p, 'wb') as out_file:
                    shutil.copyfileobj(resp, out_file)
                logger.info('Downloaded sklearn model to %s', str(p))
            except Exception as e:
                logger.exception('Failed to download sklearn model from %s: %s', url, e)
        # After attempted download, if still missing raise
    if not p.exists():
        raise FileNotFoundError(f'Sklearn model not found at {path}')
    obj = joblib.load(path)
    return obj


def predict_sklearn(model_obj: Any, client: Dict[str, Any]) -> Dict[str, Any]:
    """Predict single client dict using sklearn model object produced by load_sklearn_model.

    `client` is a mapping with keys for raw features (human-readable). 
    Uses LabelEncoders stored in model_obj to transform categorical features.
    Returns a dict with 'prediction' and optional 'probability'.
    """
    import pandas as pd
    import numpy as np
    import logging
    
    logger = logging.getLogger(__name__)
    
    feature_cols: List[str] = model_obj.get('feature_columns') or []
    model_skl = model_obj.get('model')
    encoders = model_obj.get('encoders', {})
    categorical_cols = model_obj.get('categorical_cols', [])
    numeric_cols = model_obj.get('numeric_cols', [])
    
    # Fallback: se categorical_cols/numeric_cols estiverem vazios, reconstrói a partir de feature_columns
    if not categorical_cols and not numeric_cols and feature_cols:
        logger.warning('categorical_cols and numeric_cols not found in model, reconstructing from feature_columns')
        # Features categóricas terminam com _idx
        categorical_cols = [col.replace('_idx', '') for col in feature_cols if col.endswith('_idx')]
        # Features numéricas não terminam com _idx
        numeric_cols = [col for col in feature_cols if not col.endswith('_idx')]
    
    logger.info(f'Feature columns count: {len(feature_cols)}')
    logger.info(f'Categorical cols: {categorical_cols}')
    logger.info(f'Numeric cols: {numeric_cols}')
    
    if model_skl is None or not feature_cols:
        raise RuntimeError('Invalid sklearn model object')

    # Helper para buscar valor com várias variações de chave
    def get_value(col_name):
        # Tenta com o nome original
        val = client.get(col_name)
        if val is not None:
            return val
        # Tenta com underscore
        val = client.get(col_name.replace(' ', '_'))
        if val is not None:
            return val
        # Tenta com espaço (inverso)
        val = client.get(col_name.replace('_', ' '))
        if val is not None:
            return val
        return None

    # Constrói vetor de features
    feature_values = []
    
    # Primeiro, as categóricas encodadas
    for col in categorical_cols:
        col_idx = col + "_idx"
        if col_idx in feature_cols:
            raw_value = get_value(col)
            if raw_value is None:
                raw_value = '_MISSING_'
            
            logger.info(f'Categorical {col}: {raw_value}')
            
            # Usa o encoder treinado
            encoder = encoders.get(col)
            if encoder:
                try:
                    # Tenta transformar; se valor desconhecido, usa 0
                    encoded = encoder.transform([str(raw_value)])[0]
                except (ValueError, KeyError):
                    # Valor não visto no treinamento
                    encoded = 0
                feature_values.append(encoded)
            else:
                feature_values.append(0)
    
    logger.info(f'After categorical: {len(feature_values)} features')
    
    # Depois, as numéricas
    for col in numeric_cols:
        if col in feature_cols:
            raw_value = get_value(col)
            logger.info(f'Numeric {col}: {raw_value}')
            try:
                feature_values.append(float(raw_value) if raw_value is not None else 0.0)
            except (ValueError, TypeError):
                feature_values.append(0.0)
    
    logger.info(f'Final feature count: {len(feature_values)}')
    logger.info(f'Feature values: {feature_values}')
    
    X = np.array([feature_values])
    logger.info(f'X shape: {X.shape}')
    
    proba = None
    if hasattr(model_skl, 'predict_proba'):
        proba = model_skl.predict_proba(X)[0].tolist()
    pred = model_skl.predict(X)[0].item()
    return {'prediction': pred, 'probability': proba}
