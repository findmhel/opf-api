from pyspark.sql import Row
from app.model_loader import load_sklearn_model, predict_sklearn, get_spark_model, load_model_from
from app.spark_session import get_spark
from app.utils.feature_utils import extract_assembler_input_cols
from app.utils.explanations import compute_global_importance
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Do not load the spark model at import time to avoid starting JVM on import.
_spark_model = None
_spark_model_checked = False
_feature_names_cache = []

# Cache sklearn model object to avoid repeated loads
_sklearn_cache = {
    'loaded': False,
    'obj': None,
    'path': None
}


def _get_sklearn_model_if_configured():
    path = os.environ.get('SKLEARN_MODEL_PATH')
    if not path:
        return None
    p = Path(path)

    # If cached and path matches, return cached object
    if _sklearn_cache['loaded'] and _sklearn_cache['path'] == str(p):
        return _sklearn_cache['obj']

    # Try loading (this will attempt to download if SKLEARN_MODEL_URL is set)
    try:
        obj = load_sklearn_model(str(p))
        _sklearn_cache.update({'loaded': True, 'obj': obj, 'path': str(p)})
        return obj
    except FileNotFoundError:
        logger.warning('SKLEARN_MODEL_PATH set but file not present and download failed: %s', path)
        return None
    except Exception:
        logger.exception('Unexpected error loading sklearn model from %s', path)
        return None


def run_prediction(data: dict):
    """
    Executa predição; preferencialmente usa sklearn (joblib) se `SKLEARN_MODEL_PATH` estiver configurado
    e o arquivo existir. Caso contrário, usa o pipeline Spark existente.
    """
    # Tenta sklearn primeiro
    skl = _get_sklearn_model_if_configured()
    if skl is not None:
        # Normaliza entrada para formato que predict_sklearn espera (chaves humanas/underscore)
        client = {
            'Faixa_etaria': data.get('Faixa_etaria') or data.get('Faixa etária'),
            'Estado': data.get('Estado'),
            'Sexo': data.get('Sexo'),
            'Ocupacao': data.get('Ocupacao'),
            'Escolaridade': data.get('Escolaridade'),
            'Gp_renda': data.get('Gp_renda') or data.get('Gp renda'),
            'Tipo_da_conta': data.get('Tipo_da_conta'),
            'Gp_score_de_credito': data.get('Gp_score_de_credito') or data.get('Gp score de crédito'),
            'Gp_limite_do_cartao': data.get('Gp_limite_do_cartao') or data.get('Gp limite do cartão'),
            'Tempo_conta_atv': data.get('Tempo_conta_atv'),
            'Outros_bancos': data.get('Outros_bancos'),
            'Emprestimo': data.get('Emprestimo'),
            'Financiamento': data.get('Financiamento'),
            'Cartao_de_credito': data.get('Cartao_de_credito'),
            'Usa_cheque': data.get('Usa_cheque'),
            'Atrasa_pag': data.get('Atrasa_pag'),
            'Investimentos': data.get('Investimentos'),
            'Usa_pix': data.get('Usa_pix'),
            'Usa_eBanking': data.get('Usa_eBanking'),
            'Usa_app_banco': data.get('Usa_app_banco'),
        }
        res = predict_sklearn(skl, client)
        label = "Vai aderir ao Open Finance" if res.get('prediction') == 1 else "Não vai aderir ao Open Finance"
        return {
            'prediction': int(res.get('prediction')),
            'probability': res.get('probability'),
            'label': label
        }

    # Caso contrário, usa Spark pipeline
    global _spark_model, _spark_model_checked
    if _spark_model is None and not _spark_model_checked:
        try:
            _spark_model = get_spark_model()
        except Exception:
            logger.exception('Falha ao tentar carregar Spark model')
            _spark_model = None
        finally:
            _spark_model_checked = True

    spark_model = _spark_model
    if spark_model is None:
        # Nenhum modelo disponível
        raise RuntimeError('Nenhum modelo disponível: configure SKLEARN_MODEL_PATH ou garanta MODEL_PATH com pipeline Spark')

    normalized_data = {
        "Faixa etária": data.get("Faixa_etaria") or data.get('Faixa etária'),
        "Estado": data.get("Estado"),
        "Sexo": data.get("Sexo"),
        "Ocupacao": data.get("Ocupacao"),
        "Escolaridade": data.get("Escolaridade"),
        "Gp renda": data.get("Gp_renda") or data.get('Gp renda'),
        "Tipo_da_conta": data.get("Tipo_da_conta"),
        "Gp score de crédito": data.get("Gp_score_de_credito") or data.get('Gp score de crédito'),
        "Gp limite do cartão": data.get("Gp_limite_do_cartao") or data.get('Gp limite do cartão'),
        "Tempo_conta_atv": data.get("Tempo_conta_atv"),
        "Outros_bancos": data.get("Outros_bancos"),
        "Emprestimo": data.get("Emprestimo"),
        "Financiamento": data.get("Financiamento"),
        "Cartao_de_credito": data.get("Cartao_de_credito"),
        "Usa_cheque": data.get("Usa_cheque"),
        "Atrasa_pag": data.get("Atrasa_pag"),
        "Investimentos": data.get("Investimentos"),
        "Usa_pix": data.get("Usa_pix"),
        "Usa_eBanking": data.get("Usa_eBanking"),
        "Usa_app_banco": data.get("Usa_app_banco"),
    }

    spark = get_spark()
    df = spark.createDataFrame([Row(**normalized_data)])
    out = spark_model.transform(df)

    row = out.select("prediction", "probability").first()

    return {
        "prediction": int(row.prediction),
        "probability": row.probability.toArray().tolist(),
        "label": "Vai aderir ao Open Finance" if row.prediction == 1 else "Não vai aderir ao Open Finance"
    }


def run_explanations(data: dict):
    """
    Retorna importâncias globais das features do modelo.
    """
    # Tenta sklearn primeiro
    skl = _get_sklearn_model_if_configured()
    if skl is not None:
        feature_names = skl.get('feature_columns', [])
        global_imp = compute_global_importance(skl, feature_names)
        return {
            "global_feature_importances": global_imp
        }
    
    # Usa Spark model (lazy)
    global _spark_model, _spark_model_checked, _feature_names_cache
    if _spark_model is None and not _spark_model_checked:
        try:
            _spark_model = get_spark_model()
        except Exception:
            logger.exception('Falha ao tentar carregar Spark model')
            _spark_model = None
        finally:
            _spark_model_checked = True

    model = _spark_model
    if model is None:
        raise RuntimeError('Nenhum modelo disponível: configure SKLEARN_MODEL_PATH ou garanta MODEL_PATH com pipeline Spark')

    # compute feature names lazily
    try:
        if not _feature_names_cache:
            _feature_names_cache = extract_assembler_input_cols(model)
    except Exception:
        logger.exception('Falha ao extrair feature names do modelo Spark')
        _feature_names_cache = []

    global_imp = compute_global_importance(model, _feature_names_cache)
    return {
        "global_feature_importances": global_imp
    }
