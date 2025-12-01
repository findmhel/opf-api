from pyspark.sql import Row
from app.model_loader import load_sklearn_model, predict_sklearn, get_spark_model, load_model_from
from app.spark_session import spark
from app.utils.feature_utils import extract_assembler_input_cols
from app.utils.explanations import compute_global_importance
import os
from pathlib import Path


_spark_model = get_spark_model()
feature_names = []
if _spark_model is not None:
    try:
        feature_names = extract_assembler_input_cols(_spark_model)
    except Exception:
        feature_names = []

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
    if not p.exists():
        return None

    if _sklearn_cache['loaded'] and _sklearn_cache['path'] == str(p):
        return _sklearn_cache['obj']

    obj = load_sklearn_model(str(p))
    _sklearn_cache.update({'loaded': True, 'obj': obj, 'path': str(p)})
    return obj


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
    spark_model = _spark_model or get_spark_model()
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
        model = skl
    else:
        # Usa Spark model
        model = _spark_model or get_spark_model()
        if model is None:
            raise RuntimeError('Nenhum modelo disponível: configure SKLEARN_MODEL_PATH ou garanta MODEL_PATH com pipeline Spark')
    
    global_imp = compute_global_importance(model, feature_names)
    return {
        "global_feature_importances": global_imp
    }
