from pyspark.sql import Row
from app.model_loader import model, metadata
from app.spark_session import spark
from app.utils.feature_utils import extract_assembler_input_cols
from app.utils.explanations import compute_global_importance

feature_names = extract_assembler_input_cols(model)

def run_prediction(data: dict):
    """
    Executa predição usando o modelo treinado.
    
    Args:
        data: Dicionário com os dados do usuário (deve corresponder ao UserData schema)
    
    Returns:
        Dicionário com prediction (0 ou 1) e probability (lista com probabilidades)
    """
    # Normaliza nomes das colunas (remove underscores se necessário para match com modelo)
    normalized_data = {
        "Faixa etária": data.get("Faixa_etaria"),
        "Estado": data.get("Estado"),
        "Sexo": data.get("Sexo"),
        "Ocupacao": data.get("Ocupacao"),
        "Escolaridade": data.get("Escolaridade"),
        "Gp renda": data.get("Gp_renda"),
        "Tipo_da_conta": data.get("Tipo_da_conta"),
        "Gp score de crédito": data.get("Gp_score_de_credito"),
        "Gp limite do cartão": data.get("Gp_limite_do_cartao"),
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
    out = model.transform(df)

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
    global_imp = compute_global_importance(model, feature_names)
    return {
        "global_feature_importances": global_imp
    }
