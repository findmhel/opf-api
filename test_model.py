"""
Exemplo de uso do modelo treinado para fazer predições.
Este script demonstra como carregar e usar o modelo fora da API.
"""

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

def test_model():
    """Testa o modelo treinado com uma amostra de exemplo."""
    
    # Cria sessão Spark
    spark = SparkSession.builder.appName("TestModel").getOrCreate()
    
    # Carrega o modelo
    print("Carregando modelo...")
    model = PipelineModel.load("model/modelo_openfinance_rf")
    
    # Cria uma amostra de teste
    nova_amostra = spark.createDataFrame([
        {
            "Faixa etária": "25-34",
            "Estado": "SP",
            "Sexo": "M",
            "Ocupacao": "CLT",
            "Escolaridade": "Superior",
            "Gp renda": "Média",
            "Tipo_da_conta": "Digital",
            "Gp score de crédito": "Alto",
            "Gp limite do cartão": "Baixo",
            "Tempo_conta_atv": "2-5 anos",
            "Outros_bancos": 1,
            "Emprestimo": 0,
            "Financiamento": 0,
            "Cartao_de_credito": 1,
            "Usa_cheque": 0,
            "Atrasa_pag": 0,
            "Investimentos": 1,
            "Usa_pix": 1,
            "Usa_eBanking": 1,
            "Usa_app_banco": 1
        }
    ])
    
    print("\nDados de entrada:")
    nova_amostra.show(truncate=False)
    
    # Faz a predição
    print("\nFazendo predição...")
    resultado = model.transform(nova_amostra)
    
    # Mostra o resultado
    print("\nResultado da predição:")
    resultado.select("prediction", "probability").show(truncate=False)
    
    # Interpreta o resultado
    pred = resultado.select("prediction").first()[0]
    prob = resultado.select("probability").first()[0]
    
    print(f"\n{'='*60}")
    print(f"Predição: {int(pred)}")
    print(f"Probabilidade [Não aderir, Aderir]: [{prob[0]:.2%}, {prob[1]:.2%}]")
    
    if pred == 1:
        print("Resultado: O cliente PROVAVELMENTE vai aderir ao Open Finance")
    else:
        print("Resultado: O cliente PROVAVELMENTE NÃO vai aderir ao Open Finance")
    print(f"{'='*60}")
    
    spark.stop()


if __name__ == "__main__":
    test_model()
