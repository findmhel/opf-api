"""
Script para treinar o modelo de predição de adesão ao Open Finance.
Execute este script separadamente para gerar o modelo que a API utilizará.

Uso:
    python train_model.py --input dataset_processado_opf.parquet --output model/modelo_openfinance_rf
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import argparse
import json
import os


def create_spark_session():
    """Cria uma sessão Spark."""
    return SparkSession.builder.appName("OpenFinanceModel").getOrCreate()


def load_data(spark, input_path):
    """Carrega e prepara os dados."""
    df = spark.read.parquet(input_path)
    
    # Renomeia a coluna alvo para "label"
    df = df.withColumn("label", col("Adesao_ao_OPF").cast("double"))
    df = df.drop("Adesao_ao_OPF")
    
    return df


def create_pipeline():
    """Cria o pipeline de ML com todas as transformações."""
    
    # Definição das colunas categóricas
    categorical_cols = [
        "Faixa etária", "Estado", "Sexo", "Ocupacao", "Escolaridade",
        "Gp renda", "Tipo_da_conta", "Gp score de crédito",
        "Gp limite do cartão", "Tempo_conta_atv"
    ]
    
    # Definição das colunas numéricas
    numeric_cols = [
        "Outros_bancos", "Emprestimo", "Financiamento", "Cartao_de_credito",
        "Usa_cheque", "Atrasa_pag", "Investimentos", "Usa_pix",
        "Usa_eBanking", "Usa_app_banco"
    ]
    
    # Indexadores para variáveis categóricas
    indexers = [
        StringIndexer(inputCol=col, outputCol=col + "_idx", handleInvalid="keep")
        for col in categorical_cols
    ]
    
    # Features finais (categóricas indexadas + numéricas)
    final_features = [c + "_idx" for c in categorical_cols] + numeric_cols
    
    # Assembler para criar o vetor de features
    assembler = VectorAssembler(
        inputCols=final_features,
        outputCol="features"
    )
    
    # Modelo Random Forest
    rf = RandomForestClassifier(
        labelCol="label",
        featuresCol="features",
        numTrees=300,
        maxDepth=10,
        maxBins=200,
        seed=42
    )
    
    # Pipeline completo
    pipeline = Pipeline(stages=indexers + [assembler, rf])
    
    return pipeline, categorical_cols, numeric_cols, final_features


def train_model(df, pipeline):
    """Treina o modelo com divisão treino/teste."""
    train, test = df.randomSplit([0.8, 0.2], seed=42)
    
    print(f"Treinando modelo com {train.count()} amostras de treino...")
    model = pipeline.fit(train)
    
    print("Avaliando modelo...")
    pred = model.transform(test)
    
    # Métricas de avaliação
    evaluator_auc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
    evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", metricName="f1")
    
    auc = evaluator_auc.evaluate(pred)
    acc = evaluator_acc.evaluate(pred)
    f1 = evaluator_f1.evaluate(pred)
    
    print(f"\n=== Métricas do Modelo ===")
    print(f"AUC: {auc:.4f}")
    print(f"Acurácia: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return model, {"auc": auc, "accuracy": acc, "f1": f1}


def save_model_and_metadata(model, output_path, categorical_cols, numeric_cols, final_features, metrics):
    """Salva o modelo e metadados."""
    
    # Salva o modelo
    print(f"\nSalvando modelo em {output_path}...")
    model.write().overwrite().save(output_path)
    
    # Cria metadados
    metadata = {
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "final_features": final_features,
        "metrics": metrics,
        "model_type": "RandomForestClassifier",
        "num_trees": 300,
        "max_depth": 10
    }
    
    # Salva metadados
    metadata_dir = os.path.dirname(output_path).replace("model", "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    metadata_path = os.path.join(metadata_dir, "metadata.json")
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadados salvos em {metadata_path}")
    print("\nModelo treinado e exportado com sucesso!")


def main():
    parser = argparse.ArgumentParser(description="Treinar modelo de Open Finance")
    parser.add_argument("--input", default="dataset_processado_opf.parquet", 
                       help="Caminho para o arquivo parquet de entrada")
    parser.add_argument("--output", default="model/modelo_openfinance_rf",
                       help="Caminho para salvar o modelo treinado")
    
    args = parser.parse_args()
    
    # Inicializa Spark
    spark = create_spark_session()
    
    # Carrega dados
    print(f"Carregando dados de {args.input}...")
    df = load_data(spark, args.input)
    print(f"Dataset carregado: {df.count()} registros")
    
    # Cria pipeline
    pipeline, categorical_cols, numeric_cols, final_features = create_pipeline()
    
    # Treina modelo
    model, metrics = train_model(df, pipeline)
    
    # Salva modelo e metadados
    save_model_and_metadata(model, args.output, categorical_cols, numeric_cols, 
                           final_features, metrics)
    
    spark.stop()


if __name__ == "__main__":
    main()
