"""
Script para exportar features transformadas com Spark para CSV.
Altere `FEATURE_COLUMNS` para as colunas finais que seu modelo espera.
"""
from pyspark.sql import SparkSession
import os
from pathlib import Path
from pyspark.ml import PipelineModel
from app.config import MODEL_PATH

FEATURE_COLUMNS = [
    # substitua pelos nomes das colunas de features finais
    'Estado',
    'Faixa etária',
    'Sexo',
    'Escolaridade',
    'Renda',
    'Gp renda',
    'Gp gasto mensal',
    'Gp score de crédito',
    'Adesao_ao_OPF',
    'Usa_pix',
    'Usa_eBanking',
    'Usa_app_banco',
]

OUT_CSV = os.environ.get('OUT_CSV', 'data/features.csv')


def main():
    spark = SparkSession.builder.appName('export_features').getOrCreate()
    df = spark.read.parquet('dataset_processado_opf.parquet')

    model_path = Path(MODEL_PATH)
    if model_path.exists():
        try:
            pipeline = PipelineModel.load(str(model_path))
            transformed = pipeline.transform(df)
            # Tentaremos exportar as colunas de features já preparadas
            existing = [c for c in FEATURE_COLUMNS if c in transformed.columns]
            if not existing:
                # fallback para colunas originais do df
                existing = [c for c in FEATURE_COLUMNS if c in df.columns]
            pdf = transformed.select(*existing).toPandas()
            os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
            pdf.to_csv(OUT_CSV, index=False)
            print(f'Features transformadas salvas em {OUT_CSV}')
            return
        except Exception as e:
            print(f'Falha ao aplicar pipeline: {e}; fallback para seleção de colunas brutas')

    # Caso não haja pipeline, seleciona colunas brutas do parquet
    existing = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not existing:
        raise RuntimeError('Nenhuma das FEATURE_COLUMNS encontradas no DataFrame')

    pdf = df.select(*existing).toPandas()
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    pdf.to_csv(OUT_CSV, index=False)
    print(f'Features salvas em {OUT_CSV}')


if __name__ == '__main__':
    main()
