"""
Endpoint para servir dados do dataset processado
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from app.spark_session import spark
from pyspark.sql.functions import col, sum as spark_sum
import json
import logging
import traceback
import numpy as np
import pandas as pd
import datetime

router = APIRouter()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_json_serializable(value):
    # Recursive conversion to JSON-serializable Python primitives
    # dicts and lists are processed recursively; numpy/pandas types and datetimes are converted
    if value is None:
        return None
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)):
        # Python native numeric types are fine
        return value
    # numpy scalar types
    if isinstance(value, (np.generic,)):
        try:
            return value.item()
        except Exception:
            return float(value)
    # numpy arrays or pandas Series/Index -> convert to list recursively
    if isinstance(value, (np.ndarray, list, tuple, set, pd.Series, pd.Index)):
        try:
            seq = list(value)
        except Exception:
            try:
                seq = value.tolist()
            except Exception:
                seq = [make_json_serializable(v) for v in value]
        return [make_json_serializable(v) for v in seq]

    # pandas / numpy NaT or scalar NA (guard against array-like raising)
    try:
        if pd.isna(value):
            return None
    except Exception:
        # If pd.isna raised because value is array-like, try to handle as iterable below
        pass
    # pandas Timestamp or datetime
    if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date)):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    # bytes -> decode as utf-8 if possible
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode('utf-8')
        except Exception:
            return list(value)
    # dict -> convert keys to str and recurse
    if isinstance(value, dict):
        return {str(k): make_json_serializable(v) for k, v in value.items()}
    # Generic iterable (but not string/bytes/dict) -> convert to list and recurse
    if hasattr(value, '__iter__') and not isinstance(value, (str, bytes, bytearray, dict)):
        try:
            seq = list(value)
        except Exception:
            try:
                seq = value.tolist()
            except Exception:
                seq = None
        if seq is not None:
            return [make_json_serializable(v) for v in seq]
    # list/tuple/set -> recurse
    if isinstance(value, (list, tuple, set)):
        return [make_json_serializable(v) for v in value]
    # fallback: try to use __dict__ or string conversion
    try:
        if hasattr(value, '__dict__'):
            return {str(k): make_json_serializable(v) for k, v in vars(value).items()}
    except Exception:
        pass
    return str(value)

@router.get("/clients")
def get_clients_data(limit: int = 1000, offset: int = 0):
    """
    Retorna dados dos clientes do dataset processado.

    Args:
        limit: Número máximo de registros (padrão: 1000)

    Returns:
        Lista de clientes com suas informações
    """
    try:
        # Carregar dados do parquet
        df = spark.read.parquet("dataset_processado_opf.parquet")

        # Limitar número de registros (coletar até offset+limit e depois fatiar)
        end = offset + limit
        df_limited = df.limit(end)

        # Selecionar apenas colunas necessárias para o frontend
        needed_cols = [
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

        # Filtrar colunas existentes (caso algumas não estejam no parquet)
        existing_cols = [c for c in needed_cols if c in df.columns]

        # Coletar linhas limitadas e converter cada Row para dicionário JSON-serializável
        rows = df_limited.select(*existing_cols).collect()
        # Aplicar offset localmente após coletar
        sliced = rows[offset:end]
        data = []
        for row in sliced:
            rec = row.asDict(recursive=True)
            rec_js = make_json_serializable(rec)
            # Provide underscore-normalized keys to help frontends using snake-like keys
            # e.g., front might expect 'Gp_renda' instead of 'Gp renda'
            def add_underscore_keys(d: dict):
                mapping = {}
                for k, v in d.items():
                    underscored = k.replace(' ', '_')
                    mapping[underscored] = v
                d.update(mapping)
                return d

            if isinstance(rec_js, dict):
                rec_js = add_underscore_keys(rec_js)

            data.append(rec_js)

        # Calcular estatísticas
        total = df.count()
        aderiu = df.filter(col("Adesao_ao_OPF") == 1).count()
        nao_aderiu = total - aderiu

        payload = {
            "success": True,
            "total_records": int(total),
            "returned_records": int(len(data)),
            "statistics": {
                "total": int(total),
                "aderiu": int(aderiu),
                "nao_aderiu": int(nao_aderiu),
                "taxa_adesao": round((aderiu / total) * 100, 2) if total > 0 else 0
            },
            "data": data
        }
        return JSONResponse(content=payload)
    except FileNotFoundError:
        logger.exception("Parquet file not found")
        raise HTTPException(
            status_code=404,
            detail="Arquivo dataset_processado_opf.parquet não encontrado"
        )
    except Exception as e:
        # Log completo para diagnóstico
        logger.error("Erro no endpoint /clients: %s", traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar dados: {str(e)}"
        )


@router.get("/stats")
def get_statistics():
    """
    Retorna estatísticas agregadas do dataset.
    """
    try:
        df = spark.read.parquet("dataset_processado_opf.parquet")

        # Estatísticas gerais
        total = df.count()
        aderiu = df.filter(col("Adesao_ao_OPF") == 1).count()
        
        # Agrupar por estado com informações de adesão
        estados_df = df.groupBy("Estado").agg(
            spark_sum(col("Adesao_ao_OPF").cast("int")).alias("aderiu"),
            (spark_sum((1 - col("Adesao_ao_OPF")).cast("int"))).alias("nao_aderiu")
        ).toPandas()

        # Adicionar count total
        estados_df['count'] = estados_df['aderiu'] + estados_df['nao_aderiu']
        estados_df = estados_df.where(estados_df.notna(), None)

        # Converter para lista de dicionários com tipos JSON-serializáveis
        por_estado = [make_json_serializable(rec) for rec in estados_df.to_dict(orient='records')]

        # Agrupar por faixa etária
        faixa_etaria_df = df.groupBy("Faixa etária").count().toPandas()
        faixa_etaria_df = faixa_etaria_df.where(faixa_etaria_df.notna(), None)
        por_faixa_etaria = [make_json_serializable(rec) for rec in faixa_etaria_df.to_dict(orient='records')]

        # Adesão por grupo de renda
        renda_df = df.groupBy("Gp renda", "Adesao_ao_OPF").count().toPandas()
        renda_df = renda_df.where(renda_df.notna(), None)
        por_renda = [make_json_serializable(rec) for rec in renda_df.to_dict(orient='records')]

        payload = {
            "success": True,
            "geral": {
                "total_clientes": int(total),
                "aderiu": int(aderiu),
                "nao_aderiu": int(total - aderiu),
                "taxa_adesao": round((aderiu / total) * 100, 2)
            },
            "por_estado": por_estado,
            "por_faixa_etaria": por_faixa_etaria,
            "por_renda": por_renda
        }
        return JSONResponse(content=payload)
    except Exception as e:
        logger.error("Erro no endpoint /stats: %s", traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao calcular estatísticas: {str(e)}"
        )
