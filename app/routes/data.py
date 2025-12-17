"""
Endpoint para servir dados do dataset processado
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from app.spark_session import get_spark
from pyspark.sql.functions import col, sum as spark_sum
import json
import logging
import traceback
import numpy as np
import pandas as pd
import datetime
import os
import os

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
        parquet_path = os.environ.get('PARQUET_PATH', 'dataset_processado_opf.parquet')
        csv_path = os.environ.get('PARQUET_FALLBACK_CSV', 'data/features.csv')

        # If parquet doesn't exist, try CSV fallback immediately
        if not os.path.exists(parquet_path):
            logger.warning("Parquet file not found at path: %s; trying CSV fallback", parquet_path)
            if os.path.exists(csv_path):
                logger.info('Using CSV fallback for clients: %s', csv_path)
                pdf = pd.read_csv(csv_path)
                # select only the needed columns that exist
                needed_cols = [
                    'Estado', 'Faixa etária', 'Sexo', 'Escolaridade', 'Renda', 'Gp renda',
                    'Gp gasto mensal', 'Gp score de crédito', 'Adesao_ao_OPF', 'Usa_pix',
                    'Usa_eBanking', 'Usa_app_banco'
                ]
                existing_cols = [c for c in needed_cols if c in pdf.columns]
                rows = pdf[existing_cols].iloc[offset:offset+limit].to_dict(orient='records')
                data = []
                for rec in rows:
                    rec_js = make_json_serializable(rec)
                    underscored = {k.replace(' ', '_'): v for k, v in rec_js.items()}
                    if isinstance(rec_js, dict):
                        rec_js.update(underscored)
                    data.append(rec_js)

                total = int(len(pdf))
                aderiu = int(pdf['Adesao_ao_OPF'].fillna(0).astype(int).sum()) if 'Adesao_ao_OPF' in pdf.columns else 0
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
            else:
                raise HTTPException(status_code=404, detail=f"Parquet e CSV fallback não encontrados no servidor")

        # try importing pyspark availability via get_spark (which lazy-imports pyspark)
        try:
            spark = get_spark()
            df = spark.read.parquet(parquet_path)
        except ModuleNotFoundError as e:
            logger.exception('PySpark not available in runtime: %s', str(e))
            # try CSV fallback
            csv_path = os.environ.get('PARQUET_FALLBACK_CSV', 'data/features.csv')
            if os.path.exists(csv_path):
                logger.info('Using CSV fallback for clients: %s', csv_path)
                pdf = pd.read_csv(csv_path)
                # select only the needed columns that exist
                needed_cols = [
                    'Estado', 'Faixa etária', 'Sexo', 'Escolaridade', 'Renda', 'Gp renda',
                    'Gp gasto mensal', 'Gp score de crédito', 'Adesao_ao_OPF', 'Usa_pix',
                    'Usa_eBanking', 'Usa_app_banco'
                ]
                existing_cols = [c for c in needed_cols if c in pdf.columns]
                rows = pdf[existing_cols].iloc[offset:offset+limit].to_dict(orient='records')
                data = []
                for rec in rows:
                    rec_js = make_json_serializable(rec)
                    underscored = {k.replace(' ', '_'): v for k, v in rec_js.items()}
                    if isinstance(rec_js, dict):
                        rec_js.update(underscored)
                    data.append(rec_js)

                total = int(len(pdf))
                aderiu = int(pdf['Adesao_ao_OPF'].fillna(0).astype(int).sum()) if 'Adesao_ao_OPF' in pdf.columns else 0
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
            raise HTTPException(status_code=503, detail="PySpark não está disponível neste ambiente e CSV fallback não encontrado.")

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
    except HTTPException:
        # re-raise HTTPException (404/503) raised above
        raise
    except Exception as e:
        # Log completo para diagnóstico com traceback e a carga do ambiente
        tb = traceback.format_exc()
        logger.error("Erro no endpoint /clients: %s", tb)
        # Provide a hint in the response to inspect server logs with a simple id
        err_id = abs(hash(tb)) % (10 ** 8)
        logger.error("Erro ID: %s", err_id)
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar dados (id={err_id}). Verifique os logs do servidor."
        )
    except Exception as e:
        # Log completo para diagnóstico com traceback e a carga do ambiente
        tb = traceback.format_exc()
        logger.error("Erro no endpoint /clients: %s", tb)
        # Provide a hint in the response to inspect server logs with a simple id
        err_id = abs(hash(tb)) % (10 ** 8)
        logger.error("Erro ID: %s", err_id)
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar dados (id={err_id}). Verifique os logs do servidor."
        )


@router.get("/stats")
def get_statistics():
    """
    Retorna estatísticas agregadas do dataset.
    """
    try:
        parquet_path = os.environ.get('PARQUET_PATH', 'dataset_processado_opf.parquet')
        csv_path = os.environ.get('PARQUET_FALLBACK_CSV', 'data/features.csv')

        # If parquet doesn't exist, try CSV fallback immediately
        if not os.path.exists(parquet_path):
            logger.warning("Parquet file not found at path: %s; trying CSV fallback", parquet_path)
            if os.path.exists(csv_path):
                logger.info('Using CSV fallback for stats: %s', csv_path)
                pdf = pd.read_csv(csv_path)
                total = int(len(pdf))
                aderiu = int(pdf['Adesao_ao_OPF'].fillna(0).astype(int).sum()) if 'Adesao_ao_OPF' in pdf.columns else 0
                nao_aderiu = total - aderiu

                por_estado = []
                if 'Estado' in pdf.columns:
                    grp = pdf.groupby('Estado')
                    for k, g in grp:
                        ader = int(g['Adesao_ao_OPF'].fillna(0).astype(int).sum()) if 'Adesao_ao_OPF' in g.columns else 0
                        nao = int(len(g) - ader)
                        por_estado.append({'Estado': k, 'aderiu': int(ader), 'nao_aderiu': int(nao), 'count': int(len(g))})

                # faixa etaria
                por_faixa_etaria = []
                if 'Faixa etária' in pdf.columns:
                    fe = pdf['Faixa etária'].value_counts().to_dict()
                    por_faixa_etaria = [{'Faixa etária': k, 'count': int(v)} for k, v in fe.items()]

                # renda
                por_renda = []
                if 'Gp renda' in pdf.columns and 'Adesao_ao_OPF' in pdf.columns:
                    grp = pdf.groupby(['Gp renda', 'Adesao_ao_OPF']).size().reset_index(name='count')
                    por_renda = grp.to_dict(orient='records')

                payload = {
                    "success": True,
                    "geral": {
                        "total_clientes": int(total),
                        "aderiu": int(aderiu),
                        "nao_aderiu": int(nao_aderiu),
                        "taxa_adesao": round((aderiu / total) * 100, 2) if total > 0 else 0
                    },
                    "por_estado": [make_json_serializable(r) for r in por_estado],
                    "por_faixa_etaria": [make_json_serializable(r) for r in por_faixa_etaria],
                    "por_renda": [make_json_serializable(r) for r in por_renda]
                }
                return JSONResponse(content=payload)
            else:
                raise HTTPException(status_code=404, detail=f"Parquet e CSV fallback não encontrados no servidor")

        try:
            spark = get_spark()
        except ModuleNotFoundError as e:
            logger.exception('PySpark not available in runtime: %s', str(e))
            raise HTTPException(status_code=503, detail="PySpark não está disponível neste ambiente. Configure SKLEARN_MODEL_PATH ou use uma imagem com Spark.")

        df = spark.read.parquet(parquet_path)

        # Estatísticas gerais
        total = df.count()
        aderiu = df.filter(col("Adesao_ao_OPF") == 1).count()
        
        # Agrupar por estado com informações de adesão
        try:
            estados_df = df.groupBy("Estado").agg(
                spark_sum(col("Adesao_ao_OPF").cast("int")).alias("aderiu"),
                (spark_sum((1 - col("Adesao_ao_OPF")).cast("int"))).alias("nao_aderiu")
            ).toPandas()
        except Exception:
            # If Spark operations fail, attempt CSV fallback
            csv_path = os.environ.get('PARQUET_FALLBACK_CSV', 'data/features.csv')
            if os.path.exists(csv_path):
                logger.info('Using CSV fallback for stats: %s', csv_path)
                pdf = pd.read_csv(csv_path)
                total = int(len(pdf))
                aderiu = int(pdf['Adesao_ao_OPF'].fillna(0).astype(int).sum()) if 'Adesao_ao_OPF' in pdf.columns else 0
                nao_aderiu = total - aderiu

                por_estado = []
                if 'Estado' in pdf.columns:
                    grp = pdf.groupby('Estado')
                    for k, g in grp:
                        ader = int(g['Adesao_ao_OPF'].fillna(0).astype(int).sum()) if 'Adesao_ao_OPF' in g.columns else 0
                        nao = int(len(g) - ader)
                        por_estado.append({'Estado': k, 'aderiu': int(ader), 'nao_aderiu': int(nao), 'count': int(len(g))})

                # faixa etaria
                por_faixa_etaria = []
                if 'Faixa etária' in pdf.columns:
                    fe = pdf['Faixa etária'].value_counts().to_dict()
                    por_faixa_etaria = [{'Faixa etária': k, 'count': int(v)} for k, v in fe.items()]

                # renda
                por_renda = []
                if 'Gp renda' in pdf.columns and 'Adesao_ao_OPF' in pdf.columns:
                    grp = pdf.groupby(['Gp renda', 'Adesao_ao_OPF']).size().reset_index(name='count')
                    por_renda = grp.to_dict(orient='records')

                payload = {
                    "success": True,
                    "geral": {
                        "total_clientes": int(total),
                        "aderiu": int(aderiu),
                        "nao_aderiu": int(nao_aderiu),
                        "taxa_adesao": round((aderiu / total) * 100, 2) if total > 0 else 0
                    },
                    "por_estado": [make_json_serializable(r) for r in por_estado],
                    "por_faixa_etaria": [make_json_serializable(r) for r in por_faixa_etaria],
                    "por_renda": [make_json_serializable(r) for r in por_renda]
                }
                return JSONResponse(content=payload)
            # if no CSV fallback, re-raise original exception
            raise

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
    except FileNotFoundError:
        logger.exception("Parquet file not found at path: %s", parquet_path)
        raise HTTPException(status_code=404, detail=f"Arquivo {parquet_path} não encontrado")
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Erro no endpoint /stats: %s", tb)
        err_id = abs(hash(tb)) % (10 ** 8)
        logger.error("Erro ID: %s", err_id)
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao calcular estatísticas (id={err_id}). Verifique os logs do servidor."
        )
