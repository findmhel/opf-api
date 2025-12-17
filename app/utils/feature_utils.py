def extract_assembler_input_cols(pipeline_model):
    """Extrai as colunas de input do VectorAssembler no pipeline."""
    # import local para evitar importar pyspark no tempo de import do pacote
    from pyspark.ml.feature import VectorAssembler

    for stage in pipeline_model.stages:
        if isinstance(stage, VectorAssembler):
            return list(stage.getInputCols())
    raise RuntimeError("VectorAssembler não encontrado no pipeline.")
