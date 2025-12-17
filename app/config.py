import os

# Caminho padrão (relativo ao repositório). Pode ser sobrescrito pela variável de ambiente MODEL_PATH
MODEL_PATH = os.environ.get('MODEL_PATH', "model/modelo_openfinance_rf")
METADATA_PATH = os.environ.get('METADATA_PATH', "metadata/metadata.json")

