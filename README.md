# API de Predição - Open Finance

API para predição de adesão ao Open Finance usando Random Forest com PySpark.

## Estrutura do Projeto

```
opf-api/
├── train_model.py          # Script para treinar o modelo
├── run.py                  # Inicializa a API FastAPI
├── app/
│   ├── routes/            # Endpoints da API
│   ├── services/          # Lógica de negócio (predições)
│   ├── schemas/           # Validação de dados (Pydantic)
│   └── utils/             # Utilitários (explicações, features)
├── model/                 # Modelo treinado (PipelineModel)
└── metadata/              # Metadados do modelo (JSON)
```

## Workflow

### 1. Treinar o Modelo

Execute o script de treinamento para gerar o modelo a partir dos dados:

```bash
python train_model.py --input dataset_processado_opf.parquet --output model/modelo_openfinance_rf
```

Este script irá:

- Carregar os dados do arquivo parquet
- Criar o pipeline com StringIndexers + VectorAssembler + RandomForest
- Treinar o modelo (80/20 split)
- Avaliar métricas (AUC, Acurácia, F1)
- Salvar o modelo em `model/modelo_openfinance_rf`
- Salvar metadados em `metadata/metadata.json`

### 2. Iniciar a API

Após treinar o modelo:

```bash
python run.py
```

A API estará disponível em `http://localhost:8000`

### 3. Fazer Predições

Envie uma requisição POST para `/predict/` com os dados do usuário:

**Exemplo de payload:**

```json
{
  "Faixa_etaria": "25-34",
  "Estado": "SP",
  "Sexo": "M",
  "Ocupacao": "CLT",
  "Escolaridade": "Superior",
  "Gp_renda": "Média",
  "Tipo_da_conta": "Digital",
  "Gp_score_de_credito": "Alto",
  "Gp_limite_do_cartao": "Baixo",
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
```

**Resposta:**

```json
{
  "prediction": 1,
  "probability": [0.23, 0.77],
  "label": "Vai aderir ao Open Finance",
  "global_feature_importances": {
    "Usa_pix": 0.15,
    "Gp_renda_idx": 0.12,
    ...
  }
}
```

## Features do Modelo

### Variáveis Categóricas

- `Faixa_etaria`: Faixa etária do cliente
- `Estado`: Estado (UF)
- `Sexo`: M ou F
- `Ocupacao`: Tipo de ocupação (CLT, Autônomo, etc.)
- `Escolaridade`: Nível de escolaridade
- `Gp_renda`: Grupo de renda (Alta, Média, Baixa)
- `Tipo_da_conta`: Tipo de conta (Digital, Tradicional)
- `Gp_score_de_credito`: Grupo de score de crédito
- `Gp_limite_do_cartao`: Grupo de limite do cartão
- `Tempo_conta_atv`: Tempo de conta ativa

### Variáveis Numéricas (Binárias: 0 ou 1)

- `Outros_bancos`: Possui conta em outros bancos
- `Emprestimo`: Possui empréstimo
- `Financiamento`: Possui financiamento
- `Cartao_de_credito`: Possui cartão de crédito
- `Usa_cheque`: Usa cheque
- `Atrasa_pag`: Atrasa pagamentos
- `Investimentos`: Possui investimentos
- `Usa_pix`: Usa PIX
- `Usa_eBanking`: Usa internet banking
- `Usa_app_banco`: Usa app do banco

## Documentação da API

Acesse a documentação interativa em:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Notas importantes sobre produção

- **Não** inclua datasets grandes ou modelos pesados diretamente no repositório (ver `.gitignore`).
- Armazene modelos/dados grandes em um storage (S3/GCS/Blob). No startup da API, faça o download para um diretório temporário ou monte um volume.
- Para configurar variáveis de ambiente sensíveis (DB, S3 credentials), use secrets do provedor de deploy (ex.: GitHub Actions secrets, variables no Heroku/Render/Cloud Run).

### Rodando em produção com Docker (exemplo)

Dockerfile (exemplo simplificado):

```Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . .
# Download model from S3 if necessário (script ou entrypoint pode fazer isso)
CMD ["python", "run.py"]
```

No pipeline de CI, preencha os secrets e, se necessário, baixe o modelo do storage antes de iniciar a API.
