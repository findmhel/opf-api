# Fix: Erro "Found array with 0 feature(s)" no sklearn model

## Problema

O erro ocorre quando `SKLEARN_MODEL_PATH` está configurado mas o modelo não contém as chaves necessárias:

- `categorical_cols`
- `numeric_cols`

Isso faz com que o `predict_sklearn()` receba listas vazias e não consiga construir o vetor de features.

## Solução Rápida (Recomendado)

**No Render Dashboard:**

1. Acesse o serviço `opf-api`
2. Vá em **Environment**
3. **Remova ou comente** a variável `SKLEARN_MODEL_PATH`
4. Salve e aguarde o redeploy

Isso forçará o uso do modelo Spark que funciona corretamente.

## Solução Completa: Retreinar o modelo sklearn

Se você quiser usar sklearn em produção, precisa retreinar o modelo corretamente:

```bash
cd /Users/edsonsilva/project/personal/opf-api
python scripts/train_sklearn.py
```

Verifique que o script `train_sklearn.py` salva o modelo com todas as chaves:

```python
model_obj = {
    'model': rf,
    'feature_columns': feature_cols,
    'encoders': label_encoders,
    'categorical_cols': CATEGORICAL_COLS,  # ← DEVE ESTAR PRESENTE
    'numeric_cols': NUMERIC_COLS,          # ← DEVE ESTAR PRESENTE
}
joblib.dump(model_obj, 'app/model/sk_model.joblib')
```

Depois faça commit e push do novo modelo.

## Verificação

Para verificar se o modelo tem as chaves corretas:

```python
import joblib
obj = joblib.load('app/model/sk_model.joblib')
print('categorical_cols:', obj.get('categorical_cols'))
print('numeric_cols:', obj.get('numeric_cols'))
```

Ambos devem retornar listas não vazias.
