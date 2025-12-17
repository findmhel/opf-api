"""
Script para buscar configuração que chegue próximo de 20% de erro.
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer
from sklearn.preprocessing import LabelEncoder
import joblib

INPUT_PARQUET = 'dataset_processado_opf.parquet'
TARGET = 'Adesao_ao_OPF'

CATEGORICAL_COLS = [
    "Faixa etária", "Estado", "Sexo", "Ocupacao", "Escolaridade",
    "Gp renda", "Tipo_da_conta", "Gp score de crédito",
    "Gp limite do cartão", "Tempo_conta_atv"
]

NUMERIC_COLS = [
    "Outros_bancos", "Emprestimo", "Financiamento", "Cartao_de_credito",
    "Usa_cheque", "Atrasa_pag", "Investimentos", "Usa_pix",
    "Usa_eBanking", "Usa_app_banco"
]

def preprocess(df):
    df = df.copy()
    df = df.dropna(subset=[TARGET])
    
    encoders = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = df[col].fillna('_MISSING_')
            df[col + "_idx"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    return df, encoders

print('Carregando dados...')
df = pd.read_parquet(INPUT_PARQUET)
df_proc, encoders = preprocess(df)

feature_cols = [c + "_idx" for c in CATEGORICAL_COLS if c in df.columns] + \
               [c for c in NUMERIC_COLS if c in df.columns]

X = df_proc[feature_cols]
y = df_proc[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Testando configurações avançadas...\n')

# Grid Search para encontrar melhores parâmetros
param_grid = {
    'n_estimators': [500, 800],
    'max_depth': [20, 25, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

print('Executando Grid Search (isso pode demorar)...')
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=3,
    scoring='roc_auc',
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f'\n✓ Melhores parâmetros encontrados:')
for param, value in grid_search.best_params_.items():
    print(f'  {param}: {value}')

# Testar melhor modelo
best_model = grid_search.best_estimator_
preds = best_model.predict(X_test)
proba = best_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, preds)
auc = roc_auc_score(y_test, proba)
erro = 1 - acc

print(f'\n{"="*60}')
print('RESULTADO DO MELHOR MODELO')
print(f'{"="*60}')
print(f'Acurácia: {acc:.4f} ({acc*100:.2f}%)')
print(f'Taxa de erro: {erro:.4f} ({erro*100:.2f}%)')
print(f'ROC AUC: {auc:.4f}')

if erro <= 0.20:
    print(f'\n🎉 META ATINGIDA! Erro ≤ 20%')
else:
    print(f'\n⚠️  Erro ainda acima de 20%. Diferença: {(erro-0.20)*100:.2f}%')

save = input('\nSalvar este modelo? (s/n): ')
if save.lower() == 's':
    model_obj = {
        'model': best_model,
        'encoders': encoders,
        'categorical_cols': CATEGORICAL_COLS,
        'numeric_cols': NUMERIC_COLS,
        'feature_columns': feature_cols
    }
    joblib.dump(model_obj, 'app/model/sk_model.joblib')
    print('✓ Modelo salvo em app/model/sk_model.joblib')
