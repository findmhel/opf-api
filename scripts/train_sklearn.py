"""
Treina um modelo scikit-learn a partir do parquet original usando mesma transformação do Spark.
Uso:
  OUT_MODEL=app/model/sk_model.joblib python scripts/train_sklearn.py
"""
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
import joblib

INPUT_PARQUET = os.environ.get('INPUT_PARQUET', 'dataset_processado_opf.parquet')
OUT_MODEL = os.environ.get('OUT_MODEL', 'app/model/sk_model.joblib')
TARGET = 'Adesao_ao_OPF'

# Mesmas colunas do modelo Spark
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

def preprocess(df: pd.DataFrame):
    """Aplica mesma transformação que o modelo Spark: LabelEncoder para categóricas."""
    df = df.copy()
    
    # Remove linhas com target nulo
    df = df.dropna(subset=[TARGET])
    
    # Cria encoders para cada coluna categórica
    encoders = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            # Preenche NaN com string vazia para encoding
            df[col] = df[col].fillna('_MISSING_')
            df[col + "_idx"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    
    # Preenche NaN nas colunas numéricas com 0
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    return df, encoders

def main():
    # Lê o parquet original
    df = pd.read_parquet(INPUT_PARQUET)
    
    if TARGET not in df.columns:
        raise RuntimeError(f'Target {TARGET} not in {INPUT_PARQUET}')
    
    # Aplica preprocessamento
    df_proc, encoders = preprocess(df)
    
    # Monta feature matrix
    feature_cols = [c + "_idx" for c in CATEGORICAL_COLS if c in df.columns] + \
                   [c for c in NUMERIC_COLS if c in df.columns]
    
    X = df_proc[feature_cols]
    y = df_proc[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Treina modelo otimizado para ~80% acurácia (20% erro)
    print('Treinando modelo otimizado...')
    model = RandomForestClassifier(
        n_estimators=800,           # Muito mais árvores
        max_depth=25,               # Mais profundidade
        min_samples_split=2,        # Permite splits mais granulares
        min_samples_leaf=1,         # Folhas mais específicas
        max_features='sqrt',        # Features por split
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Predições no conjunto de teste
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    
    # Métricas detalhadas
    print('\n' + '='*60)
    print('MÉTRICAS DE PERFORMANCE DO MODELO')
    print('='*60)
    
    print(f'\nConjunto de treino: {len(X_train)} amostras')
    print(f'Conjunto de teste: {len(X_test)} amostras')
    print(f'Distribuição target (teste): {dict(y_test.value_counts())}')
    
    print('\n--- Métricas Gerais ---')
    print(f'Acurácia: {accuracy_score(y_test, preds):.4f}')
    print(f'Precisão: {precision_score(y_test, preds):.4f}')
    print(f'Recall: {recall_score(y_test, preds):.4f}')
    print(f'F1-Score: {f1_score(y_test, preds):.4f}')
    print(f'ROC AUC: {roc_auc_score(y_test, proba):.4f}')
    
    print('\n--- Matriz de Confusão ---')
    cm = confusion_matrix(y_test, preds)
    print(f'True Negatives (TN):  {cm[0][0]:>6}')
    print(f'False Positives (FP): {cm[0][1]:>6}')
    print(f'False Negatives (FN): {cm[1][0]:>6}')
    print(f'True Positives (TP):  {cm[1][1]:>6}')
    
    # Taxa de erro
    total = len(y_test)
    erros = (preds != y_test).sum()
    taxa_erro = erros / total
    print(f'\n--- Taxa de Erro ---')
    print(f'Total de predições: {total}')
    print(f'Predições erradas: {erros}')
    print(f'Taxa de erro: {taxa_erro:.2%}')
    print(f'Taxa de acerto: {(1-taxa_erro):.2%}')
    
    print('\n--- Relatório por Classe ---')
    print(classification_report(y_test, preds, target_names=['Não adere', 'Adere']))
    
    print('='*60)
    
    # Salva modelo + encoders + metadados
    os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
    model_obj = {
        'model': model,
        'encoders': encoders,
        'categorical_cols': CATEGORICAL_COLS,
        'numeric_cols': NUMERIC_COLS,
        'feature_columns': feature_cols
    }
    joblib.dump(model_obj, OUT_MODEL)
    print(f'Model salvo em {OUT_MODEL} com {len(feature_cols)} features')
    print(f'Features: {feature_cols}')

if __name__ == '__main__':
    main()
