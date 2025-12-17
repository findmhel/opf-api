"""
Script para testar diferentes configurações e encontrar a melhor performance.
"""
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
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

def preprocess(df: pd.DataFrame):
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

def test_config(name, X_train, X_test, y_train, y_test, **params):
    print(f'\n{"="*60}')
    print(f'Testando: {name}')
    print(f'{"="*60}')
    
    model = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, preds)
    erro = 1 - acc
    
    print(f'Acurácia: {acc:.4f} ({acc*100:.2f}%)')
    print(f'Taxa de erro: {erro:.4f} ({erro*100:.2f}%)')
    print(f'ROC AUC: {auc:.4f}')
    print(f'F1-Score: {f1:.4f}')
    
    return {
        'name': name,
        'accuracy': acc,
        'error_rate': erro,
        'auc': auc,
        'f1': f1,
        'params': params,
        'model': model
    }

def main():
    print('Carregando dados...')
    df = pd.read_parquet(INPUT_PARQUET)
    df_proc, encoders = preprocess(df)
    
    feature_cols = [c + "_idx" for c in CATEGORICAL_COLS if c in df.columns] + \
                   [c for c in NUMERIC_COLS if c in df.columns]
    
    X = df_proc[feature_cols]
    y = df_proc[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    configs = [
        # Config 1: Modelo Spark básico
        ('Spark Base (300 trees, depth 10)', {
            'n_estimators': 300,
            'max_depth': 10
        }),
        
        # Config 2: Com balanceamento
        ('Com class_weight balanced', {
            'n_estimators': 300,
            'max_depth': 10,
            'class_weight': 'balanced'
        }),
        
        # Config 3: Mais árvores
        ('Mais árvores (500)', {
            'n_estimators': 500,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2
        }),
        
        # Config 4: Mais profundidade
        ('Mais profundidade (depth 15)', {
            'n_estimators': 300,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2
        }),
        
        # Config 5: Árvores mais profundas
        ('Depth 20 + regularização', {
            'n_estimators': 300,
            'max_depth': 20,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            'max_features': 'sqrt'
        }),
        
        # Config 6: Combinação ótima
        ('Otimizado (500 trees, depth 15)', {
            'n_estimators': 500,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt'
        }),
    ]
    
    results = []
    for name, params in configs:
        result = test_config(name, X_train, X_test, y_train, y_test, **params)
        results.append(result)
    
    # Resumo final
    print(f'\n\n{"="*80}')
    print('RESUMO COMPARATIVO')
    print(f'{"="*80}')
    print(f'{"Configuração":<40} {"Acurácia":<12} {"Erro":<12} {"AUC":<12}')
    print(f'{"-"*80}')
    
    for r in sorted(results, key=lambda x: x['error_rate']):
        print(f'{r["name"]:<40} {r["accuracy"]*100:>6.2f}%     {r["error_rate"]*100:>6.2f}%     {r["auc"]:>6.4f}')
    
    # Melhor modelo
    best = min(results, key=lambda x: x['error_rate'])
    print(f'\n🏆 MELHOR CONFIGURAÇÃO: {best["name"]}')
    print(f'   Taxa de erro: {best["error_rate"]*100:.2f}%')
    print(f'   Acurácia: {best["accuracy"]*100:.2f}%')
    print(f'   AUC: {best["auc"]:.4f}')
    
    # Salvar melhor modelo
    save = input('\n\nSalvar melhor modelo como app/model/sk_model.joblib? (s/n): ')
    if save.lower() == 's':
        os.makedirs('app/model', exist_ok=True)
        model_obj = {
            'model': best['model'],
            'encoders': encoders,
            'categorical_cols': CATEGORICAL_COLS,
            'numeric_cols': NUMERIC_COLS,
            'feature_columns': feature_cols
        }
        joblib.dump(model_obj, 'app/model/sk_model.joblib')
        print('✓ Modelo salvo!')

if __name__ == '__main__':
    main()
