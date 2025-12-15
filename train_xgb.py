# -*- coding: utf-8 -*-
"""
Treinamento do modelo XGBoost para prever a tendência diária do IBOVESPA.
- Lê o arquivo CSV histórico (2021-08-17 a 2025-08-18)
- Constrói features técnicas simples (MM, Retorno, Volatilidade, RSI, MACD, Bandas de Bollinger)
- Define o alvo como tendência do próximo dia (1 = sobe, 0 = cai)
- Separa treino (todos menos os últimos 30 dias) e teste (últimos 30 dias)
- Treina um pipeline (StandardScaler + XGBClassifier)
- Salva modelo em pickle (model_xgb.pkl) e métricas em JSON (metrics.json)
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception as e:
    raise ImportError("xgboost não está instalado. Adicione 'xgboost' no requirements.txt e instale com pip.")

# ------------------------------
# Utilitários de feature engineering
# ------------------------------

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recebe DataFrame com coluna 'Último' (float) e retorna DF com features e alvo."""
    out = pd.DataFrame(index=df.index)
    out['close'] = df['Último']
    
    # Retorno percentual diário
    out['retorno_1d'] = out['close'].pct_change()

    # Médias móveis
    out['mm_5'] = out['close'].rolling(5).mean()
    out['mm_21'] = out['close'].rolling(21).mean()

    # Volatilidade (desvio padrão dos retornos)
    out['vol_5'] = out['retorno_1d'].rolling(5).std()

    # Momentum simples (diferença vs 5 dias atrás)
    out['momentum_5'] = out['close'] - out['close'].shift(5)

    # RSI (14) – cálculo clássico
    delta = out['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    out['rsi_14'] = 100 - (100 / (1 + rs))

    # MACD (12,26) + Sinal (9)
    ema12 = ema(out['close'], 12)
    ema26 = ema(out['close'], 26)
    macd = ema12 - ema26
    out['macd_12_26'] = macd
    out['macd_signal_9'] = ema(macd, 9)

    # Bandas de Bollinger (20, 2 std)
    ma20 = out['close'].rolling(20).mean()
    std20 = out['close'].rolling(20).std()
    out['bb_upper_20_2'] = ma20 + 2 * std20
    out['bb_lower_20_2'] = ma20 - 2 * std20

    # Alvo: tendência do próximo dia (próxima variação > 0)
    next_return = out['close'].pct_change().shift(-1)
    out['alvo'] = (next_return > 0).astype(int)

    # Remove NaNs iniciais
    out = out.dropna()
    return out

# ------------------------------
# Carregar dados e preparar
# ------------------------------
DATA_PATH = Path('Dados Históricos - Ibovespa 17.08.2021 - 18.08.2025 final.csv')
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Não encontrei o arquivo '{DATA_PATH}'. Certifique-se de colocá-lo no mesmo diretório.")

# O CSV tem separador vírgula e datas dia/mês/ano
raw = pd.read_csv(DATA_PATH, sep=',')

# Padroniza o nome das colunas (alguns arquivos vêm com aspas)
raw.columns = [c.replace('"','') for c in raw.columns]

# Converte Data
raw['Data'] = pd.to_datetime(raw['Data'], dayfirst=True, errors='coerce')
raw = raw.sort_values('Data').set_index('Data')

# Limpa colunas numéricas
# Mantemos apenas 'Último' e transformamos em float
def to_float(s):
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return (s.astype(str)
              .str.replace('.', '.', regex=False)
              .str.replace(',', '.', regex=False)
              .str.replace(r'[^0-9\.-]', '', regex=True)
              .astype(float))

df = pd.DataFrame({'Último': to_float(raw['Último'])})

# Constrói features
features_df = compute_features(df)

# Separa treino e teste (últimos 30 dias)
TEST_DAYS = 30
train_df = features_df.iloc[:-TEST_DAYS]
test_df = features_df.iloc[-TEST_DAYS:]

feature_cols = [
    'close','retorno_1d','mm_5','mm_21','vol_5','momentum_5',
    'rsi_14','macd_12_26','macd_signal_9','bb_upper_20_2','bb_lower_20_2'
]
X_train, y_train = train_df[feature_cols], train_df['alvo']
X_test, y_test = test_df[feature_cols], test_df['alvo']

# Pipeline: scaler + XGBClassifier
model = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', XGBClassifier(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=0.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        eval_metric='logloss'
    ))
])

# Treina
model.fit(X_train, y_train)

# Avalia
y_pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:,1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
fbeta = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred).tolist()
report = classification_report(y_test, y_pred, output_dict=True)

metrics = {
    'periodo_teste_dias': TEST_DAYS,
    'accuracy': acc,
    'precision': prec,
    'recall': rec,
    'f1': fbeta,
    'confusion_matrix': cm,
    'classification_report': report
}

# Salva artefatos
with open('metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

with open('model_xgb.pkl', 'wb') as f:
    pickle.dump({'pipeline': model, 'feature_cols': feature_cols}, f)

# Também salva previsões de teste (para inspeção)
export = test_df.copy()
export['pred'] = y_pred
export['proba_up'] = proba
export.to_csv('test_predictions.csv', index=True)

print(f"✅ Treino concluído. Accuracy (últimos {TEST_DAYS} dias): {acc:.2%}")
print("Arquivos gerados: model_xgb.pkl, metrics.json, test_predictions.csv")
