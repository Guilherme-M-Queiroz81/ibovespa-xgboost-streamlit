# -*- coding: utf-8 -*-
from pathlib import Path
from datetime import datetime

import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ------------------------------
# Configuração e tema (fundo azul)
# ------------------------------
st.set_page_config(page_title='Ibovespa – Previsão de Tendência (XGBoost)', layout='wide')

st.markdown("""
<style>
/* Fundo da aplicação */
.stApp {
    background-color: #0f2d57;  /* azul escuro */
}
/* Cards e containers com leve transparência */
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}
/* Títulos brancos para contraste */
h1, h2, h3, h4, h5, h6, p, .stMarkdown, .stCaption {
    color: #ffffff !important;
}
/* Botões */
.stButton>button {
    background-color: #1976d2;
    color: white;
    border-radius: 8px;
    font-weight: 600;
    border: 0;
}
.stButton>button:hover {
    background-color: #1565c0;
    color: white;
}
/* Inputs */
.stTextInput>div>div>input,
.stNumberInput>div>div>input,
.stDateInput>div>div>input {
    background-color: #0b2244;
    color: #ffffff;
    border: 1px solid #2b4b7a;
}
.stSelectbox>div>div>div {
    background-color: #0b2244;
    color: #ffffff;
    border: 1px solid #2b4b7a;
}
/* Métricas */
[data-testid="stMetricDelta"] svg { display: none; }
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color: #ffffff !important; }
/* Dataframes */
.css-1mwbz9i, .stDataFrame { background: #0b2244 !important; }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Caminhos
# ------------------------------
DATA_PATH = Path('Dados Históricos - Ibovespa 17.08.2021 - 18.08.2025 final.csv')
MODEL_PATH = Path('model_xgb.pkl')
METRICS_PATH = Path('metrics.json')
LOG_PATH = Path('logs/predictions.csv')
LOG_PATH.parent.mkdir(exist_ok=True)

# ------------------------------
# Funções utilitárias
# ------------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out['close'] = df['Último']
    out['retorno_1d'] = out['close'].pct_change()
    out['mm_5'] = out['close'].rolling(5).mean()
    out['mm_21'] = out['close'].rolling(21).mean()
    out['vol_5'] = out['retorno_1d'].rolling(5).std()
    out['momentum_5'] = out['close'] - out['close'].shift(5)
    delta = out['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    out['rsi_14'] = 100 - (100 / (1 + rs))
    ema12 = ema(out['close'], 12); ema26 = ema(out['close'], 26)
    macd = ema12 - ema26
    out['macd_12_26'] = macd
    out['macd_signal_9'] = ema(macd, 9)
    ma20 = out['close'].rolling(20).mean()
    std20 = out['close'].rolling(20).std()
    out['bb_upper_20_2'] = ma20 + 2 * std20
    out['bb_lower_20_2'] = ma20 - 2 * std20
    next_return = out['close'].pct_change().shift(-1)
    out['alvo'] = (next_return > 0).astype(int)
    return out

@st.cache_data
def load_data():
    raw = pd.read_csv(DATA_PATH, sep=',')
    raw.columns = [c.replace('"','') for c in raw.columns]
    raw['Data'] = pd.to_datetime(raw['Data'], dayfirst=True, errors='coerce')
    raw = raw.sort_values('Data').set_index('Data')
    def to_float(s):
        if pd.api.types.is_numeric_dtype(s):
            return s.astype(float)
        return (s.astype(str)
                 .str.replace(',', '.', regex=False)
                 .str.replace(r'[^0-9\.-]', '', regex=True)
                 .astype(float))
    df = pd.DataFrame({'Último': to_float(raw['Último'])})
    return df

@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        obj = pickle.load(f)
    return obj['pipeline'], obj['feature_cols']

@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

# ------------------------------
# Carregamento
# ------------------------------
try:
    base_df = load_data()
    pipeline, feature_cols = load_model()
    metrics = load_metrics()
except Exception as e:
    st.error(f"Erro ao carregar recursos: {e}")
    st.stop()

# ------------------------------
# Cabeçalho
# ------------------------------
st.markdown("<h1>📈 Ibovespa – Previsão de Tendência (XGBoost)</h1>", unsafe_allow_html=True)
st.caption("Dashboard com fundo azul, previsão manual e automática, logs, avaliação e treino.")

# KPIs
col1, col2, col3, col4 = st.columns(4)
if metrics:
    col1.metric("Accuracy (30d)", f"{metrics['accuracy']*100:.2f}%")
    col2.metric("Precisão", f"{metrics['precision']*100:.2f}%")
    col3.metric("Recall", f"{metrics['recall']*100:.2f}%")
    col4.metric("F1-Score", f"{metrics['f1']*100:.2f}%")
else:
    col1.write("Treine o modelo para exibir métricas.")

# ------------------------------
# Abas
# ------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Visualizações", "🔮 Previsão", "📝 Logs", "📐 Avaliação", "⚙️ Treino"])

# ------------------------------
# Aba 1 - Visualizações
# ------------------------------
with tab1:
    feat_df = compute_features(base_df).dropna()
    plot_df = feat_df[['close','mm_5','mm_21','rsi_14']].reset_index().rename(
        columns={'Data':'data','close':'Fechamento','mm_5':'MM 5','mm_21':'MM 21','rsi_14':'RSI 14'}
    )

    st.subheader("Série histórica com Médias Móveis e RSI")

    line_close = alt.Chart(plot_df).mark_line(color="#42a5f5").encode(
        x='data:T', y='Fechamento:Q', tooltip=['data','Fechamento']
    ).properties(width=1000, height=350)

    line_mm5 = alt.Chart(plot_df).mark_line(color='#ffb74d').encode(x='data:T', y='MM 5:Q')
    line_mm21 = alt.Chart(plot_df).mark_line(color='#66bb6a').encode(x='data:T', y='MM 21:Q')

    chart_top = alt.layer(line_close, line_mm5, line_mm21).resolve_scale(y='independent')
    st.altair_chart(chart_top, use_container_width=True)

    rsi_chart = alt.Chart(plot_df).mark_line(color='#ab47bc').encode(x='data:T', y='RSI 14:Q').properties(width=1000, height=150)
    st.altair_chart(rsi_chart, use_container_width=True)

# ------------------------------
# Aba 2 - Previsão
# ------------------------------
with tab2:
    st.subheader("Prever tendência para o próximo dia")

    mode = st.radio("Modo de previsão", ["Automática (usar último valor da base)", "Manual"], index=0)

    # Modo automático: usa a última data e último fechamento da base local
    if mode.startswith("Automática"):
        last_date = base_df.index.max()
        last_close = float(base_df.loc[last_date, 'Último'])
        st.metric("Último fechamento (base local)", f"{last_close:,.2f}")
        if st.button("🔮 Prever automaticamente"):
            extended = base_df.copy()
            # Opcional: permitir atualizar manualmente esse último ponto
            full_feat = compute_features(extended).dropna()
            X_last = full_feat.iloc[[-1]][feature_cols]
            pred_class = int(pipeline.predict(X_last)[0])
            proba_up = float(pipeline.predict_proba(X_last)[0,1])

            st.success(f"Previsão para o PRÓXIMO dia: **{'Sobe (1)' if pred_class==1 else 'Cai (0)'}** | Prob. de alta = {proba_up:.2%}")

            # Log
            now = datetime.now().isoformat()
            log_row = {
                'timestamp': now,
                'data_input': str(last_date.date()),
                'ultimo_input': last_close,
                'pred_class': pred_class,
                'proba_up': proba_up,
                'proba_down': 1.0 - proba_up
            }
            if LOG_PATH.exists():
                logs = pd.read_csv(LOG_PATH)
                logs = pd.concat([logs, pd.DataFrame([log_row])], ignore_index=True)
            else:
                logs = pd.DataFrame([log_row])
            logs.to_csv(LOG_PATH, index=False)
            st.caption("✅ Registro salvo em logs/predictions.csv")

    else:
        with st.form("manual_predict_form"):
            input_date = st.date_input("Data do fechamento")
            input_close = st.number_input("Último (fechamento)", min_value=0.0, step=0.01)
            submitted = st.form_submit_button("🔮 Prever manualmente")

        if submitted:
            new_idx = pd.to_datetime(str(input_date))
            extended = base_df.copy()
            extended.loc[new_idx, 'Último'] = float(input_close)
            extended = extended.sort_index()

            full_feat = compute_features(extended).dropna()
            X_last = full_feat.iloc[[-1]][feature_cols]
            pred_class = int(pipeline.predict(X_last)[0])
            proba_up = float(pipeline.predict_proba(X_last)[0,1])

            st.success(f"Previsão para o PRÓXIMO dia: **{'Sobe (1)' if pred_class==1 else 'Cai (0)'}** | Prob. de alta = {proba_up:.2%}")

            # Log
            now = datetime.now().isoformat()
            log_row = {
                'timestamp': now,
                'data_input': str(input_date),
                'ultimo_input': float(input_close),
                'pred_class': pred_class,
                'proba_up': proba_up,
                'proba_down': 1.0 - proba_up
            }
            if LOG_PATH.exists():
                logs = pd.read_csv(LOG_PATH)
                logs = pd.concat([logs, pd.DataFrame([log_row])], ignore_index=True)
            else:
                logs = pd.DataFrame([log_row])
            logs.to_csv(LOG_PATH, index=False)
            st.caption("✅ Registro salvo em logs/predictions.csv")

    st.divider()
    colA, colB = st.columns(2)
    with colA:
        if st.button("🧹 Limpar logs"):
            if LOG_PATH.exists():
                LOG_PATH.unlink()
                st.success("Logs apagados com sucesso!")
            else:
                st.info("Nenhum log para limpar.")
    with colB:
        st.caption("Use o botão acima para limpar o histórico de previsões.")

# ------------------------------
# Aba 3 - Logs
# ------------------------------
with tab3:
    st.subheader("Últimos usos (log de previsão)")
    if LOG_PATH.exists():
        logs = pd.read_csv(LOG_PATH)
        st.dataframe(logs.tail(50), use_container_width=True)
        st.download_button('⬇️ Baixar CSV de logs', data=logs.to_csv(index=False), file_name='predictions_log.csv', mime='text/csv')
        # Pequena visão agregada
        counts = logs['pred_class'].value_counts().rename({0: 'Cai (0)', 1: 'Sobe (1)'}).reset_index()
        counts.columns = ['Classe', 'Quantidade']
        st.bar_chart(counts.set_index('Classe'))
    else:
        st.info("Nenhum log ainda. Faça uma previsão para criar o arquivo.")

# ------------------------------
# Aba 4 - Avaliação
# ------------------------------
with tab4:
    st.subheader("Avaliação do modelo no hold-out (30 dias finais)")
    TEST_DAYS = 30
    feat_df = compute_features(base_df).dropna()
    if len(feat_df) > TEST_DAYS + 50:
        test_df = feat_df.iloc[-TEST_DAYS:]
        X_test = test_df[feature_cols]
        y_test = test_df['alvo']
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        fbeta = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        st.write(f"Accuracy: {acc:.2%} | Precisão: {prec:.2%} | Recall: {rec:.2%} | F1: {fbeta:.2%}")
        cm_df = pd.DataFrame(cm, index=['Real 0', 'Real 1'], columns=['Pred 0','Pred 1'])
        st.dataframe(cm_df, use_container_width=True)
    else:
        st.info("Base insuficiente para avaliação.")

# ------------------------------
# Aba 5 - Treino
# ------------------------------
with tab5:
    st.subheader("Gerenciamento do Modelo")
    st.caption("Re-treine o modelo e atualize métricas se necessário.")
    # Botão de re-treino via subprocess (mantendo tudo em Streamlit)
    if st.button("🔄 Re-treinar modelo"):
        try:
            import subprocess
            st.info("Treinando modelo, aguarde...")
            result = subprocess.run(["python", "train_xgb.py"], capture_output=True, text=True, check=True)
            st.success("Treino concluído!")
            st.code(result.stdout[:2000], language="bash")  # mostra parte do log

            # Recarrega modelo e métricas
            pipeline, feature_cols = load_model()
            metrics = load_metrics()
            st.success("Modelo e métricas recarregados!")
        except Exception as e:
            st.error(f"Erro ao treinar: {e}")

    # Exibir métricas atuais, se existirem
    metrics = load_metrics()
    if metrics:
        st.write("Métricas atuais:")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
        k2.metric("Precisão", f"{metrics['precision']*100:.2f}%")
        k3.metric("Recall", f"{metrics['recall']*100:.2f}%")
        k4.metric("F1-Score", f"{metrics['f1']*100:.2f}%")
    else:
        st.info("Nenhuma métrica encontrada. Treine o modelo para gerar metrics.json.")