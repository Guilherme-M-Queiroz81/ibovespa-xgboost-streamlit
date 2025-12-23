# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import datetime, timedelta
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, auc, precision_recall_curve, log_loss, brier_score_loss,
    matthews_corrcoef, cohen_kappa_score
)

# ------------------------------
# Configuração e tema (fundo azul)
# ------------------------------
st.set_page_config(page_title='Ibovespa – Tendência (XGBoost)', layout='wide')
st.markdown("""
<style>
/* Fundo geral do app com gradiente */
.stApp {
    background: linear-gradient(135deg, #0f2d57, #1a237e, #4a148c);
    color: #ffffff;
}

/* Container principal */
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    color: #ffffff;
}

/* Barra lateral com gradiente */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b2244, #283593, #6a1b9a);
    color: #ffffff;
    border-right: 2px solid #4a148c;
}

/* Títulos e textos */
h1, h2, h3, h4, h5, h6, p, .stMarkdown, .stCaption {
    color: #ffffff !important;
    text-shadow: 1px 1px 2px #000000;
}

/* Botões */
.stButton>button {
    background: linear-gradient(90deg, #1976d2, #512da8);
    color: #ffffff;
    border-radius: 8px;
    font-weight: 600;
    border: 0;
    transition: 0.3s;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #1565c0, #311b92);
    color: #ffffff;
}

/* Inputs (texto, número, data, select) */
.stTextInput>div>div>input,
.stNumberInput>div>div>input,
.stDateInput>div>div>input,
.stSelectbox>div>div>div {
    background: linear-gradient(90deg, #0b2244, #1a237e);
    color: #ffffff;
    border: 1px solid #4a148c;
}

/* Métricas */
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
    color: #ffffff !important;
    text-shadow: 1px 1px 2px #000000;
}
[data-testid="stMetricDelta"] svg { display: none; }

/* Dataframes e tabelas */
.css-1mwbz9i, .stDataFrame {
    background: linear-gradient(135deg, #0b2244, #283593);
    color: #ffffff;
}

/* Gráficos Altair */
.vega-embed {
    background: linear-gradient(135deg, #0f2d57, #4a148c);
    border-radius: 8px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)


# ------------------------------
# Caminhos
# ------------------------------
DATA_PATH   = Path('Dados Históricos - Ibovespa 17.08.2021 - 18.08.2025 final.csv')
MODEL_PATH  = Path('model_xgb.pkl')
METRICS_PATH= Path('metrics.json')
LOG_PATH    = Path('logs/predictions.csv')
LOG_PATH.parent.mkdir(exist_ok=True)

# ------------------------------
# Funções utilitárias
# ------------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constrói as features usadas no treinamento (consistentes com feature_cols).
    """
    out = pd.DataFrame(index=df.index)
    out['close'] = df['Último']

    # Retornos e janelas
    out['retorno_1d'] = out['close'].pct_change()
    out['mm_5'] = out['close'].rolling(5).mean()
    out['mm_21'] = out['close'].rolling(21).mean()
    out['vol_5'] = out['retorno_1d'].rolling(5).std()
    out['momentum_5'] = out['close'] - out['close'].shift(5)

    # RSI (média simples de 14 períodos)
    delta = out['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    out['rsi_14'] = 100 - (100 / (1 + rs))

    # MACD (12-26) + sinal 9
    ema12 = ema(out['close'], 12); ema26 = ema(out['close'], 26)
    macd = ema12 - ema26
    out['macd_12_26'] = macd
    out['macd_signal_9'] = ema(macd, 9)

    # Bandas de Bollinger 20, 2
    ma20 = out['close'].rolling(20).mean()
    std20 = out['close'].rolling(20).std()
    out['bb_upper_20_2'] = ma20 + 2 * std20
    out['bb_lower_20_2'] = ma20 - 2 * std20

    # Alvo: próximo dia (binário: 1 se retorno > 0)
    next_return = out['close'].pct_change().shift(-1)
    out['alvo'] = (next_return > 0).astype(int)

    return out

def to_float(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return (
        s.astype(str)
         .str.replace(',', '.', regex=False)
         .str.replace(r'[^0-9\.\-]', '', regex=True)
         .astype(float)
    )

@st.cache_data
def load_data():
    raw = pd.read_csv(DATA_PATH, sep=',')
    raw.columns = [c.replace('"','') for c in raw.columns]
    raw['Data'] = pd.to_datetime(raw['Data'], dayfirst=True, errors='coerce')
    raw = raw.sort_values('Data').set_index('Data')
    df = pd.DataFrame({'Último': to_float(raw['Último'])})
    return df

@st.cache_resource
def load_model():
    # Linha de importação do arquivo de modelo treinado via pickle (como pedido no Tech Challenge):
    with open(MODEL_PATH, 'rb') as f:
        obj = pickle.load(f)
    return obj['pipeline'], obj['feature_cols']

@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def check_feature_alignment(feat_df: pd.DataFrame, feature_cols: list):
    cols_available = list(feat_df.columns)
    missing = [c for c in feature_cols if c not in cols_available]
    if missing:
        st.warning(f"As seguintes features do modelo não estão no dataframe: {missing}. "
                   f"Usando a interseção (pode afetar previsões).")
    use_cols = [c for c in feature_cols if c in cols_available]
    return feat_df[use_cols], use_cols

def classify_with_threshold(proba_up: np.ndarray, threshold: float) -> np.ndarray:
    return (proba_up >= threshold).astype(int)

def ks_statistic(y_true: np.ndarray, proba_up: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, proba_up)
    return float(np.max(np.abs(tpr - fpr)))

def psi_score(ref: np.ndarray, cur: np.ndarray, n_bins: int = 10) -> float:
    # Bins por quantis do ref
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) < 5 or len(cur) < 5:
        return np.nan
    quantiles = np.linspace(0, 1, n_bins + 1)
    bins = np.quantile(ref, quantiles)
    bins = np.unique(bins)
    if len(bins) < 3:
        return 0.0
    ref_hist, _ = np.histogram(ref, bins=bins)
    cur_hist, _ = np.histogram(cur, bins=bins)
    ref_pct = ref_hist / (ref_hist.sum() + 1e-12)
    cur_pct = cur_hist / (cur_hist.sum() + 1e-12)
    psi = np.sum((ref_pct - cur_pct) * np.log((ref_pct + 1e-12)/(cur_pct + 1e-12)))
    return float(psi)

# ------------------------------
# Carregamento
# ------------------------------
try:
    base_df = load_data()
    pipeline, feature_cols = load_model()
    metrics_json = load_metrics()
except Exception as e:
    st.error(f"Erro ao carregar recursos: {e}")
    st.stop()

# ------------------------------
# Cabeçalho e KPIs iniciais
# ------------------------------
st.markdown("<h1>📈 Ibovespa – Previsão de Tendência (XGBoost)</h1>", unsafe_allow_html=True)
st.caption("Dashboard com previsões, filtros, logs, avaliação, PSI e retreino.")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
if metrics_json:
    kpi1.metric("Accuracy (30d)", f"{metrics_json.get('accuracy', np.nan)*100:.2f}%")
    kpi2.metric("Precisão",        f"{metrics_json.get('precision', np.nan)*100:.2f}%")
    kpi3.metric("Recall",          f"{metrics_json.get('recall', np.nan)*100:.2f}%")
    kpi4.metric("F1-Score",        f"{metrics_json.get('f1', np.nan)*100:.2f}%")
else:
    kpi1.write("Treine o modelo para exibir métricas.")

# ------------------------------
# Barra lateral (filtros e threshold)
# ------------------------------
st.sidebar.header("🧰 Filtros & Configurações")

min_date = base_df.index.min().date()
max_date = base_df.index.max().date()
date_range = st.sidebar.date_input(
    "Intervalo de datas (visualizações)",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

show_mm5  = st.sidebar.checkbox("Mostrar MM5", value=True)
show_mm21 = st.sidebar.checkbox("Mostrar MM21", value=True)
show_rsi  = st.sidebar.checkbox("Mostrar RSI(14)", value=True)
rsi_regime = st.sidebar.selectbox(
    "Filtro por regime RSI",
    ["Todos", "RSI > 70 (sobrecompra)", "RSI < 30 ( sobrevenda )"],
    index=0
)

threshold = st.sidebar.slider(
    "Threshold p/ classificar 'Alta' (probabilidade)",
    min_value=0.1, max_value=0.9, value=0.5, step=0.05
)
st.sidebar.caption("⚠️ Ajustar o limiar pode melhorar Recall ou Precisão conforme seu objetivo.")

# ------------------------------
# Abas
# ------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📊 Visualizações", "🔮 Previsão", "🧾 Logs", "📐 Avaliação", "⚙️ Treino", "🔍 PSI"]
)

# ------------------------------
# Aba 1 - Visualizações
# ------------------------------
with tab1:
    feat_df = compute_features(base_df).dropna()

    # Filtro de data
    mask_date = (feat_df.index.date >= date_range[0]) & (feat_df.index.date <= date_range[1])
    viz_df = feat_df.loc[mask_date].copy()

    # Filtro por regime RSI
    if rsi_regime == "RSI > 70 (sobrecompra)":
        viz_df = viz_df[viz_df['rsi_14'] > 70]
    elif rsi_regime == "RSI < 30 ( sobrevenda )":
        viz_df = viz_df[viz_df['rsi_14'] < 30]

    plot_df = viz_df[['close','mm_5','mm_21','rsi_14']].reset_index().rename(
        columns={'Data':'data','close':'Fechamento','mm_5':'MM 5','mm_21':'MM 21','rsi_14':'RSI 14'}
    )

    st.subheader("Série histórica com Médias Móveis e RSI")

    layers = []
    line_close = alt.Chart(plot_df).mark_line(color="#42a5f5").encode(
        x='data:T', y='Fechamento:Q', tooltip=['data','Fechamento']
    ).properties(width=1000, height=350)
    layers.append(line_close)

    if show_mm5:
        layers.append(alt.Chart(plot_df).mark_line(color='#ffb74d').encode(x='data:T', y='MM 5:Q'))
    if show_mm21:
        layers.append(alt.Chart(plot_df).mark_line(color='#66bb6a').encode(x='data:T', y='MM 21:Q'))

    chart_top = alt.layer(*layers).resolve_scale(y='independent')
    st.altair_chart(chart_top, use_container_width=True)

    if show_rsi:
        rsi_chart = alt.Chart(plot_df).mark_line(color='#ab47bc').encode(
            x='data:T', y='RSI 14:Q'
        ).properties(width=1000, height=150)
        h70 = alt.Chart(plot_df).mark_rule(color='red').encode(y=alt.value(70))
        h30 = alt.Chart(plot_df).mark_rule(color='green').encode(y=alt.value(30))
        st.altair_chart(alt.layer(rsi_chart, h70, h30), use_container_width=True)

# ------------------------------
# Aba 2 - Previsão
# ------------------------------
with tab2:
    st.subheader("Prever tendência para o próximo dia")
    mode = st.radio("Modo de previsão", ["Automática (usar último da base)", "Manual"], index=0)

    feat_full = compute_features(base_df).dropna()
    X_feat, use_cols = check_feature_alignment(feat_full, feature_cols)

    if mode.startswith("Automática"):
        last_date = base_df.index.max()
        last_close = float(base_df.loc[last_date, 'Último'])
        st.metric("Último fechamento (base local)", f"{last_close:,.2f}")

        if st.button("🔮 Prever automaticamente"):
            X_last = X_feat.iloc[[-1]][use_cols]
            proba = float(pipeline.predict_proba(X_last)[0,1])
            pred_class = int(proba >= threshold)

            st.success(
                f"Previsão para o PRÓXIMO dia: **{'Sobe (1)' if pred_class==1 else 'Cai (0)'}** "
                f"Prob. de alta = {proba:.2%} | threshold = {threshold:.2f}"
            )

            # Log
            now = datetime.now().isoformat()
            log_row = {
                'timestamp': now,
                'data_input': str(last_date.date()),
                'ultimo_input': last_close,
                'pred_class': pred_class,
                'proba_up': proba,
                'proba_down': 1.0 - proba,
                'threshold': threshold
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
            input_date = st.date_input("Data do fechamento", value=max_date + timedelta(days=1))
            input_close = st.number_input("Último (fechamento)", min_value=0.0, step=0.01)
            submitted = st.form_submit_button("🔮 Prever manualmente")

        if submitted:
            new_idx = pd.to_datetime(str(input_date))
            last_idx = base_df.index.max()
            if new_idx <= last_idx:
                st.warning("A data inserida deve ser posterior à última data da base. Por favor, insira uma data futura.")
            else:
                extended = base_df.copy()
                extended.loc[new_idx, 'Último'] = float(input_close)
                extended = extended.sort_index()

                feat_ext = compute_features(extended).dropna()
                X_ext, use_cols_ext = check_feature_alignment(feat_ext, feature_cols)
                X_last = X_ext.iloc[[-1]][use_cols_ext]
                proba = float(pipeline.predict_proba(X_last)[0,1])
                pred_class = int(proba >= threshold)

                st.success(
                    f"Previsão para o PRÓXIMO dia: **{'Sobe (1)' if pred_class==1 else 'Cai (0)'}** "
                    f"Prob. de alta = {proba:.2%} | threshold = {threshold:.2f}"
                )

                # Log
                now = datetime.now().isoformat()
                log_row = {
                    'timestamp': now,
                    'data_input': str(input_date),
                    'ultimo_input': float(input_close),
                    'pred_class': pred_class,
                    'proba_up': proba,
                    'proba_down': 1.0 - proba,
                    'threshold': threshold
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
        st.caption("Use o botão para limpar o histórico de previsões.")

# ------------------------------
# Aba 3 - Logs
# ------------------------------
with tab3:
    st.subheader("Últimos usos (log de previsão)")
    if LOG_PATH.exists():
        logs = pd.read_csv(LOG_PATH)
        st.dataframe(logs.tail(100), use_container_width=True)
        st.download_button('⬇️ Baixar CSV de logs', data=logs.to_csv(index=False),
                           file_name='predictions_log.csv', mime='text/csv')
        counts = logs['pred_class'].value_counts().rename({0:'Cai (0)', 1:'Sobe (1)'}).reset_index()
        counts.columns = ['Classe', 'Quantidade']
        st.bar_chart(counts.set_index('Classe'))
    else:
        st.info("Nenhum log ainda. Faça uma previsão para criar o arquivo.")

# ------------------------------
# Aba 4 - Avaliação (hold-out)
# ------------------------------
with tab4:
    st.subheader("Avaliação no hold-out (30 dias finais) com threshold ajustável")

    TEST_DAYS = 30
    feat_df = compute_features(base_df).dropna()

    if len(feat_df) > TEST_DAYS + 50:
        test_df = feat_df.iloc[-TEST_DAYS:]
        X_test, use_cols_test = check_feature_alignment(test_df, feature_cols)
        y_test = test_df['alvo'].values
        proba_up = pipeline.predict_proba(X_test)[:,1]
        y_pred_thr = classify_with_threshold(proba_up, threshold)

        # Métricas
        acc = accuracy_score(y_test, y_pred_thr)
        prec = precision_score(y_test, y_pred_thr)
        rec = recall_score(y_test, y_pred_thr)
        f1 = f1_score(y_test, y_pred_thr)
        cm = confusion_matrix(y_test, y_pred_thr)

        # Balanced accuracy manual
        tn, fp, fn, tp = cm.ravel()
        tnr = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        bal_acc = (rec + tnr) / 2

        # AUCs e curvas
        fpr, tpr, _ = roc_curve(y_test, proba_up)
        roc_auc = auc(fpr, tpr)
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, proba_up)
        pr_auc = auc(rec_curve, prec_curve)

        # Log loss e Brier
        try:
            ll = log_loss(y_test, proba_up, eps=1e-15)
        except Exception:
            ll = np.nan
        brier = brier_score_loss(y_test, proba_up)

        # MCC, Kappa e KS
        mcc = matthews_corrcoef(y_test, y_pred_thr)
        kappa = cohen_kappa_score(y_test, y_pred_thr)
        ks = ks_statistic(y_test, proba_up)

        st.write(
            f"**Threshold:** {threshold:.2f} \n"
            f"Accuracy: {acc:.2%} | Precisão: {prec:.2%} | Recall: {rec:.2%} | F1: {f1:.2%} \n"
            f"Balanced Acc: {bal_acc:.2%} | AUC-ROC: {roc_auc:.3f} | PR-AUC: {pr_auc:.3f} \n"
            f"Log Loss: {ll:.4f} | Brier: {brier:.4f} | MCC: {mcc:.3f} | Kappa: {kappa:.3f} | KS: {ks:.3f}"
        )

        cm_df = pd.DataFrame(cm, index=['Real 0', 'Real 1'], columns=['Pred 0','Pred 1'])
        st.dataframe(cm_df, use_container_width=True)

        # ROC e PR
        roc_df = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
        pr_df = pd.DataFrame({'Recall': rec_curve, 'Precision': prec_curve})
        roc_chart = alt.Chart(roc_df).mark_line(color='#42a5f5').encode(
            x='FPR:Q', y='TPR:Q'
        ).properties(title='Curva ROC', width=450, height=350)
        pr_chart = alt.Chart(pr_df).mark_line(color='#ab47bc').encode(
            x='Recall:Q', y='Precision:Q'
        ).properties(title='Precision–Recall', width=450, height=350)
        st.altair_chart(alt.hconcat(roc_chart, pr_chart), use_container_width=True)

        # Estratégia simples: sinal sem custos
        next_ret = test_df['close'].pct_change().shift(-1)
        strat_ret = next_ret * y_pred_thr
        strat_cum = (1 + pd.Series(strat_ret)).cumprod().iloc[-2] - 1 if len(pd.Series(strat_ret).dropna())>1 else np.nan
        buy_hold_cum = (1 + pd.Series(next_ret)).cumprod().iloc[-2] - 1 if len(pd.Series(next_ret).dropna())>1 else np.nan
        st.info(f"Retorno acumulado (sinal) ≈ {strat_cum:.2%} vs Buy&Hold ≈ {buy_hold_cum:.2%} (ilustrativo, sem custos).")
    else:
        st.info("Base insuficiente para avaliação.")

# ------------------------------
# Aba 5 - Treino (retreino via subprocess)
# ------------------------------
with tab5:
    st.subheader("Gerenciamento do Modelo")
    st.caption("Re-treine o modelo e atualize métricas se necessário.")

    if st.button("🔄 Re-treinar modelo"):
        try:
            import subprocess
            st.info("Treinando modelo, aguarde...")
            result = subprocess.run(["python", "train_xgb.py"], capture_output=True, text=True, check=True)
            st.success("Treino concluído!")
            st.code(result.stdout[:2000], language="bash")
            # Recarrega
            pipeline, feature_cols = load_model()
            metrics_json = load_metrics()
            st.success("Modelo e métricas recarregados!")
        except Exception as e:
            st.error(f"Erro ao treinar: {e}")

    metrics_json = load_metrics()
    if metrics_json:
        st.write("Métricas atuais:")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Accuracy", f"{metrics_json.get('accuracy', np.nan)*100:.2f}%")
        k2.metric("Precisão", f"{metrics_json.get('precision', np.nan)*100:.2f}%")
        k3.metric("Recall",   f"{metrics_json.get('recall', np.nan)*100:.2f}%")
        k4.metric("F1-Score", f"{metrics_json.get('f1', np.nan)*100:.2f}%")
    else:
        st.info("Nenhuma métrica encontrada. Treine o modelo para gerar metrics.json.")

# ------------------------------
# Aba 6 - PSI (Monitoramento de estabilidade)
# ------------------------------
with tab6:
    st.subheader("Monitoramento (PSI) – últimos 30 dias vs histórico")

    feat_df = compute_features(base_df).dropna()
    if len(feat_df) > 100:
        recent = feat_df.iloc[-30:]
        ref    = feat_df.iloc[:-30]

        cols_psi = [c for c in feat_df.columns if c != 'alvo']
        psi_rows = []
        for c in cols_psi:
            score = psi_score(ref[c].values, recent[c].values, n_bins=10)
            psi_rows.append({'feature': c, 'psi': score})

        psi_df = pd.DataFrame(psi_rows).sort_values('psi', ascending=False)
        st.dataframe(psi_df, use_container_width=True)

        def psi_level(v):
            if np.isnan(v): return "insuficiente"
            if v < 0.10:    return "estável"
            if v < 0.20:    return "atenção"
            return "drift alto"

        psi_df['nível'] = psi_df['psi'].apply(psi_level)
        st.bar_chart(psi_df.set_index('feature')['psi'])
        st.caption("Interpretação: <0,10 estável | 0,10–0,20 atenção | >0,20 drift alto.")
    else:
        st.info("Base insuficiente para PSI.")