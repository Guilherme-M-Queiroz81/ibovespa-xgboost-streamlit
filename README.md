# Ibovespa – Previsão de Tendência com XGBoost + Streamlit

Aplicação interativa para prever **tendência diária** do **IBOVESPA** (1 = sobe, 0 = cai) usando um **modelo XGBoost** treinado sobre o histórico 2021–2025. O projeto inclui **deploy com Streamlit**, **monitoramento de métricas**, **gráficos interativos**, **importância das features**, **detecção de drift (PSI)** e **log de uso**.

---

## 🚀 Recursos
- **Interface Streamlit** com visualizações de *Fechamento*, **MM 5/21** e **RSI(14)**.
- **Previsão em tempo real** ao inserir um novo fechamento ("Último").
- **Previsão em lote** com **upload de CSV** (`Data`, `Último`).
- **Painel de métricas**: Accuracy, Precisão, Recall, F1, Matriz de Confusão.
- **Gráfico de Importância das Features** (XGBoost).
- **Drift/Estabilidade** com **PSI** (últimos 30 dias vs histórico).
- **Log de uso** automático em `logs/predictions.csv`.

---

## 🧱 Arquitetura do Projeto
```
.
├── app.py                  # Aplicação Streamlit (UI, gráficos, previsões, PSI, importâncias)
├── train_xgb.py           # Treino do modelo, cálculo de métricas e salvamento (pickle)
├── requirements.txt       # Dependências do projeto
├── README.md              # Este documento
├── Dados Históricos - Ibovespa 17.08.2021 - 18.08.2025 final.csv   # Base histórica
├── model_xgb.pkl          # Modelo treinado (gerado pelo train_xgb.py)
├── metrics.json           # Métricas do hold-out (30 dias finais)
└── logs/
    └── predictions.csv    # Log de uso (timestamp, entrada, previsões)
```

---

## ⚙️ Instalação e Execução Local
### 1) Ambiente
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2) Dependências
```bash
pip install -r requirements.txt
```

### 3) Treino do modelo (gera `model_xgb.pkl` e `metrics.json`)
```bash
python train_xgb.py
```

### 4) Rodar a aplicação
```bash
streamlit run app.py
```

> **Observação:** garanta que o arquivo CSV histórico (`Dados Históricos - Ibovespa 17.08.2021 - 18.08.2025 final.csv`) esteja no diretório do projeto.

---

## 🧠 Como o Modelo Funciona
- **Modelo**: `XGBClassifier` (pipeline com `StandardScaler`).
- **Features** derivadas de `Último` (fechamento):
  - Retorno diário, MM(5/21), Volatilidade(5), Momentum(5)
  - RSI(14), MACD(12,26), Sinal(9)
  - Bandas de Bollinger(20,2)
- **Alvo**: tendência do **próximo dia** (variação > 0 ⇒ 1; caso contrário ⇒ 0).
- **Validação temporal**: treino = todos menos **últimos 30 dias**; teste = **30 dias finais**.

---

## 📈 Painel de Visualização e Monitoramento (no app)
- **Métricas**: lê de `metrics.json` e também recalcula no app para o hold-out.
- **Gráficos**:
  - Série de **Fechamento** com **MM 5/21**.
  - **RSI(14)** com **regras horizontais** em 70/30 e **marcações** de sobrecompra/sobrevenda.
  - **Importância das Features** do XGBoost (barras ordenadas).
- **PSI (Population Stability Index)**: compara distribuição das features nos **últimos 30 dias** vs **histórico**.

Regra prática de PSI:
- `< 0.10`: estável
- `0.10–0.20`: atenção
- `> 0.20`: drift alto (investigar)

---

## 🔮 Previsão (Manual e Lote)
### Manual
1. Informe **Data** e **Último** (fechamento) no formulário.
2. O app calcula as features com base no histórico + ponto inserido.
3. Exibe **classe** (0/1) e **probabilidade de alta**.
4. Registra no **log** (`logs/predictions.csv`).

### Lote (Upload CSV)
1. Faça upload de um **CSV** com colunas `Data` e `Último` (nomes próximos também são aceitos: `close`, `fechamento`, `ultimo`).
2. O app calcula as features para cada data e gera a previsão do **dia seguinte**.
3. Baixe os resultados e o app adiciona ao **log**.

---

## ☁️ Deploy no Streamlit Cloud (via GitHub)
1. **GitHub**: crie um repositório e adicione todos os arquivos do projeto.
   ```bash
   git init
   git add .
   git commit -m "Projeto Ibovespa XGBoost Streamlit"
   git remote add origin https://github.com/SEU_USUARIO/ibovespa-xgboost-streamlit.git
   git push -u origin master
   ```
2. **Streamlit Cloud**: acesse [streamlit.io/cloud](https://streamlit.io/cloud) → **New app** → conecte ao repo.
3. Selecione **Branch** (`main` ou `master`) e **Main file path** = `app.py`.
4. Confirme o **requirements.txt** e **Deploy**.
5. Compartilhe o **link público**: `https://SEU_USUARIO-ibovespa-xgboost-streamlit.streamlit.app`.

> **Dicas**:
> - Arquivos muito grandes (>100MB): considere usar **Git LFS** ou hospedar em storage externo.
> - Segredos/variáveis: configure no painel do Streamlit Cloud (evite incluir em código).

---

## 🧪 Testes Rápidos
- Execute `train_xgb.py` e verifique se foram gerados:
  - `model_xgb.pkl`, `metrics.json`, `test_predictions.csv`.
- Rode `app.py` e confira:
  - Painel de métricas preenchido.
  - Gráficos renderizados (Fechamento + MM, RSI, Importância).
  - PSI calculado (se a base for suficiente).
  - Log criado ao fazer uma previsão.

---

## 🛠️ Troubleshooting
- **xgboost não encontrado**: `pip install xgboost` (já no `requirements.txt`).
- **CSV com formato diferente**: garanta colunas `Data` (D/M/A) e `Último` numérico.
- **Erro de janelas/NaNs**: envie **mais dias** no upload para completar janelas (ex.: MM21, BB20).
- **Permissões de escrita** (Streamlit Cloud): logs são mantidos no sistema de arquivos da sessão; para persistência externa, use storage/cloud.

---

## 📚 Referências (conceitos)
- XGBoost: https://xgboost.readthedocs.io
- RSI/MACD e indicadores técnicos: documentação e literatura de análise técnica.
- PSI (Population Stability Index): práticas de monitoramento de estabilidade de modelos.

---

## 📄 Licença
Distribuição para fins acadêmicos/educacionais.
