<a name="readme-top"></a>

<div align="center">
  <h1 align="center">Logcomex - Case Cientista de Dados (Pleno)</h1>
  <p align="center">
    Classificação de risco aduaneiro e Forecasting Multivariado de volume operacional por canal.
    <br />
    <br />
    <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
    <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/MLflow-%23d9ead3.svg?style=for-the-badge&logo=numpy&logoColor=blue" alt="MLflow">
  </p>
</div>

<details>
  <summary>📝 Tabela de Conteúdos</summary>
  <ol>
    <li><a href="#sobre-o-projeto">Sobre o Projeto</a></li>
    <li><a href="#metodologia-e-abordagem">Metodologia e Abordagem</a></li>
    <li><a href="#como-executar-instalacao">Como Executar (Instalação)</a></li>
    <li><a href="#organizacao-e-estrutura">Organização e Estrutura</a></li>
  </ol>
</details>

---

## Sobre o Projeto

Este projeto foi desenvolvido como resolução do case técnico para a posição de Cientista de Dados. A solução engloba duas vertentes de negócio:
1. **Compliance (Classificação):** Prever o risco individual (Canal Verde, Amarelo, Vermelho ou Cinza) de Declarações de Importação (DIs).
2. **Planejamento Operacional (Forecasting):** Prever o volume diário de parametrizações para cada canal, permitindo alocação eficiente de auditores da Receita Federal em picos de canais vermelhos/cinzas.

### Desafios Resolvidos
* **Alta Cardinalidade:** Resolução de variáveis categóricas extremas (ex: consignee_code, ncm_code) via MCA, evitando explosão de dimensionalidade.
* **Desbalanceamento Severo:** O problema clássico (90%+ de canal Verde) foi mitigado de duas formas: (1) via `class_weights` balanceados nas funções de custo e (2) **arquiteturalmente**, transformando o problema em uma Regressão Multivariada Contínua (Séries Temporais), onde o foco é o volume real, diluindo o viés da classe majoritária.
* **Vazamento Temporal:** Todo o projeto utiliza validação estritamente baseada no tempo (Split Cronológico e Janelas Deslizantes) para simular o cenário real de produção.

<p align="right">(<a href="#readme-top">voltar ao topo</a>)</p>

## Metodologia e Abordagem

1. **Análise Exploratória (EDA):** Redução de dimensionalidade com **MCA (Multiple Correspondence Analysis)** e **t-SNE** para descoberta de padrões visuais de fraude, validado pelo Silhouette Score do **K-Modes**.
2. **Processamento Temporal state-of-the-art:** Pipeline de tratamento de séries temporais com transformação de Yeo-Johnson, filtros e feature gating.
3. **Modelagem Clássica (Ensemble):** Otimização bayesiana com **Optuna** criando um VotingClassifier (LightGBM + LogisticRegression) para a tarefa de classificação individual.
4. **Forecasting Multivariado (Deep Learning):** Arquitetura **Seq2Seq com DotAttention** (PyTorch Lightning) prevendo o volume futuro simultâneo dos 4 canais (Janela de 30 dias de saída baseada em 90 dias de histórico). O mecanismo de atenção permite extrair explicabilidade de negócio (Top Lags Históricos).
5. **Rastreabilidade e Tracking:** Registro completo de hiperparâmetros, artefatos (gráficos SHAP e Atenção) e métricas (F1-Macro, SMAPE, R²) via **MLflow**.

<p align="right">(<a href="#readme-top">voltar ao topo</a>)</p>

## Como Executar (Instalação)

Este projeto utiliza o **uv**, o gerenciador de pacotes e ambientes Python de altíssima performance escrito em Rust.

### 1. Instalando o uv
Se você ainda não possui o uv instalado em sua máquina, instale-o com o comando abaixo:

**Linux / macOS:**
`curl -LsSf https://astral.sh/uv/install.sh | sh`

**Windows (PowerShell):**
`powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

### 2. Sincronizando o Ambiente
Com o uv instalado, clone este repositório, acesse a raiz do projeto e execute a sincronização. O uv criará o `.venv` automaticamente e instalará todas as dependências isoladas.

`uv sync`

### 3. Rodando os Scripts
Os pipelines estão divididos nas tarefas de Classificação Clássica e Forecasting Profundo. Execute via `uv run`:

**Processamento dos Dados:**
`uv run python notebooks/processing/00_mca_processing.py`
`uv run python notebooks/processing/01_split_dataset.py`
`uv run python notebooks/processing/02_prep_forecasting.py`

**Treinamento dos Modelos:**
`uv run python notebooks/training/00_classicalmodel.py`
`uv run python notebooks/training/03_forecasting_seq2seq.py`

**Visualizar o Dashboard de Experimentos:**
`uv run mlflow ui --backend-store-uri sqlite:///notebooks/training/mlflow.db`

<p align="right">(<a href="#readme-top">voltar ao topo</a>)</p>

## Organização e Estrutura

```text
.
├── config
├── data
│   ├── external
│   │   ├── sample_data.parquet
│   │   └── teste-pleno.pdf
│   ├── interim
│   ├── processed
│   │   ├── 01_data_mca.parquet
│   │   ├── ts_train.parquet     <- Dados diários agrupados (Treino)
│   │   ├── ts_test.parquet      <- Dados diários agrupados (Teste)
│   │   ├── test.parquet
│   │   └── train.parquet
│   └── raw
├── LICENSE
├── models
│   └── ensemble_model.pkl
├── notebooks
│   ├── eda
│   │   ├── 00_first_eda.py
│   │   ├── 01_eda_kmodes_mca.py
│   │   ├── 02_eda_kmodes_tsne.py
│   │   └── 03_eda_kmodes_silhoutte.py
│   ├── processing
│   │   ├── 00_mca_processing.py
│   │   ├── 01_split_dataset.py
│   │   └── 02_prep_forecasting.py  <- Agrupador de Séries Temporais
│   └── training
│       ├── 00_classicalmodel.py
│       ├── 03_forecasting_seq2seq.py <- Modelo Profundo de Previsão
│       ├── multivariate_forecast.png
│       ├── attention_heatmap.png
│       ├── lightning_logs/
│       ├── mlflow.db
│       └── mlruns/
├── pipe
│   ├── orchestrator.py
│   └── steps
│       ├── 01_load.py
│       ├── 02_preprocess.py
│       ├── 03_inference.py
│       └── 04_postprocess.py
├── pyproject.toml
├── README.md
├── src
│   └── ts_preprocessing.py   <- Biblioteca utilitária (Pipeline de TS)
└── uv.lock