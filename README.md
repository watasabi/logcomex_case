a name="readme-top"></a>

<div align="center">
  <h1 align="center">Logcomex - Case Cientista de Dados (Pleno)</h1>
  <p align="center">
    Modelo de classificação de risco aduaneiro para predição de canais de parametrização.
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

Este projeto foi desenvolvido como resolução do case técnico para a posição de Cientista de Dados. O objetivo principal é prever o **Canal de Parametrização** (Verde, Amarelo, Vermelho ou Cinza) de Declarações de Importação (DIs), auxiliando na identificação prévia de riscos aduaneiros.

### Desafios Resolvidos
* **Alta Cardinalidade:** Lidar com variáveis categóricas extremas (ex: consignee_code, ncm_code) sem causar explosão de dimensionalidade.
* **Desbalanceamento Severo:** A grande maioria das importações é parametrizada no canal Verde, exigindo técnicas rigorosas de balanceamento de função de custo (class weights no Optuna e PyTorch).
* **Vazamento Temporal:** Garantia de que o split de treino e teste respeita a ordem cronológica, simulando um ambiente de produção real.

<p align="right">(<a href="#readme-top">voltar ao topo</a>)</p>

## Metodologia e Abordagem

1. **Análise Exploratória (EDA):** Redução de dimensionalidade com **MCA (Multiple Correspondence Analysis)** e **t-SNE** para descoberta de padrões visuais de fraude, validado pelo Silhouette Score (Hamming) do **K-Modes**.
2. **Processamento:** Agrupamento de cauda longa (Top N) para redução de cardinalidade antes da modelagem matemática.
3. **Modelagem Clássica (Ensemble):** Otimização bayesiana com **Optuna** criando um VotingClassifier entre LightGBM e LogisticRegression.
4. **Deep Learning (Atenção Tabular):** Construção de uma arquitetura baseada em **PyTorch Lightning** com um mecanismo customizado de Feature Gating (Attention) para ponderação dinâmica das features mais importantes.
5. **Rastreabilidade:** Tracking completo dos modelos e hiperparâmetros no **MLflow**.

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
Com o uv instalado, clone este repositório, acesse a raiz do projeto e execute a sincronização. O uv criará o .venv automaticamente e instalará todas as dependências isoladas, baseadas no arquivo uv.lock.

`uv sync`

### 3. Rodando os Scripts de Treinamento
Você pode rodar qualquer script do pipeline prefixando-o com uv run.

`uv run python notebooks/processing/00_mca_processing.py`
`uv run python notebooks/processing/01_split_dataset.py`

Treinar os modelos:
`uv run python notebooks/training/00_classicalmodel.py`
`uv run python notebooks/training/01_deepmodel.py`

Visualizar o Tracking no MLflow:
`uv run mlflow ui --backend-store-uri sqlite:///notebooks/training/mlflow.db`

<p align="right">(<a href="#readme-top">voltar ao topo</a>)</p>

## Organização e Estrutura

```text
.
├── config
├── data
│   ├── external
│   │   ├── Cientista de Dados - Pleno (1).zip
│   │   ├── sample_data.parquet
│   │   └── teste-pleno.pdf
│   ├── interim
│   ├── processed
│   │   ├── 01_data_mca.parquet
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
│   │   └── 01_split_dataset.py
│   └── training
│       ├── 00_classicalmodel.py
│       ├── 01_deepmodel.py
│       ├── confusion_matrix_dl.png
│       ├── lightning_logs/
│       ├── mlflow.db
│       └── mlruns/
├── pipe
│   ├── artefacts
│   ├── orchestrator.py
│   └── steps
│       ├── 01_load.py
│       ├── 02_preprocess.py
│       ├── 03_inference.py
│       └── 04_postprocess.py
├── pyproject.toml
├── README.md
├── references
├── reports
│   └── figures/
├── src
│   ├── seq2seq attention_example.ipynb
│   └── ts_preprocessing.py
└── uv.lock