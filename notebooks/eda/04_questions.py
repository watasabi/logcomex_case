import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# Configurações visuais
sns.set_theme(style="whitegrid")
OUTPUT_DIR = Path("../../reports/figures/business")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Carregar Dados Brutos
print("Carregando dados...")
df_raw = pd.read_parquet("../../data/external/sample_data.parquet")
df_raw["channel"] = df_raw["channel"].astype(str).str.strip().str.capitalize()

# ==========================================
# PERGUNTA 1: Top 5 NCMs de Maior Risco
# ==========================================
print("\n--- Analisando Risco por NCM ---")
# Filtra NCMs com no mínimo 30 operações para ter relevância estatística
ncm_counts = df_raw.groupby("ncm_code")["channel"].value_counts().unstack(fill_value=0)
ncm_counts["Total"] = ncm_counts.sum(axis=1)

if "Vermelho" in ncm_counts.columns:
    ncm_counts["Taxa_Vermelho_%"] = (ncm_counts["Vermelho"] / ncm_counts["Total"]) * 100
    top_5_ncm = (
        ncm_counts[ncm_counts["Total"] > 30]
        .sort_values("Taxa_Vermelho_%", ascending=False)
        .head(5)
    )
    print("\nTop 5 NCMs com maior probabilidade de Canal Vermelho:")
    print(top_5_ncm[["Total", "Vermelho", "Taxa_Vermelho_%"]])
else:
    print("Canal Vermelho não encontrado na base de dados para análise de NCM.")

# ==========================================
# PERGUNTA 3: Impacto do Modo de Transporte
# ==========================================
print("\n--- Analisando Modo de Transporte ---")
df_transp = df_raw.fillna({"transport_mode_pt": "DESCONHECIDO"})
transp_cross = (
    pd.crosstab(df_transp["transport_mode_pt"], df_transp["channel"], normalize="index")
    * 100
)

plt.figure(figsize=(10, 6))
transp_cross.plot(kind="bar", stacked=True, colormap="Set2", figsize=(10, 6))
plt.title("Distribuição de Canais por Modo de Transporte (%)")
plt.ylabel("Porcentagem (%)")
plt.xlabel("Modo de Transporte")
plt.legend(title="Canal", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "q3_transport_impact.png")
plt.close()
print("Gráfico do Modo de Transporte salvo.")

# ==========================================
# PERGUNTA 4: Porte da Empresa
# ==========================================
print("\n--- Analisando Porte da Empresa ---")
df_size = df_raw.fillna({"consignee_company_size": "DESCONHECIDO"})
size_cross = (
    pd.crosstab(
        df_size["consignee_company_size"], df_size["channel"], normalize="index"
    )
    * 100
)

plt.figure(figsize=(10, 6))
size_cross.plot(kind="bar", stacked=True, colormap="Set3", figsize=(10, 6))
plt.title("Distribuição de Canais por Porte da Empresa Importadora (%)")
plt.ylabel("Porcentagem (%)")
plt.xlabel("Porte da Empresa")
plt.legend(title="Canal", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "q4_company_size_impact.png")
plt.close()
print("Gráfico do Porte da Empresa salvo.")

# ==========================================
# PERGUNTA 7: Interpretabilidade com SHAP
# ==========================================
print("\n--- Gerando Explicabilidade do Modelo (SHAP) ---")
df_train = pd.read_parquet("../../data/processed/train.parquet")

# 1. Lista segura de colunas para dropar (incluindo o yearmonth que causou o erro)
cols_to_drop = ["registry_date", "document_number", "yearmonth"]
cols_to_drop = [c for c in cols_to_drop if c in df_train.columns]
df_train = df_train.drop(columns=cols_to_drop)

y_train = df_train["channel"]
X_train = df_train.drop(columns=["channel"])

# 2. Varredura de segurança: remove qualquer outra coluna datetime escondida
date_cols = X_train.select_dtypes(include=["datetime", "datetimetz", "<M8[ns]"]).columns
X_train = X_train.drop(columns=date_cols)

# 3. Converter categóricas para códigos numéricos para o SHAP ler
for col in X_train.select_dtypes(include=["object", "category"]).columns:
    X_train[col] = LabelEncoder().fit_transform(X_train[col].astype(str))

le_target = LabelEncoder()
y_train_enc = le_target.fit_transform(y_train)

# Treinar um proxy LightGBM idêntico ao nosso (balanceado)
model = LGBMClassifier(class_weight="balanced", random_state=42, verbose=-1)
model.fit(X_train, y_train_enc)

# Gerar valores SHAP (Usamos uma amostra de 1000 linhas para ser rápido)
explainer = shap.TreeExplainer(model)
X_sample = X_train.sample(1000, random_state=42)
shap_values = explainer.shap_values(X_sample)

# Descobrir qual índice do LabelEncoder é o canal "Vermelho" ou "Cinza" (Risco)
classes = list(le_target.classes_)
risk_idx = classes.index("Vermelho") if "Vermelho" in classes else 1

plt.figure(figsize=(12, 8))
# Plota a importância global das features para prever o Canal de Risco
shap.summary_plot(shap_values[risk_idx], X_sample, plot_type="bar", show=False)
plt.title(f"SHAP Feature Importance Global - Risco ({classes[risk_idx]})")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "q7_shap_feature_importance.png")
plt.close()

print(
    f"✅ Gráfico SHAP salvo com sucesso! Ele responde claramente quais as variáveis que mais influenciam o Canal {classes[risk_idx]}."
)
