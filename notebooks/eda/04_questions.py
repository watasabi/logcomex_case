import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

sns.set_theme(style="whitegrid")
OUTPUT_DIR = Path("../../reports/figures/business")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Carregando dados...")
df_raw = pd.read_parquet("../../data/external/sample_data.parquet")
df_raw["channel"] = df_raw["channel"].astype(str).str.strip().str.capitalize()

print("\n--- Analisando Risco por NCM ---")
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

print("\n--- Gerando Explicabilidade do Modelo (SHAP) ---")

df_train = pd.read_parquet("../../data/processed/train_without_mca.parquet")
cols_to_drop = ["channel", "registry_date", "yearmonth", "document_number"]
X_train = df_train.drop(columns=[c for c in cols_to_drop if c in df_train.columns])

X_sample = X_train.sample(1000, random_state=42)

print("Carregando o modelo salvo (Ensemble/Target Encoding)...")
pipeline = joblib.load("../../models/ensemble_model.pkl")

transformer = pipeline.named_steps["transform"]
ensemble_model = pipeline.named_steps["model"]
lgbm_model = ensemble_model.named_estimators_["lgbm"]

X_sample_transformed = transformer.transform(X_sample)

raw_feature_names = transformer.get_feature_names_out()
feature_names = [
    name.replace("num__", "").replace("cat__", "") for name in raw_feature_names
]

X_sample_df = pd.DataFrame(X_sample_transformed, columns=feature_names)

print("Calculando SHAP values a partir do LightGBM otimizado...")
explainer = shap.TreeExplainer(lgbm_model)
shap_values = explainer.shap_values(X_sample_df)

le = LabelEncoder()
le.fit(df_train["channel"])
classes = list(le.classes_)
risk_idx = classes.index("Vermelho") if "Vermelho" in classes else 1

if isinstance(shap_values, list):
    shap_risk = shap_values[risk_idx]
else:
    shap_risk = shap_values[:, :, risk_idx]

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_risk, X_sample_df, plot_type="bar", show=False)
plt.title(f"SHAP Feature Importance (Target Encoding) - Risco ({classes[risk_idx]})")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "q7_shap_feature_importance.png")
plt.close()

print(
    f"✅ Gráfico SHAP gerado com sucesso! Ele responde quais variáveis mais influenciam o Canal {classes[risk_idx]}."
)
