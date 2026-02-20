import pandas as pd
import plotly.express as px
import plotly.io as pio
from pathlib import Path
from kmodes.kmodes import KModes
from sklearn.manifold import TSNE

# config
OUTPUT_DIR = Path("eda_html_reports")
OUTPUT_DIR.mkdir(exist_ok=True)
pio.templates.default = "plotly_white"

CHANNEL_COLORS = {
    "Verde": "#2ecc71",  # Verde
    "Amarelo": "#f1c40f",  # Amarelo
    "Vermelho": "#e74c3c",  # Vermelho
    "Cinza": "#34495e",  # Cinza Escuro
    "Unknown": "#95a5a6",  # Cinza Claro
}


def reduce_cardinality(df, cols, top_n=15):
    """Agrupa categorias raras em 'Other'."""
    df_reduced = df.copy()
    for col in cols:
        if col in df_reduced.columns:
            df_reduced[col] = df_reduced[col].fillna("Unknown")
            top_values = df_reduced[col].value_counts().nlargest(top_n).index
            df_reduced[col] = df_reduced[col].where(
                df_reduced[col].isin(top_values), "Other"
            )
    return df_reduced


df = pd.read_parquet("../../data/external/sample_data.parquet").drop(
    "document_number", axis=1
)
TARGET_COL = "channel"

cols_to_analyze = [
    "transport_mode_pt",
    "ncm_code",
    "shipper_name",
    "consignee_code",
    "clearance_place",
]

print("1. Processando dados (Redução de Cardinalidade)...")
df_work = reduce_cardinality(df, cols_to_analyze, top_n=20)
df_work["Target_Channel"] = df[TARGET_COL].fillna("Unknown")


print("2. Executando K-Modes (Clusterização)...")
km = KModes(n_clusters=2, init="Huang", n_init=3, verbose=0)
clusters = km.fit_predict(df_work[cols_to_analyze])
df_work["Cluster_Label"] = clusters.astype(str)

print("3. Preparando dados para t-SNE (One-Hot Encoding)...")
# t-SNE precisa de números. Transformamos texto em colunas binárias (0 ou 1)
df_encoded = pd.get_dummies(df_work[cols_to_analyze], drop_first=True)

print("4. Executando t-SNE (Isso pode demorar um pouco)...")
tsne = TSNE(
    n_components=2,
    verbose=1,
    perplexity=30,
    max_iter=400,
    init="pca",
    learning_rate="auto",
)
tsne_results = tsne.fit_transform(df_encoded)

df_work["TSNE_X"] = tsne_results[:, 0]
df_work["TSNE_Y"] = tsne_results[:, 1]

print("5. Gerando Gráficos...")

common_hover = ["ncm_code", "transport_mode_pt", "shipper_name"]

fig1 = px.scatter(
    df_work,
    x="TSNE_X",
    y="TSNE_Y",
    color="Cluster_Label",
    title="Visão t-SNE 1: Clusters Identificados (K-Modes)",
    hover_data=common_hover,
    opacity=0.7,
)
fig1.write_html(OUTPUT_DIR / "tsne_view1_clusters.html")
print(" -> Salvo: tsne_view1_clusters.html")

fig2 = px.scatter(
    df_work,
    x="TSNE_X",
    y="TSNE_Y",
    color="Target_Channel",
    title="Visão t-SNE 2: Distribuição Real dos Canais",
    color_discrete_map=CHANNEL_COLORS,
    hover_data=common_hover,
    opacity=0.7,
)
fig2.write_html(OUTPUT_DIR / "tsne_view2_target.html")
print(" -> Salvo: tsne_view2_target.html")

fig3 = px.scatter(
    df_work,
    x="TSNE_X",
    y="TSNE_Y",
    color="Cluster_Label",  # Cor = Cluster
    symbol="Target_Channel",  # Forma = Risco
    title="Visão t-SNE 3: Combinado (Cor=Cluster, Forma=Risco)",
    hover_data=common_hover,
    opacity=0.8,
    height=800,
)
fig3.update_traces(marker=dict(size=8))
fig3.write_html(OUTPUT_DIR / "tsne_view3_combined.html")
print(" -> Salvo: tsne_view3_combined.html")

print("\nConcluído! Verifique os arquivos 'tsne_*' na pasta.")
