import pandas as pd
import plotly.express as px
import plotly.io as pio
from pathlib import Path
from kmodes.kmodes import KModes
import prince

# 1. Configuração
OUTPUT_DIR = Path("eda_html_reports")
OUTPUT_DIR.mkdir(exist_ok=True)
pio.templates.default = "plotly_white"

# Cores oficiais para os canais (Fácil identificação visual)
CHANNEL_COLORS = {
    "Verde": "#2ecc71",  # Verde
    "Amarelo": "#f1c40f",  # Amarelo
    "Vermelho": "#e74c3c",  # Vermelho
    "Cinza": "#34495e",  # Cinza Escuro
    "Unknown": "#95a5a6",  # Cinza Claro
}


# 2. Preparação e Limpeza
def reduce_cardinality(df, cols, top_n=15):
    """Agrupa categorias raras em 'Other' para o MCA não explodir."""
    df_reduced = df.copy()
    for col in cols:
        if col in df_reduced.columns:
            df_reduced[col] = df_reduced[col].fillna("Unknown")
            top_values = df_reduced[col].value_counts().nlargest(top_n).index
            df_reduced[col] = df_reduced[col].where(
                df_reduced[col].isin(top_values), "Other"
            )
    return df_reduced


# Carregar Dados
df = pd.read_parquet("../../data/external/sample_data.parquet").drop(
    "document_number", axis=1
)
TARGET_COL = "channel"

# Colunas Categoricas para Análise
cols_to_analyze = [
    "transport_mode_pt",
    "ncm_code",
    "shipper_name",
    "consignee_code",
    "clearance_place",
]

# 1. Limpeza de Cardinalidade
print("1. Processando dados (Redução de Cardinalidade)...")
df_work = reduce_cardinality(df, cols_to_analyze, top_n=20)

# Garantir que o target não tenha nulos para o plot
df_work["Target_Channel"] = df[TARGET_COL].fillna("Unknown")

# 3. Modelagem (K-Modes + MCA)

# K-MODES
print("2. Executando K-Modes (Clusterização)...")
km = KModes(n_clusters=2, init="Huang", n_init=3, verbose=0)
clusters = km.fit_predict(df_work[cols_to_analyze])
df_work["Cluster_Label"] = clusters.astype(str)

# MCA (Redução de Dimensão)
print("3. Executando MCA (Coordenadas X, Y)...")
mca = prince.MCA(n_components=2, n_iter=3, random_state=42, engine="sklearn")
mca_coords = mca.fit_transform(df_work[cols_to_analyze])

# Adiciona coordenadas ao dataframe
df_work["MCA_X"] = mca_coords[0]
df_work["MCA_Y"] = mca_coords[1]

# 4. Geração dos Gráficos (3 Arquivos)
print("4. Gerando Gráficos...")

common_hover = ["ncm_code", "transport_mode_pt", "shipper_name"]

# ARQUIVO 1: Apenas Clusters (O que o algortimo viu)
fig1 = px.scatter(
    df_work,
    x="MCA_X",
    y="MCA_Y",
    color="Cluster_Label",
    title="Visão 1: Clusters Identificados pelo K-Modes",
    hover_data=common_hover,
    opacity=0.7,
)
fig1.write_html(OUTPUT_DIR / "view1_clusters_only.html")
print(" -> Salvo: view1_clusters_only.html")

# ARQUIVO 2: Apenas Target (A realidade de risco)
fig2 = px.scatter(
    df_work,
    x="MCA_X",
    y="MCA_Y",
    color="Target_Channel",
    title="Visão 2: Distribuição Real dos Canais (Risco)",
    color_discrete_map=CHANNEL_COLORS,  # Usa as cores oficiais
    hover_data=common_hover,
    opacity=0.7,
)
fig2.write_html(OUTPUT_DIR / "view2_target_only.html")
print(" -> Salvo: view2_target_only.html")

# ARQUIVO 3: Combinado (Cluster=Cor, Canal=Forma)
fig3 = px.scatter(
    df_work,
    x="MCA_X",
    y="MCA_Y",
    color="Cluster_Label",
    symbol="Target_Channel",
    title="Visão 3: Sobreposição (Cor=Cluster, Forma=Risco)",
    hover_data=common_hover,
    opacity=0.8,
    height=800,
)
fig3.update_traces(marker=dict(size=8))  # Aumenta o ponto para ver o símbolo melhor
fig3.write_html(OUTPUT_DIR / "view3_combined_analysis.html")
print(" -> Salvo: view3_combined_analysis.html")
