import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from kmodes.kmodes import KModes
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import OrdinalEncoder

# Configuração Visual
plt.rcParams["figure.figsize"] = (18, 12)  # Maior para acomodar os subplots
plt.style.use("seaborn-v0_8-whitegrid")


# 1. Preparação e Redução de Cardinalidade
def load_and_prep_data():
    df = pd.read_parquet("../../data/external/sample_data.parquet").drop(
        "document_number", axis=1
    )

    cols: list[str] = [
        "clearance_place_dispatch",  # 12
        "consignee_code",  # 3907 (Alta)
        "transport_mode_pt",  # 7
        "ncm_code",  # 28
        "shipper_name",  # 1920 (Alta)
    ]

    df_clean = df[cols].copy()

    cols_alta_card: list[str] = ["consignee_code", "shipper_name", "ncm_code"]

    for c in cols:
        df_clean[c] = df_clean[c].fillna("Unknown").astype(str)

        if c in cols_alta_card:
            top_n = df_clean[c].value_counts().nlargest(50).index
            df_clean[c] = df_clean[c].where(df_clean[c].isin(top_n), "OUTROS")

    return df_clean


print("1. Carregando e tratando alta cardinalidade...")
df_work = load_and_prep_data()

# 2. Encoding para a Métrica de Distância
print("2. Aplicando OrdinalEncoder para cálculo da Silhueta...")
# NOTE medir a distância de Hamming
encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(df_work)

# 3. Loop Range de Clusters
range_n_clusters = [2, 3, 4, 5]  # Testando de 2 a 5 clusters

fig, axes = plt.subplots(2, 2)
fig.suptitle(
    "Análise de Silhueta K-Modes para Vários Valores de K (Métrica: Hamming)",
    fontsize=16,
    fontweight="bold",
)
axes = axes.flatten()  # Facilita a iteração

scores_medios = []

print("3. Iniciando iterações do K-Modes...")

for idx, n_clusters in enumerate(range_n_clusters):
    ax = axes[idx]
    print(f" -> Avaliando K={n_clusters}...")

    # Treina o K-Modes
    km = KModes(
        n_clusters=n_clusters, init="Huang", n_init=3, verbose=0, random_state=42
    )
    cluster_labels = km.fit_predict(df_work)

    # Calcula Silhouette Médio
    silhouette_avg = silhouette_score(X_encoded, cluster_labels, metric="hamming")
    scores_medios.append((n_clusters, silhouette_avg))

    # Calcula Silhouette individual
    sample_silhouette_values = silhouette_samples(
        X_encoded, cluster_labels, metric="hamming"
    )

    ax.set_xlim([-0.1, 1])
    ax.set_ylim([0, len(X_encoded) + (n_clusters + 1) * 10])
    y_lower = 10

    for i in range(n_clusters):
        ith_cluster_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_values.sort()

        size_cluster_i = ith_cluster_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.set_title(f"K = {n_clusters} | Score Médio: {silhouette_avg:.3f}")
    ax.set_xlabel("Coeficiente de Silhueta")
    ax.set_ylabel("Label do Cluster")

    # Linha vertical da média
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_yticks([])  # Limpa ticks do eixo Y
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

plt.tight_layout(
    rect=[0, 0.03, 1, 0.95]
)  # Ajuste para não encavalar o título principal
plt.show()

# Resumo Final
print("\n=== RESUMO DOS SCORES ===")
for k, score in scores_medios:
    print(f"K={k}: {score:.4f}")

# save this report and graphs
with open("silhouette_report.txt", "w") as f:
    f.write("=== RESUMO DOS SCORES ===\n")
    for k, score in scores_medios:
        f.write(f"K={k}: {score:.4f}\n")
