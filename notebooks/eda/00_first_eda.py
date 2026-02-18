import pandas as pd
import plotly.express as px
import plotly.io as pio
from pathlib import Path

OUTPUT_DIR = Path("eda_html_reports")
OUTPUT_DIR.mkdir(exist_ok=True)

pio.templates.default = "plotly_white"

COLOR_MAP = {
    "Verde": "#2ecc71",
    "Amarelo": "#f1c40f",
    "Vermelho": "#e74c3c",
    "Cinza": "#34495e",
}

TOP_N_CATEGORIES = 15

df = pd.read_parquet("../../data/external/sample_data.parquet")

TARGET_COL = "channel"
TIME_COLS = ["yearmonth", "registry_date"]
CATEGORICAL_COLS = [
    "clearance_place_dispatch",
    "clearance_place_entry",
    "consignee_code",
    "consignee_company_size",
    "consignee_name",
    "clearance_place",
    "transport_mode_pt",
    "ncm_code",
    "shipper_name",
    "country_origin_code",
]

for col in TIME_COLS:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")


def save_interactive_plot(fig, filename):
    path = OUTPUT_DIR / f"{filename}.html"
    fig.write_html(path)
    print(f" [+] Gerado: {path}")


def plot_categorical_interactive(df, col, target):
    n_unique = df[col].nunique()

    if n_unique > TOP_N_CATEGORIES:
        top_counts = df[col].value_counts().nlargest(TOP_N_CATEGORIES).index
        df_plot = df[df[col].isin(top_counts)].copy()
        title_suffix = f"(Top {TOP_N_CATEGORIES} de {n_unique})"
    else:
        df_plot = df.copy()
        title_suffix = ""

    category_order = df_plot[col].value_counts().index.tolist()

    fig = px.histogram(
        df_plot,
        x=col,
        color=target,
        barmode="group",
        title=f"Distribuição: {col} por {target} {title_suffix}",
        category_orders={col: category_order},
        color_discrete_map=COLOR_MAP,
        text_auto=True,
    )

    fig.update_layout(
        xaxis_title=col,
        yaxis_title="Contagem",
        legend_title="Canal",
        hovermode="x unified",
    )

    save_interactive_plot(fig, f"dist_{col}")


def plot_time_evolution(df, time_col, target):
    # Correção aqui: freq="ME" ao invés de "M"
    df_grouped = (
        df.groupby([pd.Grouper(key=time_col, freq="ME"), target])
        .size()
        .reset_index(name="count")
    )

    fig = px.line(
        df_grouped,
        x=time_col,
        y="count",
        color=target,
        title=f"Evolução Temporal ({time_col})",
        markers=True,
        color_discrete_map=COLOR_MAP,
    )

    fig.update_xaxes(rangeslider_visible=True)
    save_interactive_plot(fig, "time_evolution")


print("--- Gerando Relatórios ---")

if "yearmonth" in df.columns:
    plot_time_evolution(df, "yearmonth", TARGET_COL)
elif "registry_date" in df.columns:
    plot_time_evolution(df, "registry_date", TARGET_COL)

for col in CATEGORICAL_COLS:
    if col in df.columns:
        plot_categorical_interactive(df, col, TARGET_COL)

fig_null = px.imshow(df.isnull(), title="Mapa de Nulos")
save_interactive_plot(fig_null, "missing_values_heatmap")

print(f"\n--- Finalizado em '{OUTPUT_DIR}' ---")
