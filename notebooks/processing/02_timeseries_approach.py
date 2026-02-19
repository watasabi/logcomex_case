import pandas as pd
from pathlib import Path

# Configuração
INPUT_DATA = "../../data/external/sample_data.parquet"
TRAIN_OUTPUT = "../../data/processed/ts_train.parquet"
TEST_OUTPUT = "../../data/processed/ts_test.parquet"
TEST_DAYS = 90
print("--- Montando Dataset de Séries Temporais ---")
df = pd.read_parquet(INPUT_DATA)

df["registry_date"] = pd.to_datetime(df["registry_date"]).dt.date

# pivot (data x canal) e reamostrar D
print("Agrupando por dia...")
df_ts = df.groupby(["registry_date", "channel"]).size().unstack(fill_value=0)
df_ts.index = pd.to_datetime(df_ts.index)
df_ts = df_ts.resample("D").sum().fillna(0)

expected_channels = ["VERDE", "AMARELO", "VERMELHO", "CINZA"]
for ch in expected_channels:
    if ch not in df_ts.columns:
        df_ts[ch] = 0

df_ts = df_ts[expected_channels].astype(float)

# 3. Split Temporal
split_idx = len(df_ts) - TEST_DAYS
df_train = df_ts.iloc[:split_idx]
df_test = df_ts.iloc[split_idx:]

print(f"Treino: {df_train.shape} dias")
print(f"Teste: {df_test.shape} dias")

Path(TRAIN_OUTPUT).parent.mkdir(parents=True, exist_ok=True)
df_train.to_parquet(TRAIN_OUTPUT)
df_test.to_parquet(TEST_OUTPUT)
print("Dados de forecasting salvos com sucesso!")


import plotly.express as px

df_plot = df_ts.reset_index().melt(
    id_vars="registry_date", var_name="channel", value_name="count"
)
fig = px.line(
    df_plot,
    x="registry_date",
    y="count",
    color="channel",
    title="Séries Temporais por Canal",
)
fig.update_layout(xaxis_title="Data", yaxis_title="Volume Diário", legend_title="Canal")
fig.show()
