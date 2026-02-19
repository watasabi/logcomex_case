import pandas as pd
from pathlib import Path

# config
INPUT_DATA = "../../data/processed/01_data_mca.parquet"
TRAIN_OUTPUT = "../../data/processed/train.parquet"
TEST_OUTPUT = "../../data/processed/test.parquet"
TIME_COL = "yearmonth"
TEST_SIZE = 0.2

print("--- Iniciando Script 2: Split Temporal ---")
df = pd.read_parquet(INPUT_DATA)

# NOTE order by date
df = df.sort_values(by=TIME_COL).reset_index(drop=True)

split_index = int(len(df) * (1 - TEST_SIZE))

df_train = df.iloc[:split_index].copy()
df_test = df.iloc[split_index:].copy()

print(f"Data final do Treino: {df_train[TIME_COL].max()}")
print(f"Data inicial do Teste: {df_test[TIME_COL].min()}")
print(f"Tamanho Treino: {len(df_train)} linhas")
print(f"Tamanho Teste: {len(df_test)} linhas")

# save datesets
Path(TRAIN_OUTPUT).parent.mkdir(parents=True, exist_ok=True)
df_train.to_parquet(TRAIN_OUTPUT)
df_test.to_parquet(TEST_OUTPUT)
