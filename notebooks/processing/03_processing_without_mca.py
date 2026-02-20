import pandas as pd
from pathlib import Path

# config
INPUT_DATA = "../../data/external/sample_data.parquet"
OUTPUT_DATA = "../../data/processed/01_data_cleaned.parquet"  # Mudei o nome para refletir que não tem MCA
ID_COL = "document_number"


def process_cardinality(
    df, cols, top_n=100
):  # Podemos aumentar o top_n pois o TargetEncoder lida bem com isso
    """Mantém o Top N categorias e agrupa o resto como 'OUTROS'."""
    df_clean = df.copy()
    for c in cols:
        if c in df_clean.columns:
            df_clean[c] = df_clean[c].fillna("UNKNOWN").astype(str)
            top_values = df_clean[c].value_counts().nlargest(top_n).index
            df_clean[c] = df_clean[c].where(df_clean[c].isin(top_values), "OUTROS")
    return df_clean


print("--- Iniciando Script 1: Tratamento de Cardinalidade Baseline ---")
df = pd.read_parquet(INPUT_DATA)

if ID_COL in df.columns:
    df_ids = df[[ID_COL]].copy()
    df = df.drop(columns=[ID_COL])
else:
    df_ids = pd.DataFrame(index=df.index)

cols_to_reduce = [
    "consignee_code",
    "consignee_name",
    "shipper_name",
    "country_origin_code",
]
df_reduced = process_cardinality(df, cols_to_reduce, top_n=100)

cat_cols = df_reduced.select_dtypes(include=["object", "category"]).columns
cat_cols = [c for c in cat_cols if c != "channel"]
df_reduced[cat_cols] = df_reduced[cat_cols].fillna("UNKNOWN")

cols_to_keep = ["channel", "registry_date", "yearmonth"]
final_cols = [c for c in cols_to_keep if c in df_reduced.columns] + list(cat_cols)

# Tirando duplicatas da lista de colunas finais caso ocorram
final_cols = list(dict.fromkeys(final_cols))

df_final = pd.concat([df_ids, df_reduced[final_cols]], axis=1)

Path(OUTPUT_DATA).parent.mkdir(parents=True, exist_ok=True)
df_final.to_parquet(OUTPUT_DATA)
print(f"Dados salvos em: {OUTPUT_DATA}")
