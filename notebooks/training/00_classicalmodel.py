import pandas as pd
import numpy as np
import mlflow
import optuna
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
import warnings

warnings.filterwarnings("ignore")
SEED = 42

df_train = pd.read_parquet("../../data/processed/train.parquet")
df_test = pd.read_parquet("../../data/processed/test.parquet")

cols_to_drop = ["channel", "registry_date", "yearmonth", "document_number"]
X_train = df_train.drop(columns=[c for c in cols_to_drop if c in df_train.columns])
y_train_raw = df_train["channel"]

X_test = df_test.drop(columns=[c for c in cols_to_drop if c in df_test.columns])
y_test_raw = df_test["channel"]

le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)
y_test = le.transform(y_test_raw)

classes = np.unique(y_train)
weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weights_dict = dict(zip(classes, weights))

num_feat = X_train.select_dtypes(include=np.number).columns.tolist()

num_pipeline = Pipeline(
    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

transform = ColumnTransformer(
    [
        ("num", num_pipeline, num_feat),
    ]
)

mlflow.set_experiment("case_aduaneiro_mca")


def objective(trial):
    model_name = trial.suggest_categorical(
        "model_name", ["LGBMClassifier", "LogisticRegression", "RandomForestClassifier"]
    )

    if model_name == "LGBMClassifier":
        model = LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            num_leaves=trial.suggest_int("num_leaves", 10, 64),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            random_state=SEED,
            class_weight=class_weights_dict,
            n_jobs=-1,
            verbose=-1,
        )
    elif model_name == "LogisticRegression":
        model = LogisticRegression(
            C=trial.suggest_float("C", 0.01, 10, log=True),
            solver="lbfgs",
            max_iter=500,
            random_state=SEED,
            class_weight=class_weights_dict,
        )
    elif model_name == "RandomForestClassifier":
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 200),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            random_state=SEED,
            class_weight=class_weights_dict,
            n_jobs=-1,
        )

    model_pipeline = Pipeline([("transform", transform), ("model", model)])

    with mlflow.start_run(nested=True):
        model_pipeline.fit(X_train, y_train)
        preds = model_pipeline.predict(X_test)

        score_acc = accuracy_score(y_test, preds)
        score_f1 = f1_score(y_test, preds, average="macro")
        recall = recall_score(y_test, preds, average="macro")
        mlflow.log_metric("accuracy", score_acc)
        mlflow.log_metric("f1_macro", score_f1)
        mlflow.log_metric("recall_macro", recall)
        mlflow.log_params(trial.params)

        return score_f1


with mlflow.start_run(run_name="Optuna_Optimization"):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=300, show_progress_bar=True)

df_trials = study.trials_dataframe()

# 1. Extraindo hiperparâmetros do LGBM
lgbm_trials = df_trials[df_trials["params_model_name"] == "LGBMClassifier"].dropna(
    subset=["value"]
)
if not lgbm_trials.empty:
    best_lgbm = lgbm_trials.sort_values("value", ascending=False).iloc[0]
    lgbm_n_est = int(best_lgbm["params_n_estimators"])
    lgbm_lr = float(best_lgbm["params_learning_rate"])
    lgbm_leaves = int(best_lgbm["params_num_leaves"])
    lgbm_depth = int(best_lgbm["params_max_depth"])
else:
    lgbm_n_est, lgbm_lr, lgbm_leaves, lgbm_depth = 100, 0.1, 31, -1  # Fallback seguro

# 2. Extraindo hiperparâmetros da Regressão Logística
lr_trials = df_trials[df_trials["params_model_name"] == "LogisticRegression"].dropna(
    subset=["value"]
)
if not lr_trials.empty:
    best_lr = lr_trials.sort_values("value", ascending=False).iloc[0]
    lr_c = float(best_lr["params_C"])  # Garante que enviaremos apenas um float escalar
else:
    lr_c = 1.0  # Fallback seguro

# Instanciando os modelos
model_lgbm = LGBMClassifier(
    n_estimators=lgbm_n_est,
    learning_rate=lgbm_lr,
    num_leaves=lgbm_leaves,
    max_depth=lgbm_depth,
    class_weight=class_weights_dict,
    random_state=SEED,
    verbose=-1,
)

model_lr = LogisticRegression(
    C=lr_c,
    solver="lbfgs",
    max_iter=500,
    class_weight=class_weights_dict,
    random_state=SEED,
)

ensemble = VotingClassifier(
    estimators=[("lgbm", model_lgbm), ("lr", model_lr)], voting="soft"
)

ensemble_pipeline = Pipeline([("transform", transform), ("model", ensemble)])
ensemble_pipeline.fit(X_train, y_train)

preds_ensemble = ensemble_pipeline.predict(X_test)
print(f"F1-Macro do Ensemble: {f1_score(y_test, preds_ensemble, average='macro')}")
print(f"Recall do Ensemble: {recall_score(y_test, preds_ensemble, average='macro')}")

# save model
import joblib

joblib.dump(ensemble_pipeline, "../../models/ensemble_model.pkl")
