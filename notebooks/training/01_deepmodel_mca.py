import pandas as pd
import numpy as np
import warnings
from matplotlib import pyplot as plt
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
import torchmetrics

import lightning as L
import mlflow

warnings.filterwarnings("ignore")
SEED = 42
L.seed_everything(SEED)

# load dataset
df_train = pd.read_parquet("../../data/processed/train.parquet")
df_test = pd.read_parquet("../../data/processed/test.parquet")

cols_to_drop = ["channel", "registry_date", "yearmonth", "document_number"]

X_train_raw = df_train.drop(columns=[c for c in cols_to_drop if c in df_train.columns])
y_train_raw = df_train["channel"]

X_test_raw = df_test.drop(columns=[c for c in cols_to_drop if c in df_test.columns])
y_test_raw = df_test["channel"]

le = LabelEncoder()
y_train_full = le.fit_transform(y_train_raw)
y_test = le.transform(y_test_raw)
num_classes = len(le.classes_)

class_weights = compute_class_weight(
    "balanced", classes=np.unique(y_train_full), y=y_train_full
)
weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

# preprocessing pipe
num_feat = X_train_raw.select_dtypes(include=np.number).columns.tolist()
cat_feat = X_train_raw.select_dtypes(include=["object", "category"]).columns.tolist()

cat_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        (
            "encoder",
            OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
        ),
    ]
)

num_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

transform = ColumnTransformer(
    [
        ("num", num_pipeline, num_feat),
        ("cat", cat_pipeline, cat_feat),
    ]
)

print("Transformando dados de forma vetorizada (Rápido)...")
X_train_full_tf = transform.fit_transform(X_train_raw)
X_test_tf = transform.transform(X_test_raw)

X_train_tf, X_val_tf, y_train, y_val = train_test_split(
    X_train_full_tf,
    y_train_full,
    test_size=0.2,
    random_state=SEED,
    stratify=y_train_full,
)


# dataset & dataloader
class FastTensorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = FastTensorDataset(X_train_tf, y_train)
val_dataset = FastTensorDataset(X_val_tf, y_val)
test_dataset = FastTensorDataset(X_test_tf, y_test)

train_loader = DataLoader(
    train_dataset, batch_size=256, shuffle=True, pin_memory=True, num_workers=2
)
val_loader = DataLoader(
    val_dataset, batch_size=256, shuffle=False, pin_memory=True, num_workers=2
)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, pin_memory=False)


class FeatureAttention(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        reduction_ratio = max(input_size // 4, 16)

        self.attention_gate = nn.Sequential(
            nn.Linear(input_size, reduction_ratio),
            nn.BatchNorm1d(reduction_ratio),
            nn.ReLU(),
            nn.Linear(reduction_ratio, input_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        attention_weights = self.attention_gate(x)
        return x * attention_weights


class BaseNetwork(nn.Module):
    def __init__(
        self, input_size: int, num_classes: int, hidden_sizes: list = [512, 256, 128]
    ):
        super().__init__()

        self.attention = FeatureAttention(input_size)

        layers = []
        layer_sizes = [input_size] + hidden_sizes

        for i in range(1, len(layer_sizes)):
            layers += [
                nn.Linear(layer_sizes[i - 1], layer_sizes[i]),
                nn.BatchNorm1d(layer_sizes[i]),
                nn.ReLU(),
                nn.Dropout(0.3),
            ]
        layers += [nn.Linear(layer_sizes[-1], num_classes)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x_attended = self.attention(x)
        return self.network(x_attended)


# PyTorch Lightning Module
class DeepModel(L.LightningModule):
    def __init__(self, model, class_weights, num_classes, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.learning_rate = learning_rate
        self.register_buffer("class_weights", class_weights)

        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

        self.train_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = CrossEntropyLoss(weight=self.class_weights)(logits, y)

        self.train_acc(logits, y)
        self.train_f1(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = CrossEntropyLoss(weight=self.class_weights)(logits, y)

        self.val_acc(logits, y)
        self.val_f1(logits, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = CrossEntropyLoss(weight=self.class_weights)(logits, y)

        self.test_acc(logits, y)
        self.test_f1(logits, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_f1", self.test_f1, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            logits = self(x)
            return logits.argmax(dim=1)


# mlflow+ lightning
input_size = X_train_tf.shape[1]
print(f"Features de entrada: {input_size}")

model_torch = BaseNetwork(input_size=input_size, num_classes=num_classes)
lightning_model = DeepModel(
    model_torch, class_weights=weights_tensor, num_classes=num_classes
)

early_stopping = L.pytorch.callbacks.EarlyStopping(
    monitor="val_loss", patience=15, mode="min"
)
checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
    monitor="val_f1", mode="max", save_top_k=3
)

trainer = L.Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=400,
    precision="16-mixed",
    callbacks=[early_stopping, checkpoint_callback],
    enable_progress_bar=True,
)

mlflow.set_experiment("case_aduaneiro_deeplearning")
mlflow.pytorch.autolog()

with mlflow.start_run():
    print("Iniciando treinamento da Rede Neural...")
    trainer.fit(lightning_model, train_loader, val_loader)

# evaluation
print("\nAvaliação no conjunto de Teste:")
trainer.test(lightning_model, dataloaders=test_loader, ckpt_path="best")

lightning_model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(lightning_model.device)
        preds = lightning_model.predict(x)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.numpy())

from sklearn.metrics import classification_report, confusion_matrix

print("\nRelatório de Classificação:")
print(classification_report(all_labels, all_preds, target_names=le.classes_))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_,
)
plt.title("Matriz de Confusão - Deep Learning (C/ Attention)")
plt.ylabel("Real")
plt.xlabel("Predito")
plt.tight_layout()
plt.savefig("confusion_matrix_dl.png")
print("Matriz de confusão salva como 'confusion_matrix_dl.png'")

# save model
torch.save(lightning_model.state_dict(), "../../models/best_deep_model.pth")
