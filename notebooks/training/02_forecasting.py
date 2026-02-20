import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import mlflow
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
import lightning as L

sys.path.append("/home/rwp/code/logcomex_case/")
from src.ts_preprocessing import TimeSeriesPreprocessor

L.seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_WINDOW = 90
OUTPUT_WINDOW = 30
NUM_FEATURES = 4
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3


def calculate_smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / np.maximum(denominator, 1e-8)
    return 100.0 * np.mean(diff)


df_train = pd.read_parquet("../../data/processed/ts_train.parquet")
df_test = pd.read_parquet("../../data/processed/ts_test.parquet")

preprocessor = TimeSeriesPreprocessor(
    normalize=True,
    differencing=False,
    apply_filter=False,
    self_tune=False,
    window_size=INPUT_WINDOW,
    horizon=OUTPUT_WINDOW,
)

X_train_np, y_train_np, _ = preprocessor.fit_transform(df_train.values)
X_test_np = preprocessor.transform(df_test.values)

train_dataset = TensorDataset(
    torch.FloatTensor(X_train_np), torch.FloatTensor(y_train_np)
)
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, hidden, cell


class DotAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, hidden, encoder_outputs):
        hidden_last_layer = hidden[-1]
        attn_weights = torch.bmm(
            encoder_outputs.permute(1, 0, 2), hidden_last_layer.unsqueeze(2)
        ).squeeze(2)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.bmm(
            attn_weights.unsqueeze(1), encoder_outputs.permute(1, 0, 2)
        ).squeeze(1)
        output = torch.tanh(self.fc(torch.cat((context, hidden_last_layer), 1)))
        return output, attn_weights


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False)
        self.attention = DotAttention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell, encoder_outputs):
        lstm_output, (hidden, cell) = self.lstm(x, (hidden, cell))
        attn_output, attn_weights = self.attention(hidden, encoder_outputs)
        prediction = self.fc(attn_output)
        return prediction, hidden, cell, attn_weights


class Seq2SeqAttention(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, output_window, num_layers=1
    ):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers)
        self.decoder = Decoder(output_size, hidden_size, output_size, num_layers)
        self.output_window = output_window

    def forward(self, source, target=None, teacher_forcing_ratio=0.5):
        seq_len, batch_size, _ = source.shape
        outputs = torch.zeros(
            self.output_window,
            batch_size,
            self.decoder.fc.out_features,
            device=source.device,
        )
        attention_weights = torch.zeros(
            self.output_window, batch_size, seq_len, device=source.device
        )

        encoder_outputs, hidden, cell = self.encoder(source)
        decoder_input = source[-1].unsqueeze(0)

        for t in range(self.output_window):
            decoder_output, hidden, cell, attn_weights = self.decoder(
                decoder_input, hidden, cell, encoder_outputs
            )
            outputs[t] = decoder_output
            attention_weights[t] = attn_weights

            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = target[t].unsqueeze(0)
            else:
                decoder_input = decoder_output.unsqueeze(0)

        return outputs, attention_weights


# MLFLOW WORKFLOW & TRAINING
mlflow.set_experiment("forecasting_multivariado")

with mlflow.start_run():
    mlflow.log_params(
        {
            "input_window": INPUT_WINDOW,
            "output_window": OUTPUT_WINDOW,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
        }
    )

    model = Seq2SeqAttention(
        input_size=NUM_FEATURES,
        hidden_size=64,
        output_size=NUM_FEATURES,
        output_window=OUTPUT_WINDOW,
        num_layers=2,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("Treinando Modelo...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = (
                x_batch.permute(1, 0, 2).to(device),
                y_batch.permute(1, 0, 2).to(device),
            )
            optimizer.zero_grad()
            outputs, _ = model(x_batch, y_batch, teacher_forcing_ratio=0.5)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 5 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch}/{EPOCHS}], Loss: {avg_loss:.4e}")
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

    model.eval()
    model.cpu()

    last_60_days = df_train.values[-INPUT_WINDOW:]
    last_60_days_norm = preprocessor.scaler.transform(last_60_days)
    input_tensor = torch.FloatTensor(last_60_days_norm).unsqueeze(1)

    with torch.no_grad():
        output_tensor, attn_weights = model(input_tensor, teacher_forcing_ratio=0)

    preds_norm = output_tensor.squeeze(1).numpy()
    preds_real = np.maximum(preprocessor.inverse_transform(preds_norm), 0)
    real_values = df_test.values[:OUTPUT_WINDOW]

    # Métricas
    global_smape = calculate_smape(real_values, preds_real)
    global_r2 = r2_score(real_values, preds_real)
    mlflow.log_metrics({"SMAPE": global_smape, "R2_Score": global_r2})
    print(f"\nMétricas Finais -> SMAPE: {global_smape:.2f}% | R²: {global_r2:.4f}")

    channels = ["VERDE", "AMARELO", "VERMELHO", "CINZA"]
    fig_fc, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    for i in range(4):
        axes[i].plot(
            range(OUTPUT_WINDOW),
            real_values[:, i],
            label="Real",
            marker="o",
            linewidth=2,
        )
        axes[i].plot(
            range(OUTPUT_WINDOW),
            preds_real[:, i],
            label="Previsto",
            linestyle="--",
            marker="X",
            color="red",
        )
        axes[i].set_title(f"Previsão de Volume - Canal {channels[i]}")
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()
    fc_path = "multivariate_forecast.png"
    plt.savefig(fc_path)
    mlflow.log_artifact(fc_path)
    plt.close(fig_fc)

    attn_matrix = attn_weights.squeeze(
        1
    ).numpy()  # NOTE Shape: [OUTPUT_WINDOW, INPUT_WINDOW]
    fig_attn, ax = plt.subplots(figsize=(12, 6))
    cax = ax.imshow(attn_matrix, cmap="viridis", aspect="auto")
    fig_attn.colorbar(cax, label="Peso de Atenção")
    ax.set_title("Heatmap de Atenção (Como o Futuro olha para o Passado)")
    ax.set_xlabel("Lags no Passado (0 = Mais Antigo, 89 = Dia Anterior)")
    ax.set_ylabel("Dias Previstos no Futuro")

    heatmap_path = "attention_heatmap.png"
    plt.savefig(heatmap_path)
    mlflow.log_artifact(heatmap_path)
    plt.close(fig_attn)

    mean_attention = np.mean(attn_matrix, axis=0)
    top_lags = np.argsort(mean_attention)[-10:]  # NOTE Top 10 dias mais importantes

    fig_bar, ax = plt.subplots(figsize=(10, 5))
    ax.bar([str(lag) for lag in top_lags], mean_attention[top_lags], color="skyblue")
    ax.set_title("Top 10 Dias do Passado com Maior Influência Global")
    ax.set_xlabel("Índice do Lag (0 = Mais Antigo, 89 = Dia Anterior)")
    ax.set_ylabel("Média do Peso de Atenção")
    ax.grid(axis="y", linestyle="--")

    bar_path = "attention_top_lags.png"
    plt.savefig(bar_path)
    mlflow.log_artifact(bar_path)
    plt.close(fig_bar)

    os.remove(fc_path)
    os.remove(heatmap_path)
    os.remove(bar_path)

# save the model
model_path = "seq2seq_attention.pth"
torch.save(model.state_dict(), model_path)
mlflow.log_artifact(model_path)
os.remove(model_path)
