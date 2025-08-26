import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
import optuna

# === Your model, adapted for filter count ===
class ChessPieceClassifier(nn.Module):
    def __init__(self, num_filters=128):
        super(ChessPieceClassifier, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.head = nn.Sequential(
            nn.Conv2d(512, num_filters, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(num_filters, 13, kernel_size=1)  # 13 classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        x = x.permute(0, 2, 3, 1).reshape(x.size(0), 64, 13)
        return x

# === Objective function ===
def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    num_filters = trial.suggest_int("num_filters", 64, 256, step=32)

    # Datasets
    train_ids = annotations['splits']['chessred2k']['train']['image_ids']
    val_ids = annotations['splits']['chessred2k']['val']['image_ids']
    train_dataset = ChessFENDataset(train_ids, image_to_pieces, image_id_to_path)
    val_dataset = ChessFENDataset(val_ids, image_to_pieces, image_id_to_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)

    # Model
    model = ChessPieceClassifier(num_filters=num_filters)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Train for 1 epoch (or adjust)
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out.view(-1, 13), y.view(-1))
        loss.backward()
        optimizer.step()
        break  # Optional: only one mini-batch per trial for fast tuning

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()
            break  # Optional: speed-up

    accuracy = correct / total
    return 1.0 - accuracy  # Optuna minimizes by default

# === Run the study ===
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

print("Best parameters:", study.best_params)
print("Best validation accuracy:", 1 - study.best_value)
 #make sure to do !pip install optuna for BO library
 