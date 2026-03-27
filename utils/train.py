# python -m utils.train soccernet
# python -m utils.train epickitchen
# python -m utils.train soccernet --epochs 30 --lr 1e-3 --batch_size 64
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import load_config
from model.epfe_cached import EPFECached
from data.dataset import EPFEDataset

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def train(dataset: str, epochs: int = 20, lr: float = 1e-3, batch_size: int = 64):
    config = load_config(dataset)

    train_set = EPFEDataset(config, dataset, split="train")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

    epfe = EPFECached(config)
    epfe.train()

    # Gewichtete BCE gegen Class Imbalance
    pos_weight = torch.tensor([train_set.pos_weight]).to(epfe.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(epfe.parameters(), lr=lr)

    best_loss = float("inf")
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"epfe_{dataset}.pt")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for features, labels in train_loader:
            features = features.to(epfe.device)   # (batch, buffer_size, 512)
            labels   = labels.to(epfe.device)     # (batch, buffer_size)

            optimizer.zero_grad()
            scores = epfe.forward_train(features)  # (batch, buffer_size)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch:>3}/{epochs} — Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(epfe.state_dict(), checkpoint_path)
            print(f"  -> Checkpoint gespeichert: {checkpoint_path}")

    print(f"\nTraining abgeschlossen. Bester Loss: {best_loss:.4f}")
    print(f"Eval: python -m eval.eval_cached {dataset} test --weights {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["epickitchen", "soccernet"])
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int,   default=64)
    args = parser.parse_args()
    train(args.dataset, args.epochs, args.lr, args.batch_size)