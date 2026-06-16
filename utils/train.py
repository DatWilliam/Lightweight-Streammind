import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from config import load_config
from model.epfe_mamba_cached import EPFECached
from data.dataset import EPFEDataset

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def focal_loss(logits, targets, gamma: float = 2.0, pos_weight=None):
    # weighted BCE + focal factor (1-pt)^gamma; gamma=0 -> plain weighted BCE
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    pt = torch.exp(-bce)
    return ((1 - pt) ** gamma * bce).mean()


def train(dataset: str, epochs: int = 20, lr: float = 1e-3, batch_size: int = 64,
          gamma: float = 2.0, patience: int = 0, use_mamba: bool = True):
    # trains the Mamba EPFE on cached CLIP features, saves best checkpoint by val-loss
    config = load_config(dataset)

    # build train + (optional) val loaders from cached .npz features
    train_set = EPFEDataset(config, dataset, split="train")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

    has_val = bool(getattr(config, "video_ids_val", []))
    val_loader = None
    if has_val:
        val_set = EPFEDataset(config, dataset, split="val")
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    epfe = EPFECached(config, use_mamba=use_mamba)
    epfe.train()

    # pos_weight rebalances ~99% negative frames in BCE
    pos_weight = torch.tensor([train_set.pos_weight]).to(epfe.device)
    optimizer = torch.optim.Adam(epfe.parameters(), lr=lr)

    best_metric = float("inf")
    metric_name = "val_loss" if has_val else "train_loss"
    epochs_no_improve = 0
    # checkpoint name reflects ablation: epfe_<dataset>.pt or epfe_<dataset>_nomamba.pt
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"epfe_{dataset}{'_nomamba' if not use_mamba else ''}.pt")

    for epoch in range(1, epochs + 1):
        # train pass
        epfe.train()
        train_loss = 0.0
        for features, labels in train_loader:
            features = features.to(epfe.device)
            labels   = labels.to(epfe.device)
            optimizer.zero_grad()
            scores = epfe.forward_train(features)
            loss = focal_loss(scores, labels, gamma=gamma, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # val pass (no gradients)
        val_loss = float("nan")
        if has_val:
            epfe.eval()
            v_total = 0.0
            with torch.no_grad():
                for features, labels in val_loader:
                    features = features.to(epfe.device)
                    labels   = labels.to(epfe.device)
                    scores = epfe.forward_train(features)
                    v_total += focal_loss(scores, labels, gamma=gamma, pos_weight=pos_weight).item()
            val_loss = v_total / len(val_loader)

        if has_val:
            print(f"Epoch {epoch:>3}/{epochs} — train: {train_loss:.4f}  val: {val_loss:.4f}")
        else:
            print(f"Epoch {epoch:>3}/{epochs} — Loss: {train_loss:.4f}")

        # save checkpoint only when the tracked metric improves, early-stop on plateau
        current_metric = val_loss if has_val else train_loss
        if current_metric < best_metric:
            best_metric = current_metric
            epochs_no_improve = 0
            torch.save(epfe.state_dict(), checkpoint_path)
            print(f"  -> best {metric_name} {best_metric:.4f}; saved to {checkpoint_path}")
        else:
            epochs_no_improve += 1
            if patience > 0 and epochs_no_improve >= patience:
                print(f"\nEarly stopping after {epoch} epochs (no {metric_name} improvement for {patience}).")
                break

    print(f"\nTraining done. Best {metric_name}: {best_metric:.4f}")
    print(f"Eval: python -m eval.eval_cached {dataset} test --weights {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["soccernet", "ego4d"])
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--gamma",      type=float, default=2.0)
    parser.add_argument("--patience",   type=int, default=0,
                        help="early stopping after N epochs without val-loss improvement (0 = off)")
    parser.add_argument("--no_mamba",   action="store_true", help="ablation: drop Mamba")
    args = parser.parse_args()
    train(args.dataset, args.epochs, args.lr, args.batch_size, args.gamma,
          patience=args.patience, use_mamba=not args.no_mamba)