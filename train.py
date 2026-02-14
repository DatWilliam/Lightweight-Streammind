import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from config import cfg
from model.mamba import MambaModel
from data.dataset import (
    EventDataset,
    extract_features,
    get_training_video_ids,
    get_validation_video_ids,
)


class MambaEventDetector(nn.Module):
    """Lightweight wrapper for training (no CLIP, operates on pre-extracted features)."""

    def __init__(self):
        super().__init__()
        self.mamba = MambaModel(
            d_model=cfg.d_model,
            n_layers=cfg.n_mamba_layers,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand=cfg.mamba_expand,
        )
        self.event_head = nn.Linear(cfg.d_model, 1)

    def forward(self, features):
        """
        Args:
            features: (B, L, d_model)
        Returns:
            event_logits: (B, L) - raw logits (apply sigmoid for probabilities)
            perception_tokens: (B, L, d_model)
        """
        perception_tokens, _ = self.mamba(features)
        event_logits = self.event_head(perception_tokens).squeeze(-1)
        return event_logits, perception_tokens


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Step 1: Pre-extract CLIP features ---
    print("\n=== Step 1: Feature extraction ===")
    train_ids = get_training_video_ids()
    val_ids = get_validation_video_ids()

    # Filter to videos that actually exist on disk
    from data.prepare_epickitchen import get_frame_count
    available_train = []
    for vid in train_ids:
        try:
            get_frame_count(vid)
            available_train.append(vid)
        except Exception:
            pass

    available_val = []
    for vid in val_ids:
        try:
            get_frame_count(vid)
            available_val.append(vid)
        except Exception:
            pass

    print(f"Available videos - train: {len(available_train)}, val: {len(available_val)}")

    if not available_train:
        print("ERROR: No training videos found. Check cfg.epickitchen_video_root")
        return

    extract_features(available_train)
    if available_val:
        extract_features(available_val)

    # --- Step 2: Create datasets ---
    print("\n=== Step 2: Dataset creation ===")
    train_dataset = EventDataset(available_train)
    pos_weight = torch.tensor([train_dataset.pos_weight], device=device)

    if not len(train_dataset):
        print("ERROR: No training windows created. Check seq_len vs video lengths.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = None
    if available_val:
        val_dataset = EventDataset(available_val)
        if len(val_dataset):
            val_loader = DataLoader(
                val_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

    # --- Step 3: Model, optimizer, loss ---
    print("\n=== Step 3: Training ===")
    model = MambaEventDetector().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Learnable parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Checkpoint dir
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    best_val_loss = float("inf")

    # --- Step 4: Training loop ---
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_tp, total_fp, total_fn = 0, 0, 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            event_logits, _ = model(features)
            loss = criterion(event_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * features.size(0)

            # Metrics
            preds = (event_logits.detach() > 0.0).float()
            total_tp += ((preds == 1) & (labels == 1)).sum().item()
            total_fp += ((preds == 1) & (labels == 0)).sum().item()
            total_fn += ((preds == 0) & (labels == 1)).sum().item()

        scheduler.step()

        avg_loss = total_loss / len(train_dataset)
        precision = total_tp / max(total_tp + total_fp, 1)
        recall = total_tp / max(total_tp + total_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        print(f"Epoch {epoch}/{cfg.epochs} | loss: {avg_loss:.4f} | "
              f"P: {precision:.3f} R: {recall:.3f} F1: {f1:.3f} | "
              f"lr: {scheduler.get_last_lr()[0]:.2e}")

        # --- Validation ---
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_tp, val_fp, val_fn = 0, 0, 0

            with torch.no_grad():
                for features, labels in val_loader:
                    features = features.to(device)
                    labels = labels.to(device)

                    event_logits, _ = model(features)
                    loss = criterion(event_logits, labels)
                    val_loss += loss.item() * features.size(0)

                    preds = (event_logits > 0.0).float()
                    val_tp += ((preds == 1) & (labels == 1)).sum().item()
                    val_fp += ((preds == 1) & (labels == 0)).sum().item()
                    val_fn += ((preds == 0) & (labels == 1)).sum().item()

            avg_val_loss = val_loss / len(val_loader.dataset)
            val_prec = val_tp / max(val_tp + val_fp, 1)
            val_rec = val_tp / max(val_tp + val_fn, 1)
            val_f1 = 2 * val_prec * val_rec / max(val_prec + val_rec, 1e-8)

            print(f"  val_loss: {avg_val_loss:.4f} | "
                  f"P: {val_prec:.3f} R: {val_rec:.3f} F1: {val_f1:.3f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'mamba': model.mamba.state_dict(),
                    'event_head': model.event_head.state_dict(),
                    'epoch': epoch,
                    'val_loss': avg_val_loss,
                }, ckpt_dir / "best.pt")
                print(f"  Saved best checkpoint (val_loss={avg_val_loss:.4f})")

        # Save latest checkpoint every epoch
        torch.save({
            'mamba': model.mamba.state_dict(),
            'event_head': model.event_head.state_dict(),
            'epoch': epoch,
        }, ckpt_dir / "latest.pt")

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {ckpt_dir}/")


if __name__ == "__main__":
    train()