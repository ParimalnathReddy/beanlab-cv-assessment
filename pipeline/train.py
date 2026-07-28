"""
train.py — ConvNeXt-Tiny training loop for per-bean quality classification.

Labels: 0 = Good, 1 = Bad
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS  = 60
LR_HEAD = 1e-3
LR_BACK = 2e-5

# Fixed decision threshold applied to the Bad-class softmax output.
# Set once after initial experiments and not revisited.
BAD_THRESHOLD = 0.5


def build_model(num_classes: int = 2) -> nn.Module:
    """Load pretrained ConvNeXt-Tiny and replace the classification head."""
    model       = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model.to(DEVICE)


def run_epoch(model, loader: DataLoader, optimizer, criterion, training: bool):
    model.train() if training else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            if training:
                optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader), acc, f1


def train(model: nn.Module,
          train_loader: DataLoader,
          val_loader: DataLoader) -> float:
    """
    Full training loop. Returns the best validation accuracy achieved.
    Saves the best checkpoint to checkpoints/best_bean_model.pth.
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW([
        {"params": model.classifier.parameters(), "lr": LR_HEAD},
        {"params": [p for n, p in model.named_parameters()
                    if "classifier" not in n], "lr": LR_BACK},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, _ = run_epoch(
            model, train_loader, optimizer, criterion, training=True
        )
        vl_loss, vl_acc, vl_f1 = run_epoch(
            model, val_loader, None, criterion, training=False
        )
        scheduler.step()

        print(f"[{epoch:02d}/{EPOCHS}]  "
              f"train  loss={tr_loss:.4f}  acc={tr_acc:.4f}  |  "
              f"val    loss={vl_loss:.4f}  acc={vl_acc:.4f}  f1={vl_f1:.4f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), "checkpoints/best_bean_model.pth")
            print(f"  saved  (val_acc={best_val_acc:.4f})")

    return best_val_acc
