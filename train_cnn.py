import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2


@dataclass(frozen=True)
class Config:
    train_csv: Path
    test_csv: Path
    train_dir: Path
    test_dir: Path
    submission_path: Path
    model_path: Path
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    val_fraction: float
    seed: int
    num_workers: int
    device: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
            
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No CUDA installation found")
    return torch.device(device)


def find_train_image(train_dir: Path, image_id: int, label: int) -> Path:
    preferred = train_dir / str(label) / f"{image_id}.png"
    if preferred.exists():
        return preferred

    matches = list(train_dir.glob(f"*/{image_id}.png"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Could not uniquely locate train image {image_id}.png")
    return matches[0]


class HindiDigitDataset(Dataset):
    def __init__(self, records: list[tuple[Path, int | None]], augment: bool) -> None:
        self.images = []
        self.labels = []
        for path, label in records:
            image = Image.open(path).convert("L")
            x = np.asarray(image, dtype=np.float32) / 255.0
            x = torch.from_numpy(x).unsqueeze(0)
            x = (x - 0.5) / 0.5
            self.images.append(x)
            self.labels.append(label)
        self.augment = augment
        self.transforms = v2.Compose([
            v2.RandomRotation(degrees=10),
            v2.RandomAffine(degrees=0, translate=(0.1,0.1),scale=(0.9,1.1)),
        ]) if augment else None

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        x = self.images[index]
        if self.transforms:
            x=self.transforms(x)
        if self.augment:
            if random.random() < 0.75:
                x = self._translate(x, max_shift=3)

        label = self.labels[index]
        if label is None: return x
        return x, torch.tensor(label, dtype=torch.long)

    @staticmethod
    def _translate(x: torch.Tensor, max_shift: int) -> torch.Tensor:
        dx = random.randint(-max_shift, max_shift)
        dy = random.randint(-max_shift, max_shift)
        if dx == 0 and dy == 0: return x

        shifted = torch.full_like(x, -1.0)
        _, height, width = x.shape
        src_y0 = max(0, -dy)
        src_y1 = min(height, height - dy)
        dst_y0 = max(0, dy)
        dst_y1 = min(height, height + dy)
        src_x0 = max(0, -dx)
        src_x1 = min(width, width - dx)
        dst_x0 = max(0, dx)
        dst_x1 = min(width, width + dx)
        shifted[:, dst_y0:dst_y1, dst_x0:dst_x1] = x[:, src_y0:src_y1, src_x0:src_x1]
        return shifted


class DigitCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(1, 24),
            self._block(24, 24),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.05),
            self._block(24, 48),
            self._block(48, 48),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),
            self._block(48, 96),
            self._block(96, 96),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
            self._block(96, 128),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(64, 10),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def stratified_split(df: pd.DataFrame, val_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    val_parts = []
    for _, group in df.groupby("Category"):
        group = group.sample(frac=1.0, random_state=seed)
        val_count = max(1, int(round(len(group) * val_fraction)))
        val_parts.append(group.iloc[:val_count])
        train_parts.append(group.iloc[val_count:])

    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, val_df


def make_records(df: pd.DataFrame, train_dir: Path) -> list[tuple[Path, int]]:
    return [
        (find_train_image(train_dir, int(row.Id), int(row.Category)), int(row.Category))
        for row in df.itertuples(index=False)
    ]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    use_autocast = device.type == "cuda"
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_autocast):
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=0.03)
            
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()
            optimizer.step()

        scheduler.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_seen += batch_size

    return total_loss / total_seen, total_correct / total_seen


def shift_batch(x: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    if dx == 0 and dy == 0: return x

    shifted = torch.full_like(x, -1.0)
    _, _, height, width = x.shape
    src_y0 = max(0, -dy)
    src_y1 = min(height, height - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(height, height + dy)
    src_x0 = max(0, -dx)
    src_x1 = min(width, width - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(width, width + dx)
    shifted[:, :, dst_y0:dst_y1, dst_x0:dst_x1] = x[:, :, src_y0:src_y1, src_x0:src_x1]
    return shifted


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_seen += batch_size

    return total_loss / total_seen, total_correct / total_seen


@torch.no_grad()
def predict_proba(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    probabilities = []

    for x in loader:
        x = x.to(device)
        variants = [
            x,
            shift_batch(x, dx=0, dy=1),
            shift_batch(x, dx=0, dy=-1),
            shift_batch(x, dx=1, dy=0),
            shift_batch(x, dx=-1, dy=0),
            shift_batch(x, dx=1, dy=1),
            shift_batch(x, dx=1, dy=-1),
            shift_batch(x, dx=-1, dy=1),
            shift_batch(x, dx=-1, dy=-1),
        ]
        logits = torch.stack([model(v) for v in variants]).mean(dim=0)
        probabilities.append(logits.softmax(dim=1).cpu().numpy())

    return np.concatenate(probabilities, axis=0)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    return predict_proba(model, loader, device).argmax(axis=1).astype(np.int64)


def load_model(model_path: Path, device: torch.device) -> DigitCNN:
    model = DigitCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, default=Path("train.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("test.csv"))
    parser.add_argument("--train-dir", type=Path, default=Path("train"))
    parser.add_argument("--test-dir", type=Path, default=Path("test"))
    parser.add_argument("--submission-path", type=Path, default=Path("submission.csv"))
    parser.add_argument("--model-path", type=Path, default=Path("best_cnn.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument(
        "--model-paths",
        type=Path,
        nargs="*",
        default=None,
        help="Optional list of checkpoints to average at prediction time.",
    )
    args = parser.parse_args()
    skip_training = args.skip_training
    model_paths = args.model_paths
    cfg_args = vars(args)
    cfg_args.pop("skip_training")
    cfg_args.pop("model_paths")
    cfg = Config(**cfg_args)

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"Using {device}")

    model = DigitCNN().to(device)
    scaler = torch.amp.GradScaler(device="cuda") if device.type == "cuda" else None    
    if skip_training:
        paths_to_load = model_paths if model_paths else [cfg.model_path]
        missing_models = [str(path) for path in paths_to_load if not path.exists()]
        if missing_models:
            raise FileNotFoundError(f"Missing model checkpoint: {missing_models[0]}")
        print(f"Skipping training and loading {len(paths_to_load)} checkpoint(s)")
    else:
        df = pd.read_csv(cfg.train_csv)
        train_df, val_df = stratified_split(df, cfg.val_fraction, cfg.seed)
        train_records = make_records(train_df, cfg.train_dir)
        val_records = make_records(val_df, cfg.train_dir)

        train_loader = DataLoader(
            HindiDigitDataset(train_records, augment=True),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
        )
        val_loader = DataLoader(
            HindiDigitDataset(val_records, augment=False),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.lr,
            epochs=cfg.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.20,
        )

        best_acc = 0.0
        best_val_loss = float("inf")
        best_epoch = 0
        for epoch in range(1, cfg.epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, device, scaler)
            val_loss, val_acc = evaluate(model, val_loader, device)
            print(
                f"epoch {epoch:02d}/{cfg.epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )
            if val_acc > best_acc or (val_acc == best_acc and val_loss < best_val_loss):
                best_acc = val_acc
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), cfg.model_path)

        print(
            f"Best checkpoint: epoch {best_epoch} "
            f"val_acc={best_acc:.4f} val_loss={best_val_loss:.4f}"
        )
        paths_to_load = model_paths if model_paths else [cfg.model_path]

    test_df = pd.read_csv(cfg.test_csv)
    test_records = [(cfg.test_dir / f"{int(row.Id)}.png", None) for row in test_df.itertuples(index=False)]
    missing = [str(path) for path, _ in test_records if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing test images, first missing path: {missing[0]}")

    test_loader = DataLoader(
        HindiDigitDataset(test_records, augment=False),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    summed_proba = None
    for path in paths_to_load:
        model = load_model(path, device)
        proba = predict_proba(model, test_loader, device)
        summed_proba = proba if summed_proba is None else summed_proba + proba

    test_df["Category"] = summed_proba.argmax(axis=1).astype(np.int64)
    test_df[["Id", "Category"]].to_csv(cfg.submission_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Written {cfg.submission_path}")


if __name__ == "__main__":
    main()
