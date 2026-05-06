import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


CLASS_INFO: Dict[int, Dict[str, Tuple[int, int, int]]] = {
    0: {"name": "nature", "color": (71, 158, 44)},
    1: {"name": "urban", "color": (250, 232, 112)},
    2: {"name": "water", "color": (0, 0, 245)},
}


BATCH_SIZE = 65536
DEVICE = "auto"
LOG_LEVEL = "INFO"


class PixelMLP(nn.Module):
    def __init__(self, in_features: int = 9, hidden_sizes: Tuple[int, int, int] = (128, 64, 32), out_classes: int = 3) -> None:
        super().__init__()
        h1, h2, h3 = hidden_sizes
        self.net = nn.Sequential(
            nn.Linear(in_features, h1),
            nn.ReLU(),
            nn.BatchNorm1d(h1),
            nn.Dropout(0.0),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.BatchNorm1d(h2),
            nn.Dropout(0.0),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, out_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_rgb(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def ensure_same_size(images: Dict[str, np.ndarray]) -> Tuple[int, int]:
    keys = list(images.keys())
    h0, w0 = images[keys[0]].shape[:2]
    for k in keys[1:]:
        h, w = images[k].shape[:2]
        if (h, w) != (h0, w0):
            raise ValueError(f"Image size mismatch: {keys[0]} is {h0}x{w0}, but {k} is {h}x{w}")
    return h0, w0


def build_feature_matrix(images: Dict[str, np.ndarray], feature_order) -> np.ndarray:
    channel_map = {"true": images["true"], "swir": images["swir"], "ndvi": images["ndvi"]}
    per_channel = {
        "true_r": channel_map["true"][:, :, 0],
        "true_g": channel_map["true"][:, :, 1],
        "true_b": channel_map["true"][:, :, 2],
        "swir_r": channel_map["swir"][:, :, 0],
        "swir_g": channel_map["swir"][:, :, 1],
        "swir_b": channel_map["swir"][:, :, 2],
        "ndvi_r": channel_map["ndvi"][:, :, 0],
        "ndvi_g": channel_map["ndvi"][:, :, 1],
        "ndvi_b": channel_map["ndvi"][:, :, 2],
    }
    missing = [f for f in feature_order if f not in per_channel]
    if missing:
        raise RuntimeError(f"Model expects features not provided by this script: {missing}")
    stacked = np.stack([per_channel[f] for f in feature_order], axis=2)
    return stacked.reshape(-1, len(feature_order)).astype(np.float32)


def labels_to_color_mask(labels_2d: np.ndarray, class_info: Dict[int, Dict[str, Tuple[int, int, int]]]) -> np.ndarray:
    out = np.zeros((labels_2d.shape[0], labels_2d.shape[1], 3), dtype=np.uint8)
    for class_id, meta in class_info.items():
        out[labels_2d == int(class_id)] = np.array(meta["color"], dtype=np.uint8)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run pixel-wise inference with trained MLP model.")
    p.add_argument("--model", type=Path, default=Path(__file__).parent / "nn_landuse_model.pt", help="Path to trained model file.")
    p.add_argument("--true-image", type=Path, required=True)
    p.add_argument("--swir-image", type=Path, required=True)
    p.add_argument("--ndvi-image", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="Output colorized mask PNG/JPG path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(LOG_LEVEL)

    device = resolve_device(DEVICE)
    logging.info("Using device: %s", device)

    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")

    try:
        checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.model, map_location=device)
    class_info = checkpoint.get("class_info", CLASS_INFO)
    feature_order = checkpoint.get("feature_order")
    if feature_order is None:
        raise RuntimeError("Model file has no feature_order; retrain using current train_model.py")

    raw_mean = checkpoint.get("mean")
    raw_std = checkpoint.get("std")
    if raw_mean is None or raw_std is None:
        raise RuntimeError("Model file has no mean/std normalization stats; retrain using current train_model.py")

    mean = np.asarray(raw_mean, dtype=np.float32)
    std = np.asarray(raw_std, dtype=np.float32)

    hidden_sizes = tuple(checkpoint.get("hidden_sizes", (128, 64, 32)))
    model = PixelMLP(in_features=len(feature_order), hidden_sizes=hidden_sizes, out_classes=len(class_info)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    images = {
        "true": load_rgb(args.true_image),
        "swir": load_rgb(args.swir_image),
        "ndvi": load_rgb(args.ndvi_image),
    }
    h, w = ensure_same_size(images)
    logging.info("Input size: %dx%d", w, h)

    x = build_feature_matrix(images, feature_order)
    x = (x - mean) / std
    logging.info("Feature matrix shape: %s", x.shape)

    preds = []
    with torch.no_grad():
        for start in range(0, x.shape[0], BATCH_SIZE):
            xb = torch.from_numpy(x[start : start + BATCH_SIZE].astype(np.float32)).to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.append(pred)

    y_pred = np.concatenate(preds).astype(np.uint8)
    pred_2d = y_pred.reshape(h, w)

    total = int(pred_2d.size)
    parts = []
    for class_id in sorted(class_info):
        count = int((pred_2d == int(class_id)).sum())
        pct = 100.0 * count / max(total, 1)
        name = class_info[class_id].get("name", str(class_id))
        parts.append(f"{name}={count} ({pct:.2f}%)")
    logging.info("Predicted class distribution: %s", " | ".join(parts))

    color_mask = labels_to_color_mask(pred_2d, class_info)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(color_mask).save(args.output)
    logging.info("Saved color mask: %s", args.output)


if __name__ == "__main__":
    main()
