import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from cnn_model import SimpleUNet


DEFAULT_CLASS_INFO: Dict[int, Dict[str, Tuple[int, int, int]]] = {
    0: {"name": "nature", "color": (71, 158, 44)},
    1: {"name": "urban", "color": (250, 232, 112)},
    2: {"name": "water", "color": (0, 0, 245)},
}


INFERENCE_BATCH_SIZE = 16
DEVICE = "auto"
LOG_LEVEL = "INFO"


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


def labels_to_rgb(labels: np.ndarray, class_info: Dict[int, Dict[str, Tuple[int, int, int]]]) -> np.ndarray:
    out = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    for class_id, meta in class_info.items():
        out[labels == int(class_id)] = np.array(meta["color"], dtype=np.uint8)
    return out


def build_feature_tensor(images: Dict[str, np.ndarray]) -> np.ndarray:
    stacked = np.concatenate([images["true"], images["swir"], images["ndvi"]], axis=2)
    return stacked.astype(np.float32) / 255.0


def pad_image(image: np.ndarray, patch_size: int, stride: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = image.shape[:2]
    target_h = max(h, patch_size)
    target_w = max(w, patch_size)

    if stride > 0:
        rem_h = (target_h - patch_size) % stride
        rem_w = (target_w - patch_size) % stride
        if rem_h != 0:
            target_h += stride - rem_h
        if rem_w != 0:
            target_w += stride - rem_w

    pad_h = target_h - h
    pad_w = target_w - w
    if pad_h == 0 and pad_w == 0:
        return image, (0, 0)

    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    return padded, (pad_h, pad_w)


def sliding_window_predict(
    model: SimpleUNet,
    image: np.ndarray,
    patch_size: int,
    stride: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    h, w, c = image.shape
    positions_y = list(range(0, h - patch_size + 1, stride))
    positions_x = list(range(0, w - patch_size + 1, stride))
    if positions_y[-1] != h - patch_size:
        positions_y.append(h - patch_size)
    if positions_x[-1] != w - patch_size:
        positions_x.append(w - patch_size)

    prob_sum = np.zeros((3, h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)

    coords: List[Tuple[int, int]] = []
    patches: List[np.ndarray] = []

    def flush_batch() -> None:
        nonlocal coords, patches, prob_sum, weight_sum
        if not patches:
            return
        xb = np.stack(patches, axis=0)
        xb_t = torch.from_numpy(np.transpose(xb, (0, 3, 1, 2))).to(device)
        with torch.no_grad():
            logits = model(xb_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        for (top, left), prob in zip(coords, probs):
            prob_sum[:, top : top + patch_size, left : left + patch_size] += prob
            weight_sum[top : top + patch_size, left : left + patch_size] += 1.0
        coords = []
        patches = []

    for top in positions_y:
        for left in positions_x:
            coords.append((top, left))
            patches.append(image[top : top + patch_size, left : left + patch_size])
            if len(patches) >= batch_size:
                flush_batch()
    flush_batch()

    prob_sum /= np.maximum(weight_sum[None, :, :], 1e-6)
    pred = np.argmax(prob_sum, axis=0).astype(np.uint8)
    return pred


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict land-use classes on an unseen image using CNN segmentation.")
    p.add_argument("--model", type=Path, default=Path(__file__).parent / "cnn_landuse_model.pt", help="Trained model checkpoint.")
    p.add_argument("--true-image", type=Path, required=True, help="True-color image.")
    p.add_argument("--swir-image", type=Path, required=True, help="SWIR image.")
    p.add_argument("--ndvi-image", type=Path, required=True, help="NDVI image.")
    p.add_argument("--output", type=Path, required=True, help="Output color mask image.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(LOG_LEVEL)

    device = resolve_device(DEVICE)
    logging.info("Using device: %s", device)

    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")

    checkpoint = torch.load(args.model, map_location=device)
    class_info = checkpoint.get("class_info", DEFAULT_CLASS_INFO)
    patch_size = int(checkpoint.get("patch_size", 256))
    stride = int(max(1, patch_size // 2))
    in_channels = int(checkpoint.get("in_channels", 9))
    out_classes = int(checkpoint.get("out_classes", 3))

    model = SimpleUNet(in_channels=in_channels, out_channels=out_classes, base_channels=32).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    images = {
        "true": load_rgb(args.true_image),
        "swir": load_rgb(args.swir_image),
        "ndvi": load_rgb(args.ndvi_image),
    }

    h, w = ensure_same_size(images)

    image = build_feature_tensor(images)
    padded, (pad_h, pad_w) = pad_image(image, patch_size, stride)
    logging.info("Input size: %dx%d | padded to: %dx%d | patch_size=%d | stride=%d", w, h, padded.shape[1], padded.shape[0], patch_size, stride)

    pred = sliding_window_predict(model, padded, patch_size, stride, device, INFERENCE_BATCH_SIZE)
    if pad_h > 0:
        pred = pred[:-pad_h, :]
    if pad_w > 0:
        pred = pred[:, :-pad_w]

    total = int(pred.size)
    parts = []
    for class_id in sorted(class_info):
        count = int((pred == int(class_id)).sum())
        pct = 100.0 * count / max(total, 1)
        parts.append(f"{class_info[class_id]['name']}={count} ({pct:.2f}%)")
    logging.info("Predicted class distribution: %s", " | ".join(parts))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(labels_to_rgb(pred, class_info)).save(args.output)
    logging.info("Saved color mask to: %s", args.output)


if __name__ == "__main__":
    main()
