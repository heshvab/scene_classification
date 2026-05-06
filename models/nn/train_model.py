import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset


CLASS_INFO: Dict[int, Dict[str, Tuple[int, int, int]]] = {
	0: {"name": "nature", "color": (71, 158, 44)},
	1: {"name": "urban", "color": (250, 232, 112)},
	2: {"name": "water", "color": (0, 0, 245)},
}


FEATURE_ORDER = [
	"true_r",
	"true_g",
	"true_b",
	"swir_r",
	"swir_g",
	"swir_b",
	"ndvi_r",
	"ndvi_g",
	"ndvi_b",
]


VAL_RATIO = 0.2
TRAIN_PIXEL_FRACTION = 0.3
EPOCHS = 25
BATCH_SIZE = 4096
LR = 1e-3
WEIGHT_DECAY = 1e-5
SEED = 42
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
			nn.Dropout(0.2),
			nn.Linear(h1, h2),
			nn.ReLU(),
			nn.BatchNorm1d(h2),
			nn.Dropout(0.2),
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


def mask_rgb_to_labels(mask_rgb: np.ndarray) -> np.ndarray:
	palette = np.array([CLASS_INFO[i]["color"] for i in sorted(CLASS_INFO)], dtype=np.float32)
	pixels = mask_rgb.reshape(-1, 3).astype(np.float32)
	d2 = ((pixels[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
	labels = np.argmin(d2, axis=1).astype(np.int64)
	return labels.reshape(mask_rgb.shape[0], mask_rgb.shape[1])


def labels_to_rgb(labels: np.ndarray) -> np.ndarray:
	out = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
	for class_id, meta in CLASS_INFO.items():
		out[labels == class_id] = np.array(meta["color"], dtype=np.uint8)
	return out


def summarize_labels(y: np.ndarray, prefix: str) -> None:
	total = int(y.size)
	parts = []
	for class_id in sorted(CLASS_INFO):
		name = CLASS_INFO[class_id]["name"]
		count = int((y == class_id).sum())
		pct = 100.0 * count / max(total, 1)
		parts.append(f"{name}={count} ({pct:.2f}%)")
	logging.info("%s class distribution: %s", prefix, " | ".join(parts))


def get_sample_dirs(data_root: Path) -> List[Path]:
	sample_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
	if not sample_dirs:
		raise RuntimeError(f"No sample folders found in: {data_root}")
	return sample_dirs


def find_data_root(requested: Path) -> Path:
	if requested.exists():
		return requested
	if requested.name == "train_data":
		fallback = requested.parent / "test_data"
		if fallback.exists():
			logging.warning("train_data not found, using fallback: %s", fallback)
			return fallback
	raise FileNotFoundError(f"Dataset folder not found: {requested}")


def load_sample(sample_dir: Path) -> Tuple[np.ndarray, np.ndarray, int, int]:
	paths = {
		"true": sample_dir / "true.jpg",
		"swir": sample_dir / "swir.jpg",
		"ndvi": sample_dir / "ndvi.jpg",
		"mask": sample_dir / "mask.jpg",
	}
	for key, p in paths.items():
		if not p.exists():
			raise FileNotFoundError(f"Missing {key} in {sample_dir}: {p.name}")

	true_rgb = load_rgb(paths["true"])
	swir_rgb = load_rgb(paths["swir"])
	ndvi_rgb = load_rgb(paths["ndvi"])
	mask_rgb = load_rgb(paths["mask"])

	h, w = true_rgb.shape[:2]
	if not (
		swir_rgb.shape[:2] == (h, w)
		and ndvi_rgb.shape[:2] == (h, w)
		and mask_rgb.shape[:2] == (h, w)
	):
		raise ValueError(f"Image size mismatch in sample {sample_dir}")

	features = np.concatenate([true_rgb, swir_rgb, ndvi_rgb], axis=2)
	labels = mask_rgb_to_labels(mask_rgb)
	return features, labels, h, w


def flatten_pixels(features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	x = features.reshape(-1, features.shape[2]).astype(np.float32)
	y = labels.reshape(-1).astype(np.int64)
	return x, y


def split_samples(sample_dirs: List[Path], val_ratio: float, seed: int) -> Tuple[List[Path], List[Path]]:
	if len(sample_dirs) == 1:
		return sample_dirs, []

	rng = np.random.default_rng(seed)
	idx = np.arange(len(sample_dirs))
	rng.shuffle(idx)

	val_count = max(1, int(len(sample_dirs) * val_ratio))
	train_count = len(sample_dirs) - val_count
	if train_count <= 0:
		raise RuntimeError("Not enough samples for training. Reduce --val-ratio")

	train_dirs = [sample_dirs[i] for i in idx[:train_count]]
	val_dirs = [sample_dirs[i] for i in idx[train_count:]]
	return train_dirs, val_dirs


def collect_dataset(sample_dirs: List[Path], train_pixel_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
	rng = np.random.default_rng(seed)
	xs: List[np.ndarray] = []
	ys: List[np.ndarray] = []

	for sample_dir in sample_dirs:
		features, labels, h, w = load_sample(sample_dir)
		x, y = flatten_pixels(features, labels)

		sample_count = x.shape[0]
		if 0 < train_pixel_fraction < 1:
			sample_count = max(1, int(np.ceil(x.shape[0] * train_pixel_fraction)))
		if sample_count < x.shape[0]:
			idx = rng.choice(x.shape[0], size=sample_count, replace=False)
			x = x[idx]
			y = y[idx]
			logging.info("Sample %s: using %d/%d pixels (%.1f%%)", sample_dir.name, x.shape[0], h * w, 100.0 * x.shape[0] / max(h * w, 1))
		else:
			logging.info("Sample %s: using %d/%d pixels (100.0%%)", sample_dir.name, h * w, h * w)

		xs.append(x)
		ys.append(y)

	return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def evaluate_model(model: nn.Module, x: np.ndarray, y: np.ndarray, batch_size: int, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
	model.eval()
	preds = []

	with torch.no_grad():
		for start in range(0, x.shape[0], batch_size):
			xb = torch.from_numpy(x[start : start + batch_size]).to(device)
			logits = model(xb)
			pred = torch.argmax(logits, dim=1).cpu().numpy()
			preds.append(pred)

	y_pred = np.concatenate(preds)
	return y, y_pred


def evaluate_on_samples(model: nn.Module, sample_dirs: List[Path], batch_size: int, device: torch.device, mean: np.ndarray, std: np.ndarray) -> None:
	if not sample_dirs:
		logging.warning("No validation samples available; skipping validation stage")
		return

	start = time.time()
	y_true_all: List[np.ndarray] = []
	y_pred_all: List[np.ndarray] = []

	for sample_dir in sample_dirs:
		features, labels, h, w = load_sample(sample_dir)
		x, y_true = flatten_pixels(features, labels)
		x = (x - mean) / std

		_, y_pred = evaluate_model(model, x, y_true, batch_size, device)
		y_true_all.append(y_true)
		y_pred_all.append(y_pred)

		acc = accuracy_score(y_true, y_pred)
		logging.info("Val sample %s: pixel_acc=%.4f", sample_dir.name, acc)

	y_true = np.concatenate(y_true_all)
	y_pred = np.concatenate(y_pred_all)

	logging.info("Validation pixel accuracy: %.4f", accuracy_score(y_true, y_pred))
	summarize_labels(y_true, "Validation true")
	summarize_labels(y_pred, "Validation pred")

	report = classification_report(
		y_true,
		y_pred,
		labels=[0, 1, 2],
		target_names=[CLASS_INFO[i]["name"] for i in sorted(CLASS_INFO)],
		digits=4,
	)
	cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
	logging.info("Classification report:\n%s", report)
	logging.info("Confusion matrix (rows=true, cols=pred):\n%s", cm)
	logging.info("Validation stage finished in %.2fs", time.time() - start)


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Train a simple pixel-wise neural network (MLP) for land-use classification.")
	p.add_argument("--train-data", type=Path, default=Path(__file__).parent / "../../train_data", help="Root folder with sample subfolders.")
	p.add_argument("--model-out", type=Path, default=Path(__file__).parent / "nn_landuse_model.pt", help="Output model path.")
	return p.parse_args()


def main() -> None:
	args = parse_args()
	setup_logging(LOG_LEVEL)
	t0 = time.time()

	np.random.seed(SEED)
	torch.manual_seed(SEED)

	data_root = find_data_root(args.train_data)
	sample_dirs = get_sample_dirs(data_root)
	train_dirs, val_dirs = split_samples(sample_dirs, VAL_RATIO, SEED)

	logging.info("Starting pixel-MLP training pipeline")
	logging.info(
		"Config: data_root=%s epochs=%d batch_size=%d lr=%.6f val_ratio=%.2f train_pixel_fraction=%.3f seed=%d",
		data_root,
		EPOCHS,
		BATCH_SIZE,
		LR,
		VAL_RATIO,
		TRAIN_PIXEL_FRACTION,
		SEED,
	)

	logging.info("Total samples: %d | train: %d | val: %d", len(sample_dirs), len(train_dirs), len(val_dirs))
	logging.info("Train sample IDs: %s", [p.name for p in train_dirs])
	logging.info("Val sample IDs: %s", [p.name for p in val_dirs])

	data_t0 = time.time()
	x_train, y_train = collect_dataset(train_dirs, TRAIN_PIXEL_FRACTION, SEED)
	logging.info("Dataset collection finished in %.2fs", time.time() - data_t0)

	mean = x_train.mean(axis=0, keepdims=True)
	std = x_train.std(axis=0, keepdims=True) + 1e-6
	x_train = (x_train - mean) / std

	logging.info("Train matrix shape: X=%s y=%s", x_train.shape, y_train.shape)
	logging.info(
		"Feature stats (normalized): min=%.3f max=%.3f mean=%.3f std=%.3f",
		float(x_train.min()),
		float(x_train.max()),
		float(x_train.mean()),
		float(x_train.std()),
	)
	summarize_labels(y_train, "Train")

	class_counts = np.array([(y_train == i).sum() for i in sorted(CLASS_INFO)], dtype=np.float64)
	class_weights = class_counts.sum() / (len(class_counts) * np.maximum(class_counts, 1.0))

	device = resolve_device(DEVICE)
	logging.info("Using device: %s", device)

	model = PixelMLP(in_features=len(FEATURE_ORDER), out_classes=len(CLASS_INFO)).to(device)
	criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
	optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

	ds = TensorDataset(torch.from_numpy(x_train.astype(np.float32)), torch.from_numpy(y_train.astype(np.int64)))
	train_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

	best_loss = float("inf")
	best_state = None

	fit_t0 = time.time()
	for epoch in range(1, EPOCHS + 1):
		model.train()
		running_loss = 0.0
		seen = 0
		e0 = time.time()

		for xb, yb in train_loader:
			xb = xb.to(device)
			yb = yb.to(device)

			optimizer.zero_grad(set_to_none=True)
			logits = model(xb)
			loss = criterion(logits, yb)
			loss.backward()
			optimizer.step()

			batch_size = xb.size(0)
			running_loss += loss.item() * batch_size
			seen += batch_size

		epoch_loss = running_loss / max(1, seen)
		logging.info("Epoch %d/%d | train_loss=%.6f | time=%.2fs", epoch, EPOCHS, epoch_loss, time.time() - e0)

		if epoch_loss < best_loss:
			best_loss = epoch_loss
			best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

	logging.info("Training finished in %.2fs | best_train_loss=%.6f", time.time() - fit_t0, best_loss)

	if best_state is not None:
		model.load_state_dict(best_state)

	args.model_out.parent.mkdir(parents=True, exist_ok=True)
	torch.save(
		{
			"model_state_dict": model.state_dict(),
			"class_info": CLASS_INFO,
			"feature_order": FEATURE_ORDER,
			"mean": mean.astype(np.float32).tolist(),
			"std": std.astype(np.float32).tolist(),
			"hidden_sizes": (128, 64, 32),
		},
		args.model_out,
	)
	logging.info("Saved model to: %s", args.model_out)

	evaluate_on_samples(
		model=model,
		sample_dirs=val_dirs,
		batch_size=BATCH_SIZE,
		device=device,
		mean=mean,
		std=std,
	)
	logging.info("Pipeline finished in %.2fs", time.time() - t0)


if __name__ == "__main__":
	main()

