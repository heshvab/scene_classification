import argparse
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from cnn_model import SimpleUNet


CLASS_INFO: Dict[int, Dict[str, Tuple[int, int, int]]] = {
	0: {"name": "nature", "color": (71, 158, 44)},
	1: {"name": "urban", "color": (250, 232, 112)},
	2: {"name": "water", "color": (0, 0, 245)},
}


VAL_RATIO = 0.2
SAMPLES_PER_EPOCH = 2048
EPOCHS = 25
BATCH_SIZE = 16
LR = 1e-3
SEED = 42
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


def find_mask_path(sample_dir: Path) -> Path:
	for name in ("mask.png", "mask.jpg", "mask.jpeg"):
		path = sample_dir / name
		if path.exists():
			return path
	raise FileNotFoundError(f"Missing mask file in {sample_dir}: expected mask.png or mask.jpg")


def mask_rgb_to_labels(mask_rgb: np.ndarray) -> np.ndarray:
	palette = np.array([CLASS_INFO[i]["color"] for i in sorted(CLASS_INFO)], dtype=np.float32)
	pixels = mask_rgb.reshape(-1, 3).astype(np.float32)
	d2 = ((pixels[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
	labels = np.argmin(d2, axis=1).astype(np.uint8)
	return labels.reshape(mask_rgb.shape[0], mask_rgb.shape[1])


def labels_to_rgb(labels: np.ndarray) -> np.ndarray:
	out = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
	for class_id, meta in CLASS_INFO.items():
		out[labels == class_id] = np.array(meta["color"], dtype=np.uint8)
	return out


def get_sample_dirs(root: Path) -> List[Path]:
	sample_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
	if not sample_dirs:
		raise RuntimeError(f"No sample folders found in: {root}")
	return sample_dirs


def load_sample(sample_dir: Path) -> Tuple[np.ndarray, np.ndarray, int, int]:
	paths = {
		"true": sample_dir / "true.jpg",
		"swir": sample_dir / "swir.jpg",
		"ndvi": sample_dir / "ndvi.jpg",
		"mask": find_mask_path(sample_dir),
	}
	for key, p in paths.items():
		if not p.exists():
			raise FileNotFoundError(f"Missing {key} file in {sample_dir}: {p.name}")

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


class RandomPatchDataset(Dataset):
	def __init__(self, sample_dirs: List[Path], patch_size: int, samples_per_epoch: int, augment: bool, seed: int) -> None:
		self.sample_dirs = sample_dirs
		self.patch_size = patch_size
		self.samples_per_epoch = samples_per_epoch
		self.augment = augment
		self.seed = seed
		self.rng = np.random.default_rng(seed)
		self.cache: Dict[Path, Tuple[np.ndarray, np.ndarray]] = {}

	def __len__(self) -> int:
		return self.samples_per_epoch

	def _load_sample(self, sample_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
		if sample_dir in self.cache:
			return self.cache[sample_dir]

		paths = {
			"true": sample_dir / "true.jpg",
			"swir": sample_dir / "swir.jpg",
			"ndvi": sample_dir / "ndvi.jpg",
			"mask": find_mask_path(sample_dir),
		}
		for key, p in paths.items():
			if not p.exists():
				raise FileNotFoundError(f"Missing {key} file in {sample_dir}: {p.name}")

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

		image = np.concatenate([true_rgb, swir_rgb, ndvi_rgb], axis=2)
		labels = mask_rgb_to_labels(mask_rgb)
		self.cache[sample_dir] = (image, labels)
		return image, labels

	@staticmethod
	def _apply_augment(image: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
		if rng.random() < 0.5:
			image = np.flip(image, axis=1).copy()
			labels = np.flip(labels, axis=1).copy()
		if rng.random() < 0.5:
			image = np.flip(image, axis=0).copy()
			labels = np.flip(labels, axis=0).copy()
		if rng.random() < 0.8:
			k = int(rng.integers(0, 4))
			if k > 0:
				image = np.rot90(image, k, axes=(0, 1)).copy()
				labels = np.rot90(labels, k, axes=(0, 1)).copy()
		if rng.random() < 0.7:
			brightness = float(rng.uniform(0.85, 1.15))
			contrast = float(rng.uniform(0.85, 1.15))
			imgf = image.astype(np.float32)
			mean = imgf.mean(axis=(0, 1), keepdims=True)
			imgf = (imgf - mean) * contrast + mean
			image = np.clip(imgf * brightness, 0, 255).astype(np.uint8)
		if rng.random() < 0.4:
			noise = rng.normal(0.0, 4.0, size=image.shape).astype(np.float32)
			image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
		return image, labels

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
		sample_dir = self.sample_dirs[idx % len(self.sample_dirs)]
		image, labels = self._load_sample(sample_dir)
		h, w = labels.shape[:2]

		if h < self.patch_size or w < self.patch_size:
			pad_h = max(0, self.patch_size - h)
			pad_w = max(0, self.patch_size - w)
			image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
			labels = np.pad(labels, ((0, pad_h), (0, pad_w)), mode="edge")
			h, w = labels.shape[:2]

		top = int(self.rng.integers(0, h - self.patch_size + 1))
		left = int(self.rng.integers(0, w - self.patch_size + 1))
		patch = image[top : top + self.patch_size, left : left + self.patch_size]
		patch_labels = labels[top : top + self.patch_size, left : left + self.patch_size]

		if self.augment:
			patch, patch_labels = self._apply_augment(patch, patch_labels, self.rng)

		x = torch.from_numpy(np.transpose(patch.astype(np.float32) / 255.0, (2, 0, 1)))
		y = torch.from_numpy(patch_labels.astype(np.int64))
		return x, y


def compute_class_weights(sample_dirs: List[Path]) -> torch.Tensor:
	counts = np.zeros(len(CLASS_INFO), dtype=np.float64)
	for sample_dir in sample_dirs:
		mask_rgb = load_rgb(find_mask_path(sample_dir))
		labels = mask_rgb_to_labels(mask_rgb)
		for class_id in range(len(CLASS_INFO)):
			counts[class_id] += float((labels == class_id).sum())

	total = counts.sum()
	if total == 0 or np.any(counts == 0):
		raise RuntimeError(f"Invalid class distribution in training masks: {counts.tolist()}")

	weights = total / (len(CLASS_INFO) * counts)
	logging.info("Train class counts: %s", {CLASS_INFO[i]["name"]: int(counts[i]) for i in range(len(CLASS_INFO))})
	logging.info("Computed class weights: %s", {CLASS_INFO[i]["name"]: float(weights[i]) for i in range(len(CLASS_INFO))})
	return torch.tensor(weights.astype(np.float32))


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
	preds = torch.argmax(logits, dim=1)
	correct = (preds == targets).float().sum().item()
	total = float(targets.numel())
	acc = correct / max(total, 1.0)

	ious = []
	for class_id in range(len(CLASS_INFO)):
		pred_c = preds == class_id
		target_c = targets == class_id
		inter = (pred_c & target_c).sum().item()
		union = (pred_c | target_c).sum().item()
		if union > 0:
			ious.append(inter / union)
	miou = float(np.mean(ious)) if ious else 0.0
	return {"acc": acc, "miou": miou}


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
	model.eval()
	loss_sum = 0.0
	count = 0
	metric_sum = {"acc": 0.0, "miou": 0.0}
	criterion = nn.CrossEntropyLoss()

	with torch.no_grad():
		for xb, yb in loader:
			xb = xb.to(device)
			yb = yb.to(device)
			logits = model(xb)
			loss = criterion(logits, yb)
			metrics = compute_metrics(logits, yb)
			loss_sum += loss.item()
			metric_sum["acc"] += metrics["acc"]
			metric_sum["miou"] += metrics["miou"]
			count += 1

	if count == 0:
		return {"loss": 0.0, "acc": 0.0, "miou": 0.0}
	return {"loss": loss_sum / count, "acc": metric_sum["acc"] / count, "miou": metric_sum["miou"] / count}


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Train a CNN for pixel-wise land-use classification.")
	p.add_argument("--train-data", type=Path, default=Path(__file__).parent / "../../train_data", help="Root folder with sample subfolders.")
	p.add_argument("--model-out", type=Path, default=Path(__file__).parent / "cnn_landuse_model.pt", help="Output model checkpoint.")
	return p.parse_args()


def main() -> None:
	args = parse_args()
	setup_logging(LOG_LEVEL)
	t0 = time.time()

	np.random.seed(SEED)
	random.seed(SEED)
	torch.manual_seed(SEED)

	device = resolve_device(DEVICE)
	logging.info("Using device: %s", device)

	if not args.train_data.exists() or not args.train_data.is_dir():
		raise FileNotFoundError(f"train_data folder not found: {args.train_data}")

	sample_dirs = get_sample_dirs(args.train_data)
	if len(sample_dirs) == 1:
		train_dirs, val_dirs = sample_dirs, []
	else:
		rng = np.random.default_rng(SEED)
		order = np.arange(len(sample_dirs))
		rng.shuffle(order)
		val_count = max(1, int(len(sample_dirs) * VAL_RATIO))
		train_dirs = [sample_dirs[i] for i in order[:-val_count]]
		val_dirs = [sample_dirs[i] for i in order[-val_count:]]

	logging.info("Total samples: %d | train: %d | val: %d", len(sample_dirs), len(train_dirs), len(val_dirs))
	logging.info("Train sample IDs: %s", [p.name for p in train_dirs])
	logging.info("Val sample IDs: %s", [p.name for p in val_dirs])
	logging.info("Training uses %d-channel input patches and 3-class per-pixel output", 9)

	weight_t0 = time.time()
	class_weights = compute_class_weights(train_dirs)
	logging.info("Class weight computation finished in %.2fs", time.time() - weight_t0)

	train_ds = RandomPatchDataset(train_dirs, patch_size=256, samples_per_epoch=SAMPLES_PER_EPOCH, augment=True, seed=SEED)
	train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

	val_ds = None
	val_loader = None
	if val_dirs:
		val_ds = RandomPatchDataset(val_dirs, patch_size=256, samples_per_epoch=max(64, len(val_dirs) * 16), augment=False, seed=SEED + 1)
		val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

	model = SimpleUNet(in_channels=9, out_channels=3, base_channels=32).to(device)
	criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
	optimizer = torch.optim.Adam(model.parameters(), lr=LR)

	best_val_loss = float("inf")
	best_state = None

	for epoch in range(1, EPOCHS + 1):
		epoch_t0 = time.time()
		model.train()
		running_loss = 0.0
		running_acc = 0.0
		running_miou = 0.0
		batches = 0

		logging.info("Epoch %d/%d start", epoch, EPOCHS)
		for step, (xb, yb) in enumerate(train_loader, start=1):
			xb = xb.to(device)
			yb = yb.to(device)

			optimizer.zero_grad(set_to_none=True)
			logits = model(xb)
			loss = criterion(logits, yb)
			loss.backward()
			optimizer.step()

			metrics = compute_metrics(logits.detach(), yb)
			running_loss += loss.item()
			running_acc += metrics["acc"]
			running_miou += metrics["miou"]
			batches += 1

			if step % max(1, SAMPLES_PER_EPOCH // (BATCH_SIZE * 10)) == 0 or step == len(train_loader):
				elapsed = time.time() - epoch_t0
				logging.info(
					"Epoch %d step %d/%d | loss=%.4f | acc=%.4f | miou=%.4f | speed=%.1f batches/s",
					epoch,
					step,
					len(train_loader),
					running_loss / batches,
					running_acc / batches,
					running_miou / batches,
					batches / max(elapsed, 1e-6),
				)

		train_loss = running_loss / max(batches, 1)
		train_acc = running_acc / max(batches, 1)
		train_miou = running_miou / max(batches, 1)

		if val_loader is not None:
			val_metrics = evaluate_model(model, val_loader, device)
		else:
			val_metrics = {"loss": 0.0, "acc": 0.0, "miou": 0.0}

		logging.info(
			"Epoch %d done in %.1fs | train_loss=%.4f train_acc=%.4f train_miou=%.4f | val_loss=%.4f val_acc=%.4f val_miou=%.4f",
			epoch,
			time.time() - epoch_t0,
			train_loss,
			train_acc,
			train_miou,
			val_metrics["loss"],
			val_metrics["acc"],
			val_metrics["miou"],
		)

		if val_loader is not None and val_metrics["loss"] < best_val_loss:
			best_val_loss = val_metrics["loss"]
			best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
			logging.info("New best validation loss: %.4f", best_val_loss)

	if best_state is not None:
		model.load_state_dict(best_state)

	args.model_out.parent.mkdir(parents=True, exist_ok=True)
	torch.save(
		{
			"model_state_dict": model.state_dict(),
			"class_info": CLASS_INFO,
			"patch_size": args.patch_size,
			"in_channels": 9,
			"out_classes": 3,
		},
		args.model_out,
	)
	logging.info("Saved model to: %s", args.model_out)

	logging.info("Training finished in %.2fs", time.time() - t0)


if __name__ == "__main__":
	main()

