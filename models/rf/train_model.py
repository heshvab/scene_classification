import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


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
N_ESTIMATORS = 300
MAX_DEPTH = None
MIN_SAMPLES_LEAF = 1
SEED = 42
LOG_LEVEL = "INFO"


def setup_logging(level: str) -> None:
	logging.basicConfig(
		level=getattr(logging, level.upper(), logging.INFO),
		format="%(asctime)s | %(levelname)s | %(message)s",
		datefmt="%H:%M:%S",
	)


def load_rgb(path: Path) -> np.ndarray:
	if not path.exists():
		raise FileNotFoundError(path)
	return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


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


def get_sample_dirs(train_data: Path) -> List[Path]:
	sample_dirs = sorted([p for p in train_data.iterdir() if p.is_dir()])
	if not sample_dirs:
		raise RuntimeError(f"No sample folders found in: {train_data}")
	return sample_dirs


def load_sample(sample_dir: Path) -> Tuple[np.ndarray, np.ndarray, int, int]:
	paths = {
		"true": sample_dir / "true.jpg",
		"swir": sample_dir / "swir.jpg",
		"ndvi": sample_dir / "ndvi.jpg",
		"mask": sample_dir / "mask.jpg",
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


def flatten_pixels(features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	x = features.reshape(-1, features.shape[2]).astype(np.float32)
	y = labels.reshape(-1).astype(np.uint8)
	return x, y


def summarize_labels(y: np.ndarray, prefix: str) -> None:
	total = int(y.size)
	parts = []
	for class_id in sorted(CLASS_INFO):
		name = CLASS_INFO[class_id]["name"]
		count = int((y == class_id).sum())
		pct = 100.0 * count / max(total, 1)
		parts.append(f"{name}={count} ({pct:.2f}%)")
	logging.info("%s class distribution: %s", prefix, " | ".join(parts))


def split_samples(sample_dirs: List[Path], val_ratio: float, seed: int) -> Tuple[List[Path], List[Path]]:
	rng = np.random.default_rng(seed)
	indices = np.arange(len(sample_dirs))
	rng.shuffle(indices)

	val_count = max(1, int(len(sample_dirs) * val_ratio))
	train_count = len(sample_dirs) - val_count
	if train_count <= 0:
		raise RuntimeError("Not enough samples for training. Reduce --val-ratio.")

	train_dirs = [sample_dirs[i] for i in indices[:train_count]]
	val_dirs = [sample_dirs[i] for i in indices[train_count:]]
	return train_dirs, val_dirs


def collect_dataset(sample_dirs: List[Path], train_pixel_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
	x_all: List[np.ndarray] = []
	y_all: List[np.ndarray] = []
	rng = np.random.default_rng(seed)

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
			logging.info(
				"Sample %s: using %d/%d pixels (%.1f%%)",
				sample_dir.name,
				x.shape[0],
				h * w,
				100.0 * x.shape[0] / max(h * w, 1),
			)
		else:
			logging.info("Sample %s: using %d/%d pixels (100.0%%)", sample_dir.name, h * w, h * w)

		x_all.append(x)
		y_all.append(y)

	x_cat = np.concatenate(x_all, axis=0)
	y_cat = np.concatenate(y_all, axis=0)
	return x_cat, y_cat


def evaluate_on_samples(
	model: RandomForestClassifier,
	sample_dirs: List[Path],
) -> None:
	start = time.time()
	y_true_all: List[np.ndarray] = []
	y_pred_all: List[np.ndarray] = []

	for sample_dir in sample_dirs:
		features, labels, h, w = load_sample(sample_dir)
		x, y_true = flatten_pixels(features, labels)
		y_pred = model.predict(x).astype(np.uint8)

		y_true_all.append(y_true)
		y_pred_all.append(y_pred)

		acc = accuracy_score(y_true, y_pred)
		logging.info("Val sample %s: pixel_acc=%.4f", sample_dir.name, acc)

	y_true = np.concatenate(y_true_all)
	y_pred = np.concatenate(y_pred_all)

	overall_acc = accuracy_score(y_true, y_pred)
	logging.info("Validation pixel accuracy: %.4f", overall_acc)
	summarize_labels(y_true, "Validation true")
	summarize_labels(y_pred, "Validation pred")

	target_names = [CLASS_INFO[i]["name"] for i in sorted(CLASS_INFO)]
	report = classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=target_names, digits=4)
	cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

	logging.info("Classification report:\n%s", report)
	logging.info("Confusion matrix (rows=true, cols=pred):\n%s", cm)
	logging.info("Validation stage finished in %.2fs", time.time() - start)


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Train pixel-wise Random Forest for land-use classification.")
	p.add_argument("--train-data", type=Path, default=Path(__file__).parent / "../../train_data", help="Dataset folder with sample subfolders.")
	p.add_argument("--model-out", type=Path, default=Path(__file__).parent / "rf_landuse_model.joblib", help="Output path for trained model.")
	return p.parse_args()


def main() -> None:
	t0 = time.time()
	args = parse_args()
	setup_logging(LOG_LEVEL)
	logging.info("Starting random-forest training pipeline")
	logging.info(
		"Config: train_data=%s val_ratio=%.2f n_estimators=%d max_depth=%s min_samples_leaf=%d train_pixel_fraction=%.3f seed=%d",
		args.train_data,
		VAL_RATIO,
		N_ESTIMATORS,
		str(MAX_DEPTH),
		MIN_SAMPLES_LEAF,
		TRAIN_PIXEL_FRACTION,
		SEED,
	)

	if not args.train_data.exists() or not args.train_data.is_dir():
		raise FileNotFoundError(f"train_data folder not found: {args.train_data}")

	sample_dirs = get_sample_dirs(args.train_data)
	train_dirs, val_dirs = split_samples(sample_dirs, VAL_RATIO, SEED)

	logging.info("Total samples: %d | train: %d | val: %d", len(sample_dirs), len(train_dirs), len(val_dirs))
	logging.info("Train sample IDs: %s", [p.name for p in train_dirs])
	logging.info("Val sample IDs: %s", [p.name for p in val_dirs])
	logging.info("Spectral features per pixel: 9 (true+swir+ndvi RGB)")

	data_t0 = time.time()
	x_train, y_train = collect_dataset(train_dirs, TRAIN_PIXEL_FRACTION, SEED)
	logging.info("Dataset collection finished in %.2fs", time.time() - data_t0)

	logging.info("Train matrix shape: X=%s y=%s", x_train.shape, y_train.shape)
	logging.info(
		"Feature stats: min=%.2f max=%.2f mean=%.2f std=%.2f",
		float(x_train.min()),
		float(x_train.max()),
		float(x_train.mean()),
		float(x_train.std()),
	)
	summarize_labels(y_train, "Train")

	model = RandomForestClassifier(
		n_estimators=N_ESTIMATORS,
		max_depth=MAX_DEPTH,
		min_samples_leaf=MIN_SAMPLES_LEAF,
		class_weight="balanced_subsample",
		n_jobs=-1,
		random_state=SEED,
		oob_score=True,
		verbose=0,
	)

	logging.info("Training RandomForest...")
	fit_t0 = time.time()
	model.fit(x_train, y_train)
	logging.info("Training finished in %.2fs", time.time() - fit_t0)
	logging.info("OOB score: %.4f", float(model.oob_score_))

	importances = model.feature_importances_
	top_idx = np.argsort(importances)[::-1][:8]
	top_text = ", ".join([f"{FEATURE_ORDER[i]}={importances[i]:.4f}" for i in top_idx])
	logging.info("Top feature importances: %s", top_text)

	args.model_out.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(
		{
			"model": model,
			"class_info": CLASS_INFO,
			"feature_order": FEATURE_ORDER,
		},
		args.model_out,
	)
	logging.info("Saved model to: %s", args.model_out)

	logging.info("Running validation (pixel-wise on full val images)...")
	evaluate_on_samples(model, val_dirs)
	logging.info("Pipeline finished in %.2fs", time.time() - t0)


if __name__ == "__main__":
	main()

