import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from PIL import Image


DEFAULT_CLASS_INFO: Dict[int, Dict[str, Tuple[int, int, int]]] = {
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
	h, w = labels_2d.shape
	out = np.zeros((h, w, 3), dtype=np.uint8)
	for class_id, meta in class_info.items():
		out[labels_2d == int(class_id)] = np.array(meta["color"], dtype=np.uint8)
	return out


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Run pixel-wise land-use classification using trained KNN model.")
	p.add_argument("--model", type=Path, default=Path(__file__).parent / "knn_landuse_model.joblib", help="Path to trained model file.")
	p.add_argument("--true-image", type=Path, required=True, help="Path to true-color image.")
	p.add_argument("--swir-image", type=Path, required=True, help="Path to SWIR image.")
	p.add_argument("--ndvi-image", type=Path, required=True, help="Path to NDVI image.")
	p.add_argument("--output", type=Path, required=True, help="Output path for colorized predicted mask.")
	return p.parse_args()


def main() -> None:
	args = parse_args()
	setup_logging(LOG_LEVEL)

	if not args.model.exists():
		raise FileNotFoundError(f"Model file not found: {args.model}")

	logging.info("Loading model: %s", args.model)
	model_bundle = joblib.load(args.model)
	if isinstance(model_bundle, dict) and "model" in model_bundle:
		model = model_bundle["model"]
		class_info = model_bundle.get("class_info", DEFAULT_CLASS_INFO)
		feature_order = model_bundle.get("feature_order", FEATURE_ORDER)
	else:
		model = model_bundle
		class_info = DEFAULT_CLASS_INFO
		feature_order = FEATURE_ORDER

	if len(feature_order) != len(FEATURE_ORDER):
		raise RuntimeError(f"Model checkpoint was trained with {len(feature_order)} features, but the current script provides {len(FEATURE_ORDER)}. Retrain the model without NIR/NDBI.")

	images = {
		"true": load_rgb(args.true_image),
		"swir": load_rgb(args.swir_image),
		"ndvi": load_rgb(args.ndvi_image),
	}

	h, w = ensure_same_size(images)
	logging.info("Input image size: %dx%d", w, h)

	x = build_feature_matrix(images, feature_order)
	logging.info("Feature matrix shape: %s", x.shape)

	logging.info("Running prediction...")
	y_pred = model.predict(x).astype(np.uint8)
	pred_2d = y_pred.reshape(h, w)

	total = int(pred_2d.size)
	parts = []
	for class_id in sorted(class_info):
		count = int((pred_2d == int(class_id)).sum())
		pct = 100.0 * count / max(total, 1)
		class_name = class_info[class_id].get("name", str(class_id))
		parts.append(f"{class_name}={count} ({pct:.2f}%)")
	logging.info("Predicted class distribution: %s", " | ".join(parts))

	color_mask = labels_to_color_mask(pred_2d, class_info)
	args.output.parent.mkdir(parents=True, exist_ok=True)
	Image.fromarray(color_mask).save(args.output)
	logging.info("Saved color prediction to: %s", args.output)


if __name__ == "__main__":
	main()
