import argparse
import logging
from pathlib import Path

import numpy as np
from PIL import Image


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_rgb(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32)


def compute_ndbi_rgb(swir_rgb: np.ndarray, nir_rgb: np.ndarray) -> np.ndarray:
    if swir_rgb.shape[:2] != nir_rgb.shape[:2]:
        raise ValueError(
            f"SWIR and NIR size mismatch: swir={swir_rgb.shape[:2]} nir={nir_rgb.shape[:2]}"
        )

    swir = swir_rgb.mean(axis=2)
    nir = nir_rgb.mean(axis=2)

    ndbi = (swir - nir) / (swir + nir + 1e-6)
    ndbi_01 = (ndbi + 1.0) / 2.0
    ndbi_255 = np.clip(ndbi_01 * 255.0, 0, 255).astype(np.uint8)

    return np.stack([ndbi_255, ndbi_255, ndbi_255], axis=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create NDBI image from SWIR and NIR images.")
    p.add_argument("--swir-input", type=Path, required=True, help="Path to SWIR image")
    p.add_argument("--nir-input", type=Path, required=True, help="Path to NIR image")
    p.add_argument("--output", type=Path, required=True, help="Path to output NDBI image")
    p.add_argument("--log-level", type=str, default="INFO", help="DEBUG, INFO, WARNING...")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    logging.info("Loading SWIR: %s", args.swir_input)
    swir = load_rgb(args.swir_input)
    logging.info("Loading NIR: %s", args.nir_input)
    nir = load_rgb(args.nir_input)

    logging.info("Computing NDBI...")
    ndbi_rgb = compute_ndbi_rgb(swir, nir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(ndbi_rgb).save(args.output)

    logging.info("Saved NDBI image: %s", args.output)
    logging.info(
        "NDBI pixel stats (0..255): min=%d max=%d mean=%.2f",
        int(ndbi_rgb[:, :, 0].min()),
        int(ndbi_rgb[:, :, 0].max()),
        float(ndbi_rgb[:, :, 0].mean()),
    )


if __name__ == "__main__":
    main()
