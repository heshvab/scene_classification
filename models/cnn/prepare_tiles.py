import argparse
import logging
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
from PIL import Image


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_rgb_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def mask_to_binary(mask_rgb: np.ndarray) -> np.ndarray:
    r = mask_rgb[:, :, 0].astype(np.int16)
    g = mask_rgb[:, :, 1].astype(np.int16)
    b = mask_rgb[:, :, 2].astype(np.int16)

    # JPEG compression can alter exact color values, so use tolerant thresholds.
    road = (r < 50) & (g < 50) & (b < 50)
    not_road = (r > 150) & (g < 120) & (b < 120)

    binary = np.zeros((mask_rgb.shape[0], mask_rgb.shape[1]), dtype=np.uint8)
    binary[road] = 1

    unknown = ~(road | not_road)
    if np.any(unknown):
        logging.debug("Unknown mask pixels found: %d (treated as not-road)", int(unknown.sum()))

    return binary


def iter_tile_starts(length: int, tile_size: int, stride: int, cover_all: bool) -> Iterator[int]:
    if length <= tile_size:
        yield 0
        return

    last_start = length - tile_size
    starts = list(range(0, last_start + 1, stride))

    if cover_all and starts[-1] != last_start:
        starts.append(last_start)

    for s in starts:
        yield s


def extract_tile(arr: np.ndarray, y: int, x: int, tile_size: int) -> np.ndarray:
    return arr[y : y + tile_size, x : x + tile_size]


def save_mask_color(mask_binary: np.ndarray, output_path: Path) -> None:
    h, w = mask_binary.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    mask_rgb[mask_binary == 0] = np.array([255, 0, 0], dtype=np.uint8)
    mask_rgb[mask_binary == 1] = np.array([0, 0, 0], dtype=np.uint8)
    Image.fromarray(mask_rgb).save(output_path)


def process_sample(
    sample_dir: Path,
    out_true_dir: Path,
    out_swir_dir: Path,
    out_ndvi_dir: Path,
    out_masks_dir: Path,
    tile_size: int,
    stride: int,
    cover_all: bool,
) -> int:
    true_path = sample_dir / "true.jpg"
    swir_path = sample_dir / "swir.jpg"
    ndvi_path = sample_dir / "ndvi.jpg"
    mask_path = sample_dir / "mask.jpg"

    for p in [true_path, swir_path, ndvi_path, mask_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    true_rgb = load_rgb_image(true_path)
    swir_rgb = load_rgb_image(swir_path)
    ndvi_rgb = load_rgb_image(ndvi_path)
    mask_rgb = load_rgb_image(mask_path)
    mask_bin = mask_to_binary(mask_rgb)

    if not (true_rgb.shape[:2] == swir_rgb.shape[:2] == ndvi_rgb.shape[:2] == mask_bin.shape[:2]):
        raise ValueError(f"Image sizes do not match in sample: {sample_dir}")

    h, w = true_rgb.shape[:2]

    tile_count = 0
    sample_name = sample_dir.name

    for y in iter_tile_starts(h, tile_size, stride, cover_all):
        for x in iter_tile_starts(w, tile_size, stride, cover_all):
            true_tile = extract_tile(true_rgb, y, x, tile_size)
            swir_tile = extract_tile(swir_rgb, y, x, tile_size)
            ndvi_tile = extract_tile(ndvi_rgb, y, x, tile_size)
            mask_tile = extract_tile(mask_bin, y, x, tile_size)

            tile_id = f"{sample_name}_y{y}_x{x}"
            Image.fromarray(true_tile).save(out_true_dir / f"{tile_id}.png")
            Image.fromarray(swir_tile).save(out_swir_dir / f"{tile_id}.png")
            Image.fromarray(ndvi_tile).save(out_ndvi_dir / f"{tile_id}.png")
            save_mask_color(mask_tile, out_masks_dir / f"{tile_id}.png")
            tile_count += 1

    logging.info("Sample %s: generated %d tiles", sample_name, tile_count)
    return tile_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split road training data into 128x128 tiles.")
    parser.add_argument(\"--train-data\", type=Path, default=Path(__file__).parent / \"../../train_data\", help=\"Path to training source directory.\")
    parser.add_argument(\"--output\", type=Path, default=Path(__file__).parent / \"tiles_128\", help=\"Output folder for generated tiles.\")
    parser.add_argument("--tile-size", type=int, default=128, help="Tile size in pixels.")
    parser.add_argument("--stride", type=int, default=128, help="Stride for sliding window.")
    parser.add_argument(
        "--cover-all",
        action="store_true",
        help="Ensure right/bottom borders are covered by adding final tile positions.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    out_true_dir = args.output / "true"
    out_swir_dir = args.output / "swir"
    out_ndvi_dir = args.output / "ndvi"
    out_masks_dir = args.output / "masks"
    out_true_dir.mkdir(parents=True, exist_ok=True)
    out_swir_dir.mkdir(parents=True, exist_ok=True)
    out_ndvi_dir.mkdir(parents=True, exist_ok=True)
    out_masks_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted([p for p in args.train_data.iterdir() if p.is_dir()])
    if not sample_dirs:
        raise RuntimeError(f"No sample folders found in: {args.train_data}")

    total_tiles = 0
    for sample_dir in sample_dirs:
        total_tiles += process_sample(
            sample_dir=sample_dir,
            out_true_dir=out_true_dir,
            out_swir_dir=out_swir_dir,
            out_ndvi_dir=out_ndvi_dir,
            out_masks_dir=out_masks_dir,
            tile_size=args.tile_size,
            stride=args.stride,
            cover_all=args.cover_all,
        )

    logging.info("Done. Total generated tiles: %d", total_tiles)
    logging.info("True tiles: %s", out_true_dir)
    logging.info("SWIR tiles: %s", out_swir_dir)
    logging.info("NDVI tiles: %s", out_ndvi_dir)
    logging.info("Mask tiles: %s", out_masks_dir)


if __name__ == "__main__":
    main()
