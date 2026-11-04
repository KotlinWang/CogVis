"""Input contract of the xBD instance export: one materialised route, no decode fork.

The inference entry point never decodes imagery itself.  It reads a
**materialised input manifest**: one JSONL row per frozen pair, giving the paths
of the PRE/POST RGB and of the common valid-domain mask.  Whoever materialises
owns the decode (raster, dtype, domain), so the model route has exactly one way
to obtain pixels and cannot silently disagree with the other methods on the same
common505 scope.

:func:`materialize_published_pair` is the retained decode helper: it drives the
published reader (``Benchmark.inputs(pair_id, oracle_assets=False)``) and writes
the three files of one manifest row.  Materialisation is a separate, CPU-only
step; the inference route above does not import the reader at all.

No answer-key field is ever read: the loader whitelists the input fields it
needs, so ``gt_counts`` / ``gt_total`` cannot reach the model, the predictions or
the report even when they sit in the same rows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

#: Fields the inference route reads from a materialised input manifest.
MATERIALIZED_INPUT_FIELDS = (
    "image_id",
    "pair_id",
    "dataset_id",
    "event_group_id",
    "split",
    "pre_rgb",
    "post_rgb",
    "valid_mask",
    "width",
    "height",
)

#: Fields without which a row cannot be scored or attributed.
REQUIRED_INPUT_FIELDS = ("image_id", "pair_id", "pre_rgb", "post_rgb", "valid_mask")

def load_input_manifest(path: Path) -> List[Dict[str, object]]:
    """Read the materialised input manifest, whitelisting input fields only."""
    path = Path(path)
    rows: List[Dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = [field for field in REQUIRED_INPUT_FIELDS if field not in record]
            if missing:
                raise ValueError(
                    f"{path}:{line_number} is not a materialised input row: missing {missing}. "
                    f"required fields are {list(REQUIRED_INPUT_FIELDS)}"
                )
            rows.append({field: record[field] for field in MATERIALIZED_INPUT_FIELDS if field in record})
    if not rows:
        raise ValueError(f"{path} holds no input rows; refusing to run over an empty scope")
    return rows


def select_rows(rows: Sequence[Dict[str, object]], limit: int, offset: int) -> List[Dict[str, object]]:
    if int(offset) < 0 or int(limit) < 0:
        raise ValueError(f"limit/offset must be non-negative, got limit={limit} offset={offset}")
    selected = list(rows[int(offset):])
    if int(limit) > 0:
        selected = selected[: int(limit)]
    return selected


def load_materialized_pair(row: Dict[str, object]):
    """Load one materialised pair and prove it matches its own declaration."""
    pre = np.load(str(row["pre_rgb"]), allow_pickle=False)
    post = np.load(str(row["post_rgb"]), allow_pickle=False)
    valid = np.load(str(row["valid_mask"]), allow_pickle=False)

    for name, array in (("pre_rgb", pre), ("post_rgb", post)):
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"{row['pair_id']}: {name} has shape {array.shape}; expected HxWx3")
        if array.dtype != np.uint8:
            raise ValueError(
                f"{row['pair_id']}: {name} has dtype {array.dtype}; the materialised route stores the "
                "8-bit RGB the model consumes, so a float or wider integer array means the "
                "materialisation step is not the one this route expects"
            )
    if pre.shape != post.shape:
        raise ValueError(f"{row['pair_id']}: PRE {pre.shape} and POST {post.shape} differ")
    if valid.dtype != np.bool_:
        raise ValueError(
            f"{row['pair_id']}: valid_mask has dtype {valid.dtype}; expected bool"
        )
    if valid.shape != pre.shape[:2]:
        raise ValueError(
            f"{row['pair_id']}: valid_mask {valid.shape} does not match the PRE grid {pre.shape[:2]}"
        )
    if int(valid.sum()) == 0:
        raise ValueError(f"{row['pair_id']}: the common valid domain is empty")

    declared = (row.get("height"), row.get("width"))
    if all(value is not None for value in declared) and declared != pre.shape[:2]:
        raise ValueError(
            f"{row['pair_id']}: manifest declares {(declared[0], declared[1])} but the PRE array is "
            f"{pre.shape[:2]}"
        )
    return (
        Image.fromarray(pre, mode="RGB"),
        Image.fromarray(post, mode="RGB"),
        valid,
        {"pre": "uint8_verbatim", "post": "uint8_verbatim"},
    )


# --------------------------------------------------------------------------- #
# the retained materialisation helper (CPU only, never imported by inference)
# --------------------------------------------------------------------------- #
def import_published_reader(published_root):
    """Import the published reader without writing into the published tree.

    The reader ships as ``<published_root>/code/ovda_benchmark``; it is imported
    from there instead of being vendored, so materialisation always decodes with
    the same code the evaluator uses.
    """
    root = Path(published_root)
    code = root / "code"
    reader_path = code / "ovda_benchmark" / "reader.py"
    if not reader_path.exists():
        raise FileNotFoundError(
            f"the published reader is missing: expected {reader_path}. Point published_root at the "
            "published benchmark root (the directory holding code/, index.sqlite and pairs.jsonl)"
        )
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    from ovda_benchmark.reader import Benchmark

    return Benchmark, str(reader_path)


def image_from_reader_array(array: np.ndarray) -> Tuple[Image.Image, str]:
    """Convert a published-reader RGB array to the 8-bit PIL image the model sees.

    The reader preserves native radiometry (int16, or float32 when a POST is
    warped onto the PRE grid).  The audited xBD data lies inside [0, 255]; the
    conversion is returned as a label and stored in the manifest row so a float
    pair can never pass silently.
    """
    array = np.asarray(array)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"published reader returned {array.shape}; expected HxWx3 RGB")
    if array.dtype == np.uint8:
        return Image.fromarray(array, mode="RGB"), "uint8_verbatim"
    clipped = np.clip(array.astype(np.float64), 0.0, 255.0)
    if np.issubdtype(array.dtype, np.integer):
        note = f"{array.dtype}_clip"
    else:
        clipped = np.rint(clipped)
        note = "float32_round"
    return Image.fromarray(clipped.astype(np.uint8), mode="RGB"), note


def decode_published_pair(reader, row: Dict[str, object]):
    """Decode one frozen pair through the published reader (never oracle assets)."""
    payload = reader.inputs(str(row["pair_id"]), oracle_assets=False)
    pre_image, pre_note = image_from_reader_array(payload["pre"])
    post_image, post_note = image_from_reader_array(payload["post"])
    valid = np.asarray(payload["valid"], dtype=bool)
    expected = (int(pre_image.size[1]), int(pre_image.size[0]))
    if valid.shape != expected:
        raise ValueError(
            f"{row['pair_id']}: reader valid mask {valid.shape} does not match the PRE grid {expected}"
        )
    if int(valid.sum()) == 0:
        raise ValueError(f"{row['pair_id']}: the published evaluation domain is empty")
    return pre_image, post_image, valid, {"pre": pre_note, "post": post_note}


def materialize_published_pair(published_root, output_root, row: Dict[str, object]) -> Dict[str, object]:
    """Write one frozen pair as the three files of a materialised input row.

    Returns the materialised manifest row (caller appends it to the manifest
    JSONL).  This is the only place that decodes xBD imagery for this run, and it
    never touches a GPU.
    """
    Benchmark, reader_path = import_published_reader(published_root)
    reader = Benchmark(Path(published_root))
    try:
        pre_image, post_image, valid, notes = decode_published_pair(reader, row)
    finally:
        reader.close()

    pair_id = str(row["pair_id"])
    directory = Path(output_root) / pair_id.replace("/", "__")
    directory.mkdir(parents=True, exist_ok=True)
    pre_path = directory / "pre_rgb.npy"
    post_path = directory / "post_rgb.npy"
    valid_path = directory / "valid_mask.npy"
    np.save(pre_path, np.asarray(pre_image, dtype=np.uint8))
    np.save(post_path, np.asarray(post_image, dtype=np.uint8))
    np.save(valid_path, np.asarray(valid, dtype=bool))

    height, width = np.asarray(pre_image).shape[:2]
    materialised = {
        "image_id": int(row["image_id"]),
        "pair_id": pair_id,
        "dataset_id": str(row.get("dataset_id", "xbd")),
        "pre_rgb": str(pre_path),
        "post_rgb": str(post_path),
        "valid_mask": str(valid_path),
        "width": int(width),
        "height": int(height),
        "valid_pixels": int(valid.sum()),
        "input_dtypes": notes,
        "reader_module": reader_path,
    }
    if "event_group_id" in row:
        materialised["event_group_id"] = row["event_group_id"]
    if "split" in row:
        materialised["split"] = row["split"]
    return materialised
