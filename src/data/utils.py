import glob
import os
import re
from typing import List, Optional

import numpy as np
import pandas as pd

import logging
logger = logging.getLogger("neurojepa")
_VIRTUAL_SEGMENT_PATTERN = re.compile(r"^(?P<raw>.+?)::(?P<range>\d{4}-\d{4})$")

try:
    from nilearn.connectome import ConnectivityMeasure
except Exception:
    ConnectivityMeasure = None



def _resolve_source_path(path: str, base_dir: Optional[str] = None) -> str:
    """Resolve a source path, preferring existing files."""
    raw = os.path.expanduser(str(path).strip())
    candidates = [raw]
    if base_dir:
        candidates.append(os.path.join(base_dir, raw))
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return os.path.abspath(candidates[0])


def _parse_virtual_segment_path(path: str):
    raw = str(path).strip()
    match = _VIRTUAL_SEGMENT_PATTERN.match(raw)
    if match is None:
        return None
    start_text, end_text = match.group("range").split("-", 1)
    return match.group("raw"), int(start_text), int(end_text)


def _load_series_paths_from_csv(csv_path: str, path_col: str = "Path") -> List[str]:
    """Read sample file paths from a CSV column (default: Path)."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Failed to read CSV {csv_path}: {e}")
        return []

    col_map = {str(c).strip(): c for c in df.columns}
    if path_col not in col_map:
        logger.warning(f"CSV {csv_path} does not contain '{path_col}' column. Available: {list(df.columns)}")
        return []

    resolved: List[str] = []
    missing = 0
    bad_ext = 0
    base_dir = os.path.dirname(csv_path)
    for raw_path in df[col_map[path_col]].dropna().astype(str).tolist():
        virtual = _parse_virtual_segment_path(raw_path)
        if virtual is not None:
            base_path, start, end = virtual
            p = _resolve_source_path(base_path, base_dir=base_dir)
            if not os.path.exists(p):
                missing += 1
                continue
            if not (p.endswith(".npy") or p.endswith(".npz")):
                bad_ext += 1
                continue
            resolved.append(f"{p}::{start:04d}-{end:04d}")
            continue

        p = _resolve_source_path(raw_path, base_dir=base_dir)
        if not os.path.exists(p):
            missing += 1
            continue
        if not (p.endswith(".npy") or p.endswith(".npz")):
            bad_ext += 1
            continue
        resolved.append(p)

    if missing > 0 or bad_ext > 0:
        logger.warning(
            f"CSV {csv_path}: kept {len(resolved)} paths, skipped missing={missing}, unsupported_ext={bad_ext}"
        )
    return resolved


def _load_series_paths_from_txt(txt_path: str) -> List[str]:
    """Read sample file paths from a TXT file, one path per line."""
    resolved: List[str] = []
    missing = 0
    bad_ext = 0
    base_dir = os.path.dirname(txt_path)

    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            raw_paths = [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Failed to read TXT {txt_path}: {e}")
        return []

    for raw_path in raw_paths:
        virtual = _parse_virtual_segment_path(raw_path)
        if virtual is not None:
            base_path, start, end = virtual
            p = _resolve_source_path(base_path, base_dir=base_dir)
            if not os.path.exists(p):
                missing += 1
                continue
            if not (p.endswith(".npy") or p.endswith(".npz")):
                bad_ext += 1
                continue
            resolved.append(f"{p}::{start:04d}-{end:04d}")
            continue

        p = _resolve_source_path(raw_path, base_dir=base_dir)
        if not os.path.exists(p):
            missing += 1
            continue
        if not (p.endswith(".npy") or p.endswith(".npz")):
            bad_ext += 1
            continue
        resolved.append(p)

    if missing > 0 or bad_ext > 0:
        logger.warning(
            f"TXT {txt_path}: kept {len(resolved)} paths, skipped missing={missing}, unsupported_ext={bad_ext}"
        )
    return resolved


def _load_series_paths_from_source(source_path: str, path_col: str = "Path") -> List[str]:
    """Read sample paths from either a CSV column or a TXT line list."""
    if source_path.lower().endswith(".csv"):
        return _load_series_paths_from_csv(source_path, path_col=path_col)
    if source_path.lower().endswith(".txt"):
        return _load_series_paths_from_txt(source_path)

    logger.warning(f"Unsupported source type for sample list: {source_path}")
    return []



def _normalize_raw_series_layout(layout: Optional[str]) -> Optional[str]:
    if layout is None:
        return None
    key = str(layout).strip().lower().replace("-", "_")
    aliases = {
        "auto": None,
        "time_channel": "time_channel",
        "time_first": "time_channel",
        "tc": "time_channel",
        "channel_time": "channel_time",
        "channel_first": "channel_time",
        "ct": "channel_time",
    }
    if key not in aliases:
        raise ValueError(
            f"Unsupported raw_series_layout={layout!r}. "
            "Expected one of: auto, time_channel, time_first, channel_time, channel_first."
        )
    return aliases[key]


def _load_series_array(file_path: str, raw_series_layout: Optional[str] = None) -> np.ndarray:
    virtual = _parse_virtual_segment_path(file_path)
    resolved_path = virtual[0] if virtual is not None else file_path
    raw_series_layout = _normalize_raw_series_layout(raw_series_layout)

    if resolved_path.endswith(".npz"):
        with np.load(resolved_path) as data_file:
            key = list(data_file.keys())[0]
            arr = data_file[key]
    elif resolved_path.endswith(".npy"):
        arr = np.load(resolved_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D ROI-level series, got shape {arr.shape} from {file_path}")
    if virtual is not None:
        _, start, end = virtual
        time_axis = 1 if raw_series_layout == "channel_time" else 0
        total_t = int(arr.shape[time_axis])
        if start < 0 or end < start or start >= total_t:
            raise ValueError(
                f"Virtual segment {start:04d}-{end:04d} is out of bounds for {resolved_path} with length {total_t}"
            )
        requested = end - start + 1
        slice_end = min(end, total_t - 1)
        if time_axis == 0:
            arr = arr[start : slice_end + 1]
            if arr.shape[0] < requested:
                pad = np.zeros((requested - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
                arr = np.concatenate([arr, pad], axis=0)
        else:
            arr = arr[:, start : slice_end + 1]
            if arr.shape[1] < requested:
                pad = np.zeros((arr.shape[0], requested - arr.shape[1]), dtype=arr.dtype)
                arr = np.concatenate([arr, pad], axis=1)
    return arr


def _get_series_time_length(file_path: str, raw_series_layout: Optional[str] = None) -> int:
    virtual = _parse_virtual_segment_path(file_path)
    resolved_path = virtual[0] if virtual is not None else file_path
    raw_series_layout = _normalize_raw_series_layout(raw_series_layout)

    if resolved_path.endswith(".npz"):
        with np.load(resolved_path) as data_file:
            key = list(data_file.keys())[0]
            arr = data_file[key]
    elif resolved_path.endswith(".npy"):
        arr = np.load(resolved_path, mmap_mode="r")
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D ROI-level series, got shape {arr.shape} from {file_path}")

    if virtual is not None:
        _, start, end = virtual
        return int(end - start + 1)

    time_axis = 1 if raw_series_layout == "channel_time" else 0
    if raw_series_layout is None and arr.shape[0] < arr.shape[1]:
        time_axis = 1
    return int(arr.shape[time_axis])


def _expand_long_series_virtual_segments(
    file_path: str,
    segment_length: int,
    stride: Optional[int] = None,
    include_tail: bool = False,
    raw_series_layout: Optional[str] = None,
) -> List[str]:
    path_text = str(file_path).strip()
    if not path_text:
        return []
    if _parse_virtual_segment_path(path_text) is not None:
        return [path_text]

    segment_length = int(segment_length)
    stride = int(stride) if stride is not None else int(segment_length)
    if segment_length <= 0:
        raise ValueError(f"segment_length must be > 0, got {segment_length}")
    if stride <= 0:
        raise ValueError(f"stride must be > 0, got {stride}")

    total_t = _get_series_time_length(path_text, raw_series_layout=raw_series_layout)
    if total_t <= segment_length:
        return [path_text]

    starts = list(range(0, total_t - segment_length + 1, stride))
    if include_tail:
        tail_start = total_t - segment_length
        if tail_start >= 0 and (not starts or starts[-1] != tail_start):
            starts.append(tail_start)

    return [f"{path_text}::{start:04d}-{start + segment_length - 1:04d}" for start in starts]


def _infer_expected_channels(args) -> Optional[int]:
    if args is None or not hasattr(args, "model"):
        return None
    img_size = getattr(args.model, "img_size", None)
    if isinstance(img_size, (tuple, list)) and len(img_size) >= 1:
        return int(img_size[0])
    return None


def _infer_raw_series_layout(args) -> Optional[str]:
    if args is None or not hasattr(args, "data"):
        return None
    data_cfg = getattr(args, "data")
    if isinstance(data_cfg, dict):
        raw_series_layout = data_cfg.get("raw_series_layout")
    else:
        raw_series_layout = getattr(data_cfg, "raw_series_layout", None)
    return _normalize_raw_series_layout(raw_series_layout)


def _enable_fc_from_args(args) -> bool:
    if args is None or not hasattr(args, "data"):
        return False
    data_cfg = getattr(args, "data")
    if isinstance(data_cfg, dict):
        return bool(data_cfg.get("enable_fc", False))
    return bool(getattr(data_cfg, "enable_fc", False))

def _dataset_name_from_path(file_path: str, dataset_lookup: dict, up_levels: int = 3) -> Optional[str]:
    """
    Infer dataset name from path components.

    Historically this used a fixed ancestor depth, but the ROI datasets in this
    repo do not share a single directory layout. We therefore scan ancestor
    folder names from nearest to farthest and return the first known dataset.
    """
    virtual = _parse_virtual_segment_path(file_path)
    file_path = virtual[0] if virtual is not None else file_path
    norm_path = str(file_path).replace("\\", "/").strip("/")
    parts = [p for p in norm_path.split("/") if p]
    if len(parts) < 2:
        return None

    for key in reversed(parts[:-1]):
        dataset_name = dataset_lookup.get(key.lower())
        if dataset_name is not None:
            return dataset_name

    if len(parts) < (up_levels + 1):
        return None
    key = parts[-(up_levels + 1)].lower()
    return dataset_lookup.get(key)

def _to_channel_time(
    arr: np.ndarray,
    expected_channels: Optional[int] = None,
    raw_series_layout: Optional[str] = None,
) -> np.ndarray:
    # Input could be (T, C) or (C, T). Output must be (C, T).
    raw_series_layout = _normalize_raw_series_layout(raw_series_layout)
    if raw_series_layout == "channel_time":
        return arr
    if raw_series_layout == "time_channel":
        return arr.T

    if expected_channels is not None:
        if arr.shape[0] == expected_channels:
            return arr
        if arr.shape[1] == expected_channels:
            return arr.T

    # Fallback heuristic: time dim is usually longer.
    if arr.shape[0] >= arr.shape[1]:
        return arr.T
    return arr


def _crop_time(ct: np.ndarray, crop_length: int, random_crop: bool) -> Optional[np.ndarray]:
    t = ct.shape[1]
    if t < crop_length:
        if random_crop:
            return None
        pad = np.zeros((ct.shape[0], crop_length - t), dtype=ct.dtype)
        return np.concatenate([ct, pad], axis=1)
    if t == crop_length:
        return ct

    if random_crop:
        start = np.random.randint(0, t - crop_length + 1)
    else:
        start = 0
    return ct[:, start : start + crop_length]


def _zscore_per_roi(ct: np.ndarray) -> np.ndarray:
    mu = ct.mean(axis=1, keepdims=True)
    sd = ct.std(axis=1, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return (ct - mu) / sd


def _to_fc_matrix(ct: np.ndarray) -> np.ndarray:
    """Convert ROI time series (C, T) to ROI-by-ROI Pearson FC."""
    ct = np.asarray(ct, dtype=np.float32)
    if ct.ndim != 2:
        raise ValueError(f"Expected (C, T) ROI series before FC conversion, got {ct.shape}")

    if ConnectivityMeasure is not None:
        fc = ConnectivityMeasure(kind="correlation").fit_transform([ct.T])[0]
    else:
        fc = np.corrcoef(ct)

    fc = np.asarray(fc, dtype=np.float32)
    fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
    fc = 0.5 * (fc + fc.T)
    np.fill_diagonal(fc, 1.0)
    return np.clip(fc, -1.0, 1.0)
