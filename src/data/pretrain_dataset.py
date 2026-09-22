import glob
import json
import os
import random
import re
from typing import List, Optional, Tuple, Union
import pandas as pd
import torch
from torch.utils.data import Dataset
from .augmentations import FMRIDINOAugmentation, FMRITemporalAugmentation
from .augmentations import FMRIDeterministicMultiViewAugmentation
from .utils import (
    _resolve_source_path,
    _load_series_paths_from_source,
    _load_series_array,
    _expand_long_series_virtual_segments,
    _to_channel_time,
    _crop_time,
    _zscore_per_roi,
    _infer_expected_channels,
    _infer_raw_series_layout,
    _dataset_name_from_path,
    _enable_fc_from_args,
    _to_fc_matrix,
)
import logging
logger = logging.getLogger("neurojepa")


def _dedupe_preserve_order(items):
    seen = set()
    kept = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        kept.append(item)
    return kept


def _get_data_cfg_value(args, key, default=None):
    if args is None or not hasattr(args, "data"):
        return default
    data_cfg = getattr(args, "data")
    if isinstance(data_cfg, dict):
        return data_cfg.get(key, default)
    return getattr(data_cfg, key, default)


_DEFAULT_COLUMN_ALIASES = {
    "Subject": ("Subject", "ID"),
    "Gender": ("Gender", "sex", "Gender.1"),
    "age": ("age", "Age", "AGE_AT_SCAN_1"),
}


def _find_matching_column(columns, candidates):
    lowered = {str(col).strip().lower(): col for col in columns}
    for candidate in candidates:
        matched = lowered.get(str(candidate).strip().lower())
        if matched is not None:
            return matched
    return None


def _resolve_column_name(columns, requested_name):
    requested_name = str(requested_name).strip()
    aliases = _DEFAULT_COLUMN_ALIASES.get(requested_name, ())
    return _find_matching_column(columns, (requested_name, *aliases))


def _normalize_subject_value(value) -> str:
    if pd.isna(value):
        return ""
    subject = str(value).strip()
    if re.fullmatch(r"-?\d+\.0+", subject):
        subject = subject.split(".", 1)[0]
    return subject


class fMRIDataset(Dataset):
    def __init__(
        self,
        csv_list,
        crop_length=40,
        transform=False,
        args=None,
        transform_mode=None,
        subject_session_json=None,
        subject_sampling_seed=42,
    ):
        self.samples = []
        self.file_paths: List[str] = []
        self.subject_sampling_enabled = subject_session_json is not None
        self.subject_records = []
        self.subject_sampling_seed = int(subject_sampling_seed)
        self.crop_length = int(crop_length)
        self.transform_mode = str(transform_mode or ("train" if transform else "none"))
        self.transform = self.transform_mode != "none"
        self.expected_channels = _infer_expected_channels(args)
        self.raw_series_layout = _infer_raw_series_layout(args)
        self.enable_fc = _enable_fc_from_args(args)
        self.use_augment = False
        augment_cfg = getattr(getattr(args, "data", None), "augment", None) if args is not None else None
        self.full_sequence_to_augment = bool(getattr(augment_cfg, "full_sequence_to_augment", False))

        if self.transform_mode == "train":
            if args is None:
                logger.warning("transform=True but args is None; disabling augmentation.")
            elif self.enable_fc:
                logger.warning("enable_fc=True with transform=True; disabling augmentation for FC inputs.")
            else:
                self.augment = FMRIDINOAugmentation(args.data.augment, target_t=self.crop_length)
                self.use_augment = True
        elif self.transform_mode == "hard_val":
            if args is None:
                logger.warning("transform_mode=hard_val but args is None; disabling deterministic validation augmentation.")
            elif self.enable_fc:
                logger.warning("enable_fc=True with transform_mode=hard_val; disabling augmentation for FC inputs.")
            else:
                hard_val_cfg = getattr(args.validation, "deterministic_multiview", None)
                if hard_val_cfg is not None and bool(getattr(hard_val_cfg, "enable", False)):
                    self.augment = FMRIDeterministicMultiViewAugmentation(
                        hard_val_cfg,
                        target_t=self.crop_length,
                        expected_channels=self.expected_channels,
                    )
                    self.use_augment = True

        if self.subject_sampling_enabled:
            self._load_subject_session_json(subject_session_json)
            self.set_epoch(0)
        else:
            source_paths = []
            for source in csv_list:
                is_task = False
                if isinstance(source, dict):
                    s = str(source.get("path", ""))
                    is_task = bool(source.get("is_task", False))
                elif isinstance(source, (tuple, list)) and len(source) >= 2:
                    s = str(source[0])
                    is_task = bool(source[1])
                else:
                    s = str(source)
                resolved_source = _resolve_source_path(s)
                if not os.path.exists(resolved_source):
                    logger.warning(f"Source file not found: {resolved_source}")
                    continue
                if resolved_source.lower().endswith((".npy", ".npz")):
                    source_paths.append((resolved_source, is_task))
                    continue
                if resolved_source.lower().endswith((".csv", ".txt")):
                    source_paths.append((resolved_source, is_task))
                    continue
                logger.warning(f"Only CSV/TXT/NPY/NPZ sources are supported now, skipping: {resolved_source}")

            for source_path, is_task in source_paths:
                if not os.path.exists(source_path):
                    logger.warning(f"Source file not found: {source_path}")
                    continue
                if source_path.lower().endswith((".npy", ".npz")):
                    self.samples.append({"path": source_path, "is_task": is_task})
                else:
                    self.samples.extend(
                        {"path": path, "is_task": is_task}
                        for path in _load_series_paths_from_source(source_path, path_col="Path")
                    )

            deduped_samples = []
            seen_paths = set()
            for sample in self.samples:
                path = sample["path"]
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                deduped_samples.append(sample)
            self.samples = deduped_samples
            self.file_paths = [sample["path"] for sample in self.samples]
        if len(self.file_paths) == 0:
            raise RuntimeError("No valid samples found from CSV sources for fMRIDataset.")
        task_count = sum(1 for sample in self.samples if sample.get("is_task", False))
        logger.info(f"Dataset loaded. Total files found: {len(self.file_paths)} (task={task_count})")

    def _load_subject_session_json(self, subject_session_json):
        json_paths = subject_session_json
        if isinstance(json_paths, (str, os.PathLike)):
            json_paths = [json_paths]

        records = []
        for json_path in json_paths:
            resolved_path = _resolve_source_path(str(json_path))
            if not os.path.exists(resolved_path):
                logger.warning(f"Subject-session JSON not found: {resolved_path}")
                continue
            with open(resolved_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            subjects = payload.get("subjects", [])
            if not isinstance(subjects, list):
                raise ValueError(f"Expected `subjects` list in {resolved_path}")
            for item in subjects:
                paths = [
                    _resolve_source_path(path)
                    for path in item.get("paths", [])
                    if str(path).lower().endswith((".npy", ".npz"))
                ]
                if not paths:
                    continue
                records.append(
                    {
                        "subject_key": str(item.get("subject_key", item.get("subject_id", len(records)))),
                        "paths": _dedupe_preserve_order(paths),
                        "is_task": bool(item.get("is_task", False)),
                    }
                )

        if not records:
            raise RuntimeError("No valid subjects found from subject-session JSON sources.")
        self.subject_records = records
        logger.info(
            "Subject-session dataset loaded. Subjects=%d, total_sessions=%d",
            len(self.subject_records),
            sum(len(record["paths"]) for record in self.subject_records),
        )

    def set_epoch(self, epoch: int) -> None:
        if not self.subject_sampling_enabled:
            return
        rng = random.Random(self.subject_sampling_seed + int(epoch))
        active_samples = []
        for record in self.subject_records:
            active_samples.append(
                {
                    "path": rng.choice(record["paths"]),
                    "is_task": bool(record.get("is_task", False)),
                    "subject_key": record.get("subject_key"),
                }
            )
        self.samples = active_samples
        self.file_paths = [sample["path"] for sample in self.samples]

    def __len__(self):
        return len(self.file_paths)

    def _load_one_with_reason(self, idx: int, random_crop: bool) -> Tuple[Optional[torch.Tensor], Optional[str]]:
        file_path = self.samples[idx]["path"]
        try:
            arr = _load_series_array(file_path, raw_series_layout=self.raw_series_layout)
            ct = _to_channel_time(
                arr,
                self.expected_channels,
                raw_series_layout=self.raw_series_layout,
            )
            force_full_sequence = (
                (self.transform_mode == "hard_val" and self.use_augment)
                or (self.transform_mode == "train" and self.use_augment and self.full_sequence_to_augment)
            )
            crop_length = ct.shape[1] if force_full_sequence else self.crop_length
            ct = _crop_time(ct, crop_length, random_crop=random_crop)
            if ct is None:
                reason = (
                    f"time_length_too_short: cropped input has T={arr.shape[-1] if arr.ndim == 2 else 'unknown'} "
                    f"before layout normalization, expected at least {crop_length}; "
                    f"post-layout shape={tuple(_to_channel_time(arr, self.expected_channels, self.raw_series_layout).shape)}"
                )
                return None, reason
            ct = _zscore_per_roi(ct)
            if self.enable_fc:
                ct = _to_fc_matrix(ct)
            return torch.from_numpy(ct).float(), None
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"

    def _load_one(self, idx: int, random_crop: bool) -> Optional[torch.Tensor]:
        data_tensor, reason = self._load_one_with_reason(idx, random_crop=random_crop)
        if data_tensor is None and reason is not None:
            logger.error(f"Error loading file {self.file_paths[idx]}: {reason}")
        return data_tensor

    def __getitem__(self, idx):
        # Robust retry for invalid samples.
        n = len(self.file_paths)
        failed_samples = []
        for k in range(min(n, 16)):
            j = (idx + k) % n
            is_task = bool(self.samples[j].get("is_task", False))
            data_tensor, reason = self._load_one_with_reason(j, random_crop=self.transform)
            if data_tensor is None:
                failed_samples.append((j, self.file_paths[j], reason or "unknown_error"))
                continue
            if self.use_augment:
                return self.augment(data_tensor, is_task=is_task)
            return data_tensor
        if failed_samples:
            detail_lines = [
                f"  retry[{attempt}] idx={sample_idx} path={path} reason={reason}"
                for attempt, (sample_idx, path, reason) in enumerate(failed_samples)
            ]
            logger.error(
                "Failed to fetch a valid sample after 16 retries.\n"
                f"  requested_idx={idx}\n"
                + "\n".join(detail_lines)
            )
        raise RuntimeError("Failed to fetch a valid sample after retries.")


class fMRILeWorldDataset(fMRIDataset):
    def __init__(
        self,
        csv_list,
        crop_length=40,
        transform=False,
        args=None,
        transform_mode=None,
        subject_session_json=None,
        subject_sampling_seed=42,
    ):
        super().__init__(
            csv_list=csv_list,
            crop_length=crop_length,
            transform=False,
            args=args,
            transform_mode=transform_mode,
            subject_session_json=subject_session_json,
            subject_sampling_seed=subject_sampling_seed,
        )
        if self.transform_mode == "hard_val":
            return
        self.transform = bool(transform)
        self.transform_mode = "train" if self.transform else "none"
        self.use_augment = False
        if self.transform:
            if args is None:
                logger.warning("transform=True but args is None; disabling augmentation.")
            elif self.enable_fc:
                logger.warning("enable_fc=True with transform=True; disabling augmentation for FC inputs.")
            else:
                self.augment = FMRITemporalAugmentation(args.data.augment, target_t=self.crop_length)
                self.use_augment = True


class fMRIDatasetWithlabel(Dataset):
    def __init__(
        self,
        csv_list,
        csv_root,
        crop_length=40,
        transform=False,
        args=None,
        return_dataset_idx=False,
        dataset_names=None,
        task_type="classification",
        label_col=None,
        split_name=None,
    ):
        self.file_list: List[Tuple[str, str, Union[int, float]]] = []
        self.crop_length = int(crop_length)
        self.transform = bool(transform)
        self.return_dataset_idx = bool(return_dataset_idx)
        self.task_type = str(task_type)
        self.label_col = label_col or ("Gender" if self.task_type == "classification" else "age")
        self._args_ref = args
        self.label_cache = {}
        self.dataset_id_map = {}
        self.dataset_lookup = {}
        self.expected_channels = _infer_expected_channels(args)
        self.raw_series_layout = _infer_raw_series_layout(args)
        self.enable_fc = _enable_fc_from_args(args)
        self.explicit_dataset_names = list(dataset_names) if dataset_names is not None else None
        self.split_name = str(split_name or "").strip().lower()

        if self.transform:
            if args is None:
                raise ValueError("args is required when transform=True")
            # if self.enable_fc:
            #     logger.warning("enable_fc=True with transform=True; disabling augmentation for FC inputs.")
            #     self.transform = False
            else:
                self.augment = FMRIDINOAugmentation(args.data.augment, target_t=self.crop_length)

        logger.info(f"Loading phenotype CSVs from {csv_root}...")
        for csv_path in sorted(glob.glob(os.path.join(csv_root, "*.csv"))):
            dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
            try:
                df = pd.read_csv(csv_path)
                subject_col = _resolve_column_name(df.columns, "Subject")
                label_col = _resolve_column_name(df.columns, self.label_col)
                if subject_col is None or label_col is None:
                    continue
                df[subject_col] = df[subject_col].map(_normalize_subject_value)
                df = df[df[subject_col] != ""].copy()
                if self.task_type == "classification":
                    df[label_col] = pd.to_numeric(df[label_col], errors='coerce').fillna(-1).astype(int)
                elif self.task_type == "regression":
                    df[label_col] = pd.to_numeric(df[label_col], errors='coerce')
                    df = df.dropna(subset=[label_col])
                else:
                    raise ValueError(f"Unsupported task_type: {self.task_type}")

                self.label_cache[dataset_name] = df.set_index(subject_col)[label_col].to_dict()
                self.dataset_lookup[dataset_name.lower()] = dataset_name
            except Exception as e:
                logger.error(f"Error reading {csv_path}: {e}")

        self.dataset_id_map = {name: i for i, name in enumerate(sorted(self.label_cache.keys()))}
        if self.explicit_dataset_names is not None and len(self.explicit_dataset_names) != len(csv_list):
            raise ValueError(
                f"dataset_names length ({len(self.explicit_dataset_names)}) must match csv_list length ({len(csv_list)})"
            )

        skipped_no_label = 0
        skipped_unknown_dataset = 0
        skipped_bad_subject_id = 0
        dataset_name_conflicts = 0
        conflict_examples: List[str] = []
        missing_label_examples: List[str] = []
        total_candidate_paths = 0
        segmented_source_paths = 0
        segmented_virtual_paths = 0
        source_dataset_names = (
            self.explicit_dataset_names
            if self.explicit_dataset_names is not None
            else [None] * len(csv_list)
        )

        for src_path, explicit_dataset_name in zip(csv_list, source_dataset_names):
            src_path = _resolve_source_path(src_path)
            if not os.path.exists(src_path):
                logger.warning(f"Source file not found: {src_path}")
                continue
            if not (src_path.lower().endswith(".csv") or src_path.lower().endswith(".txt")):
                logger.warning(f"Only CSV/TXT sources are supported now, skipping unsupported source: {src_path}")
                continue

            sample_paths: List[str] = _load_series_paths_from_source(src_path, path_col="Path")
            sample_paths, source_segmented_count, virtual_segment_count = self._maybe_expand_eval_sample_paths(sample_paths)
            segmented_source_paths += source_segmented_count
            segmented_virtual_paths += virtual_segment_count
            fixed_dataset_name = None
            if explicit_dataset_name is not None:
                fixed_dataset_name = self.dataset_lookup.get(str(explicit_dataset_name).lower())
                if fixed_dataset_name is None:
                    raise ValueError(
                        f"Unknown dataset name '{explicit_dataset_name}'. "
                        f"Available labeled datasets: {sorted(self.label_cache.keys())}"
                    )

            for f_path in sample_paths:
                total_candidate_paths += 1
                inferred_dataset_name = _dataset_name_from_path(f_path, self.dataset_lookup, up_levels=3)
                current_dataset_name = fixed_dataset_name
                if current_dataset_name is None:
                    current_dataset_name = inferred_dataset_name
                elif inferred_dataset_name is not None and inferred_dataset_name != current_dataset_name:
                    dataset_name_conflicts += 1
                    if len(conflict_examples) < 5:
                        conflict_examples.append(
                            f"{os.path.basename(f_path)} -> explicit={current_dataset_name}, inferred={inferred_dataset_name}"
                        )
                if current_dataset_name is None:
                    skipped_unknown_dataset += 1
                    continue

                sid = self.extract_subject_id(current_dataset_name, f_path)
                if not sid:
                    skipped_bad_subject_id += 1
                    continue
                label_value = self.label_cache.get(current_dataset_name, {}).get(sid, None)
                if self.task_type == "classification":
                    is_valid_label = label_value is not None and int(label_value) != -1
                else:
                    is_valid_label = label_value is not None

                if is_valid_label:
                    if self.task_type == "classification":
                        stored_label = int(label_value)
                    else:
                        stored_label = float(label_value)
                    self.file_list.append((current_dataset_name, f_path, stored_label))
                else:
                    skipped_no_label += 1
                    if len(missing_label_examples) < 5:
                        missing_label_examples.append(
                            f"{current_dataset_name}:{sid} ({os.path.basename(f_path)})"
                        )

        self.file_list = _dedupe_preserve_order(self.file_list)
        if len(self.file_list) == 0:
            if dataset_name_conflicts > 0:
                logger.error(
                    "Detected %d source paths whose inferred dataset name conflicts with the explicit dataset_names. "
                    "Examples: %s",
                    dataset_name_conflicts,
                    conflict_examples,
                )
            if missing_label_examples:
                logger.error(
                    "Examples of samples with parsed subject IDs but missing phenotype labels: %s",
                    missing_label_examples,
                )
            raise RuntimeError(
                "No valid labeled samples found from CSV sources. "
                f"Candidates={total_candidate_paths}, "
                f"skipped_missing_label={skipped_no_label}, "
                f"skipped_unknown_dataset={skipped_unknown_dataset}, "
                f"skipped_bad_subject_id={skipped_bad_subject_id}, "
                f"explicit_dataset_names={self.explicit_dataset_names}. "
                "This usually means the sample lists, dataset_names, and phenotype CSVs do not describe the same dataset."
            )
        logger.info(
            f"Dataset loaded: {len(self.file_list)} samples. "
            f"Skipped {skipped_no_label} due to missing {self.label_col}, "
            f"{skipped_unknown_dataset} due to unmatched dataset names, "
            f"{skipped_bad_subject_id} due to subject-id parsing failures."
        )
        if segmented_source_paths > 0:
            logger.info(
                "Auto-segmented %d long %s source samples into %d virtual segments "
                "(segment_length=%d).",
                segmented_source_paths,
                self.split_name or "dataset",
                segmented_virtual_paths,
                self.crop_length,
            )

    def _maybe_expand_eval_sample_paths(self, sample_paths: List[str]) -> Tuple[List[str], int, int]:
        enabled = bool(_get_data_cfg_value(self._args_ref, "auto_segment_long_samples", False))
        if not enabled:
            return sample_paths, 0, 0

        split_targets = _get_data_cfg_value(self._args_ref, "auto_segment_splits", ["val", "test"])
        if split_targets is None:
            split_targets = ["val", "test"]
        split_targets = {str(item).strip().lower() for item in split_targets}
        if "all" not in split_targets and self.split_name not in split_targets:
            return sample_paths, 0, 0

        segment_length = int(_get_data_cfg_value(self._args_ref, "auto_segment_length", self.crop_length))
        stride = _get_data_cfg_value(self._args_ref, "auto_segment_stride", segment_length)
        include_tail = bool(_get_data_cfg_value(self._args_ref, "auto_segment_include_tail", False))

        expanded_paths: List[str] = []
        segmented_source_paths = 0
        virtual_segment_count = 0
        for path_text in sample_paths:
            segments = _expand_long_series_virtual_segments(
                path_text,
                segment_length=segment_length,
                stride=stride,
                include_tail=include_tail,
                raw_series_layout=self.raw_series_layout,
            )
            if len(segments) > 1:
                segmented_source_paths += 1
                virtual_segment_count += len(segments)
            expanded_paths.extend(segments)
        return expanded_paths, segmented_source_paths, virtual_segment_count

    @staticmethod
    def extract_subject_id(dataset_name, file_path):
        filename = os.path.basename(file_path.replace('\\', '/'))
        stem = os.path.splitext(filename)[0]
        try:
            match = None
            if dataset_name in ["HCP", "PPMI", "SALD"]:
                if dataset_name == "PPMI":
                    match = re.search(r'sub-(\d{6})', filename)
                else:
                    match = re.search(r'(\d{6})', filename)
            elif dataset_name in ["ADHD"]:
                match = re.search(r'(\d{7})', filename)
            elif dataset_name in ["PNC"]:
                match = re.search(r'sub-(\d+)(?:_|$)', filename)
            elif dataset_name in ["CORR"]:
                match = re.search(r'(\d{7})', filename)
            elif dataset_name in ["CCNP"]:
                match = re.search(r'sub-[A-Za-z]+0*(\d+)(?:_|$)', stem)
            elif dataset_name in ["MDD"]:
                match = re.search(r'^(IS\d{3}-\d-\d{4})(?:_|$)', stem)
            elif dataset_name in ["CHCP"]:
                match = re.search(r'^(?:sub-)?(\d{4})(?:_|$)', filename)
            elif dataset_name in ["ISYB"]:
                match = re.search(r'sub-(\d{4})', filename)
            elif dataset_name in ["CalCC"]:
                match = re.search(r'sub-(CC\d{4})', filename)
            elif dataset_name in ["SLIM"]:
                match = re.search(r'sub-(\d{5})', filename)
            elif dataset_name in ["PIOP1"]:
                match = re.search(r'sub-(\d{1,4})(?:_|$)', filename)
            elif dataset_name in ["PIOP2"]:
                match = re.search(r'^(?:sub-)?0*(\d{1,4})(?:_|$)', filename)
            elif dataset_name in ["ABCD"]:
                match = re.search(r'(sub-[^_]+)', filename)
                return match.group(1) if match else None
            elif dataset_name in ["HBN"]:
                match = re.search(r'sub-([^_]+)', filename)
                return match.group(1) if match else None
            elif dataset_name in ["BHRC"]:
                match = re.search(r'(\d{5})', filename)
            elif dataset_name in ["NKI"]:
                match = re.search(r'(\d{8})', filename)
            elif dataset_name in ["ABIDE"]:
                match = re.search(r'(\d{7})', filename)
            elif dataset_name in ["ADNI", "ADNI(ALL)"]:
                match = re.search(r'sub-(\d{3}S\d{4})(?:_|$)', filename)
                return match.group(1) if match else None
            return match.group(1).lstrip('0') if match else None
        except Exception:
            return None

    def __len__(self):
        return len(self.file_list)

    def get_regression_label_stats(self) -> Tuple[float, float]:
        if self.task_type != "regression":
            raise ValueError("Regression label stats are only available for regression tasks.")

        labels = torch.tensor([float(label) for _, _, label in self.file_list], dtype=torch.float32)
        mean = float(labels.mean().item())
        std = float(labels.std(unbiased=False).item())
        if std < 1e-8:
            std = 1.0
        return mean, std

    def __getitem__(self, idx):
        dataset_name, file_path, label_value = self.file_list[idx]

        arr = _load_series_array(file_path, raw_series_layout=self.raw_series_layout)
        ct = _to_channel_time(
            arr,
            self.expected_channels,
            raw_series_layout=self.raw_series_layout,
        )
        ct = _crop_time(ct, self.crop_length, random_crop=self.transform)
        if ct is None:
            return self.__getitem__((idx + 1) % len(self.file_list))

        ct = _zscore_per_roi(ct)
        if self.enable_fc:
            ct = _to_fc_matrix(ct)
        fmri_tensor = torch.from_numpy(ct).float()
        if self.task_type == "classification":
            label_tensor = torch.tensor(int(label_value), dtype=torch.long)
        else:
            label_tensor = torch.tensor(float(label_value), dtype=torch.float32).view(1)

        if self.return_dataset_idx:
            dataset_tensor = torch.tensor(self.dataset_id_map[dataset_name], dtype=torch.long)
            if self.transform:
                return self.augment(fmri_tensor), label_tensor, dataset_tensor
            return fmri_tensor, label_tensor, dataset_tensor

        if self.transform:
            return self.augment(fmri_tensor), label_tensor
        return fmri_tensor, label_tensor


class fMRILeWorldDatasetWithlabel(fMRIDatasetWithlabel):
    def __init__(
        self,
        csv_list,
        csv_root,
        crop_length=40,
        transform=False,
        args=None,
        return_dataset_idx=False,
        dataset_names=None,
        task_type="classification",
        label_col=None,
    ):
        super().__init__(
            csv_list=csv_list,
            csv_root=csv_root,
            crop_length=crop_length,
            transform=False,
            args=args,
            return_dataset_idx=return_dataset_idx,
            dataset_names=dataset_names,
            task_type=task_type,
            label_col=label_col,
        )
        self.transform = bool(transform)
        if self.transform:
            if args is None:
                raise ValueError("args is required when transform=True")
            elif self.enable_fc:
                logger.warning("enable_fc=True with transform=True; disabling augmentation for FC inputs.")
                self.transform = False
            else:
                self.augment = FMRITemporalAugmentation(args.data.augment, target_t=self.crop_length)
