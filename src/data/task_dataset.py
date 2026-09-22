import os
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import (
    _crop_time,
    _enable_fc_from_args,
    _infer_expected_channels,
    _infer_raw_series_layout,
    _load_series_array,
    _to_channel_time,
    _to_fc_matrix,
    _zscore_per_roi,
)


def _cfg_get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    if hasattr(obj, "get"):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _read_manifest(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Task manifest not found: {path}")
    if path.lower().endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


class TaskManifestDataset(Dataset):
    """Window-level fMRI dataset backed by task manifests.

    The manifest owns labels at sample/window level, unlike phenotype CSVs where
    one subject has one label. This keeps movie/task fMRI support small and
    avoids adding one dataset class per benchmark.
    """

    def __init__(
        self,
        manifest_list,
        crop_length=200,
        args=None,
        task_type="classification",
        label_col="label",
        label_cols=None,
    ):
        self.crop_length = int(crop_length)
        self.task_type = str(task_type)
        self.label_col = label_col or "label"
        self.label_cols = list(label_cols or [])
        self.expected_channels = _infer_expected_channels(args)
        self.raw_series_layout = _infer_raw_series_layout(args)
        self.enable_fc = _enable_fc_from_args(args)

        frames = []
        for manifest_path in manifest_list:
            df = _read_manifest(str(manifest_path))
            df["_manifest_path"] = str(manifest_path)
            frames.append(df)
        if not frames:
            raise RuntimeError("No task manifest files were provided.")

        self.manifest = pd.concat(frames, ignore_index=True)
        if "Path" not in self.manifest.columns:
            raise ValueError("Task manifest must contain a 'Path' column.")
        if "subject" not in self.manifest.columns:
            raise ValueError("Task manifest must contain a 'subject' column.")
        if "dataset" not in self.manifest.columns:
            self.manifest["dataset"] = "task_manifest"

        if self.task_type == "multilabel_classification":
            if not self.label_cols:
                raise ValueError("task.label_cols is required for multilabel_classification.")
            missing = [col for col in self.label_cols if col not in self.manifest.columns]
            if missing:
                raise ValueError(f"Missing multilabel columns in manifest: {missing}")
            self.manifest = self.manifest.dropna(subset=["Path", "subject", *self.label_cols]).copy()
        else:
            if self.label_col not in self.manifest.columns:
                raise ValueError(f"Task manifest must contain label column '{self.label_col}'.")
            self.manifest = self.manifest.dropna(subset=["Path", "subject", self.label_col]).copy()

        if len(self.manifest) == 0:
            raise RuntimeError("No valid rows remain after filtering task manifest labels.")

        self.manifest["Path"] = self.manifest["Path"].astype(str)
        self.manifest["dataset"] = self.manifest["dataset"].astype(str)
        self.manifest["subject"] = self.manifest["subject"].astype(str)

        self.path_to_subject: Dict[str, str] = dict(
            zip(self.manifest["Path"].tolist(), self.manifest["subject"].tolist())
        )
        self.file_list: List[Tuple[str, str, object]] = []
        for row in self.manifest.itertuples(index=False):
            dataset = str(getattr(row, "dataset"))
            path = str(getattr(row, "Path"))
            if self.task_type == "multilabel_classification":
                label = tuple(float(getattr(row, col)) for col in self.label_cols)
            else:
                label = getattr(row, self.label_col)
            self.file_list.append((dataset, path, label))

    def __len__(self):
        return len(self.file_list)

    def extract_subject_id(self, dataset_name, file_path):
        subject = self.path_to_subject.get(str(file_path))
        if subject is not None:
            return subject
        base_path = str(file_path).split("::", 1)[0]
        return self.path_to_subject.get(base_path)

    def get_regression_label_stats(self):
        if self.task_type != "regression":
            raise ValueError("Regression label stats are only available for regression tasks.")
        labels = torch.tensor(
            [float(label) for _, _, label in self.file_list],
            dtype=torch.float32,
        )
        mean = float(labels.mean().item())
        std = float(labels.std(unbiased=False).item())
        if std < 1e-8:
            std = 1.0
        return mean, std

    def _load_tensor(self, file_path):
        arr = _load_series_array(file_path, raw_series_layout=self.raw_series_layout)
        ct = _to_channel_time(
            arr,
            self.expected_channels,
            raw_series_layout=self.raw_series_layout,
        )
        ct = _crop_time(ct, self.crop_length, random_crop=False)
        if ct is None:
            raise RuntimeError(f"Sample shorter than crop_length={self.crop_length}: {file_path}")
        ct = _zscore_per_roi(ct)
        if self.enable_fc:
            ct = _to_fc_matrix(ct)
        return torch.from_numpy(ct).float()

    def __getitem__(self, idx):
        dataset_name, file_path, label_value = self.file_list[idx]
        fmri_tensor = self._load_tensor(file_path)

        if self.task_type == "classification":
            label_tensor = torch.tensor(int(label_value), dtype=torch.long)
        elif self.task_type == "regression":
            label_tensor = torch.tensor(float(label_value), dtype=torch.float32).view(1)
        elif self.task_type == "multilabel_classification":
            label_tensor = torch.tensor(label_value, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported task_type: {self.task_type}")

        return fmri_tensor, label_tensor
