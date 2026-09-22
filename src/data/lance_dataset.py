"""Lance-backed fMRI pretraining dataset with batched random row access."""

from __future__ import annotations

import logging
import os
import random
from collections import OrderedDict
from typing import Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentations import FMRIDINOAugmentation, FMRIDeterministicMultiViewAugmentation
from .utils import (
    _crop_time,
    _enable_fc_from_args,
    _infer_expected_channels,
    _infer_raw_series_layout,
    _to_channel_time,
    _to_fc_matrix,
    _zscore_per_roi,
)


logger = logging.getLogger("neurojepa")


def _cfg_get(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _modality_embed_enabled(args) -> bool:
    model_cfg = getattr(args, "model", None) if args is not None else None
    modality_cfg = _cfg_get(model_cfg, "modality_embed")
    return bool(_cfg_get(modality_cfg, "enable", False))


class LanceSubjectSessionDataset(Dataset):
    """Read full ROI sessions from Lance and preserve subject/session sampling semantics.

    ``__getitems__`` lets PyTorch fetch a whole batch with one Lance ``take`` call.
    The Lance handle is opened lazily inside each worker and removed during pickling,
    which keeps forkserver/spawn workers independent and safe.
    """

    DATA_COLUMNS = ["row_id", "data", "shape"]
    METADATA_COLUMNS = ["row_id", "subject_key", "is_task", "source_path", "qc_pass"]

    def __init__(
        self,
        uri: str,
        lance_config,
        split: str,
        crop_length: int,
        transform: bool,
        args,
        transform_mode: str,
        seed: int = 42,
        selection_mode: str = "all_sessions",
    ):
        split = str(split).strip().lower()
        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported Lance split: {split}")
        self.uri = str(uri)
        self.split = split
        self.seed = int(seed)
        self.selection_mode = str(selection_mode or "all_sessions")
        valid_modes = {"epoch_random_one_session", "fixed_one_session", "fixed_session", "all_sessions"}
        if self.selection_mode not in valid_modes:
            raise ValueError(f"Unsupported subject_sampling.mode: {self.selection_mode}")

        self.endpoint = str(_cfg_get(lance_config, "endpoint", "") or "")
        self.region = str(_cfg_get(lance_config, "region", "cn-beijing") or "cn-beijing")
        self.virtual_hosted_style_request = bool(
            _cfg_get(lance_config, "virtual_hosted_style_request", True)
        )
        self.access_key_env = str(_cfg_get(lance_config, "access_key_env", "AWS_ACCESS_KEY_ID"))
        self.secret_key_env = str(_cfg_get(lance_config, "secret_key_env", "AWS_SECRET_ACCESS_KEY"))
        self.session_token_env = str(_cfg_get(lance_config, "session_token_env", "AWS_SESSION_TOKEN"))
        self.metadata_cache_size_bytes = int(
            _cfg_get(lance_config, "metadata_cache_size_bytes", 512 * 1024 * 1024)
        )
        self.require_qc_pass = bool(_cfg_get(lance_config, "require_qc_pass", True))
        self._dataset = None

        self.crop_length = int(crop_length)
        self.transform = bool(transform)
        self.transform_mode = str(transform_mode or ("train" if transform else "none"))
        self.expected_channels = _infer_expected_channels(args)
        self.raw_series_layout = _infer_raw_series_layout(args)
        self.enable_fc = _enable_fc_from_args(args)
        self.return_modality_id = _modality_embed_enabled(args)
        self.use_augment = False
        augment_cfg = getattr(getattr(args, "data", None), "augment", None) if args is not None else None
        self.full_sequence_to_augment = bool(getattr(augment_cfg, "full_sequence_to_augment", False))

        if self.transform_mode == "train":
            if args is None:
                raise ValueError("args is required for Lance pretraining augmentation")
            if self.enable_fc:
                raise ValueError("Lance pretraining does not support enable_fc=True with augmentation")
            self.augment = FMRIDINOAugmentation(args.data.augment, target_t=self.crop_length)
            self.use_augment = True
        elif self.transform_mode == "hard_val":
            if args is None:
                raise ValueError("args is required for Lance hard validation augmentation")
            if self.enable_fc:
                raise ValueError("Lance hard validation does not support enable_fc=True")
            hard_val_cfg = getattr(args.validation, "deterministic_multiview", None)
            if hard_val_cfg is not None and bool(getattr(hard_val_cfg, "enable", False)):
                self.augment = FMRIDeterministicMultiViewAugmentation(
                    hard_val_cfg,
                    target_t=self.crop_length,
                    expected_channels=self.expected_channels,
                )
                self.use_augment = True

        self._all_samples = self._load_metadata()
        if not self._all_samples:
            raise RuntimeError(f"No rows found in Lance split={self.split}: {self.uri}")
        self.subject_records = self._group_subjects(self._all_samples)
        self.samples = []
        self.file_paths = []
        self.set_epoch(0)
        logger.info(
            "Lance dataset loaded. uri=%s split=%s rows=%d subjects=%d mode=%s",
            self.uri,
            self.split,
            len(self._all_samples),
            len(self.subject_records),
            self.selection_mode,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_dataset"] = None
        return state

    def _storage_options(self) -> Optional[dict[str, str]]:
        if not self.uri.lower().startswith("s3://"):
            return None
        access_key = os.environ.get(self.access_key_env)
        secret_key = os.environ.get(self.secret_key_env)
        if not access_key or not secret_key:
            raise RuntimeError(
                f"Lance S3 credentials are missing. Set {self.access_key_env} and {self.secret_key_env}."
            )
        options = {
            "access_key_id": access_key,
            "secret_access_key": secret_key,
            "aws_region": self.region,
            "virtual_hosted_style_request": str(self.virtual_hosted_style_request).lower(),
        }
        if self.endpoint:
            options["aws_endpoint"] = self.endpoint
        session_token = os.environ.get(self.session_token_env)
        if session_token:
            options["aws_session_token"] = session_token
        return options

    def _open_dataset(self):
        if self._dataset is None:
            try:
                import lance
            except ImportError as exc:
                raise RuntimeError(
                    "The Lance data backend requires pylance. Install it with `pip install pylance==8.0.0`."
                ) from exc
            kwargs = {"metadata_cache_size_bytes": self.metadata_cache_size_bytes}
            storage_options = self._storage_options()
            if storage_options is not None:
                kwargs["storage_options"] = storage_options
            self._dataset = lance.dataset(self.uri, **kwargs)
        return self._dataset

    def _load_metadata(self) -> list[dict]:
        membership_column = "in_train" if self.split == "train" else "in_val"
        columns = [*self.METADATA_COLUMNS, membership_column]
        predicate = f"{membership_column} = true"
        if self.require_qc_pass:
            predicate += " AND qc_pass = true"
        table = self._open_dataset().to_table(columns=columns, filter=predicate)
        result = []
        for row in table.to_pylist():
            result.append(
                {
                    "row_id": int(row["row_id"]),
                    "subject_key": str(row["subject_key"]),
                    "is_task": bool(row["is_task"]),
                    "source_path": str(row["source_path"]),
                }
            )
        return result

    @staticmethod
    def _group_subjects(samples: Iterable[dict]) -> list[dict]:
        grouped: OrderedDict[str, dict] = OrderedDict()
        for sample in samples:
            key = sample["subject_key"]
            record = grouped.setdefault(key, {"subject_key": key, "samples": []})
            record["samples"].append(sample)
        return list(grouped.values())

    def set_epoch(self, epoch: int) -> None:
        if self.selection_mode == "all_sessions":
            self.samples = list(self._all_samples)
        else:
            if self.selection_mode in {"fixed_one_session", "fixed_session"}:
                rng = random.Random(self.seed)
            else:
                rng = random.Random(self.seed + int(epoch))
            self.samples = [rng.choice(record["samples"]) for record in self.subject_records]
        self.file_paths = [
            f"{self.uri}#row={sample['row_id']}:{sample['source_path']}" for sample in self.samples
        ]

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _extract_arrays(table) -> tuple[list[np.ndarray], list[list[int]]]:
        data_column = table["data"].combine_chunks()
        offsets = data_column.offsets.to_numpy(zero_copy_only=False)
        values = data_column.values.to_numpy(zero_copy_only=False)
        shapes = table["shape"].to_pylist()
        arrays = []
        for index, shape in enumerate(shapes):
            begin, end = int(offsets[index]), int(offsets[index + 1])
            shape = [int(value) for value in shape]
            flat = np.asarray(values[begin:end], dtype=np.float32)
            if int(np.prod(shape)) != flat.size:
                raise ValueError(f"Lance row has shape={shape} but data length={flat.size}")
            arrays.append(flat.reshape(shape))
        return arrays, shapes

    def _process_array(self, arr: np.ndarray, is_task: bool):
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
        ct = _crop_time(ct, crop_length, random_crop=self.transform)
        if ct is None:
            raise ValueError(
                f"Lance row is shorter than crop length {crop_length}; normalized shape={tuple(arr.shape)}"
            )
        ct = _zscore_per_roi(ct)
        if self.enable_fc:
            ct = _to_fc_matrix(ct)
        tensor = torch.from_numpy(ct).float()
        if self.use_augment:
            views = self.augment(tensor, is_task=is_task)
            return (views, is_task) if self.return_modality_id else views
        return (tensor, is_task) if self.return_modality_id else tensor

    def __getitems__(self, indices):
        normalized = [int(index) for index in indices]
        selected = [self.samples[index] for index in normalized]
        row_ids = [sample["row_id"] for sample in selected]
        table = self._open_dataset().take(row_ids, columns=self.DATA_COLUMNS)
        returned_ids = table["row_id"].to_pylist()
        if returned_ids != row_ids:
            raise RuntimeError("Lance take returned rows in an unexpected order")
        arrays, _ = self._extract_arrays(table)
        output = []
        for sample, arr in zip(selected, arrays):
            try:
                output.append(self._process_array(arr, bool(sample["is_task"])))
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to process Lance row={sample['row_id']} source={sample['source_path']}: {exc}"
                ) from exc
        return output

    def __getitem__(self, index):
        return self.__getitems__([index])[0]
