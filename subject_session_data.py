import json
import os
import random

import torch
from torch.utils.data import DistributedSampler

from src.data import (
    _build_variable_roi_collate_fn,
    _make_pretrain_loader,
    create_pretrain_dataloaders as _create_pretrain_dataloaders,
    fMRIDataset,
    fMRILeWorldDataset,
    fMRIDatasetWithlabel,
    fMRILeWorldDatasetWithlabel,
)
from src.data.augmentations import FMRIDINOAugmentation
from src.data.lance_dataset import LanceSubjectSessionDataset
from src.data.utils import (
    _crop_time,
    _enable_fc_from_args,
    _infer_expected_channels,
    _infer_raw_series_layout,
    _load_series_array,
    _to_channel_time,
    _to_fc_matrix,
    _zscore_per_roi,
)


def _dedupe_preserve_order(items):
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _modality_embed_enabled(args) -> bool:
    model_cfg = getattr(args, "model", None)
    modality_cfg = getattr(model_cfg, "modality_embed", None) if model_cfg is not None else None
    if modality_cfg is None and isinstance(model_cfg, dict):
        modality_cfg = model_cfg.get("modality_embed")
    if modality_cfg is None:
        return False
    if isinstance(modality_cfg, dict):
        return bool(modality_cfg.get("enable", False))
    return bool(getattr(modality_cfg, "enable", False))


class EpochRandomSubjectSessionDataset:
    """Expose one selected session per subject, with configurable epoch/fixed selection."""

    def __init__(
        self,
        subject_json_list,
        crop_length,
        transform,
        args,
        transform_mode,
        seed=42,
        selection_mode="epoch_random_one_session",
        use_leworld=False,
    ):
        if use_leworld:
            raise ValueError("EpochRandomSubjectSessionDataset currently supports LeJEPA/DINO-style augmentation only.")
        if isinstance(subject_json_list, (str, os.PathLike)):
            subject_json_list = [subject_json_list]

        self.seed = int(seed)
        self.selection_mode = str(selection_mode or "epoch_random_one_session")
        valid_modes = {"epoch_random_one_session", "fixed_one_session", "fixed_session", "all_sessions"}
        if self.selection_mode not in valid_modes:
            raise ValueError(f"Unsupported subject_sampling.mode: {self.selection_mode}")
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
                raise ValueError("args is required for subject-session augmentation.")
            if self.enable_fc:
                raise ValueError("subject-session pretraining does not support enable_fc=True with augmentation.")
            self.augment = FMRIDINOAugmentation(args.data.augment, target_t=self.crop_length)
            self.use_augment = True

        self.subject_records = []
        self.samples = []
        self.file_paths = []

        for json_path in subject_json_list:
            with open(json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            subjects = payload.get("subjects", [])
            if not isinstance(subjects, list):
                raise ValueError(f"Expected `subjects` list in {json_path}")
            for item in subjects:
                paths = _dedupe_preserve_order(
                    str(path)
                    for path in item.get("paths", [])
                    if str(path).lower().endswith((".npy", ".npz"))
                )
                if not paths:
                    continue
                self.subject_records.append(
                    {
                        "subject_key": str(item.get("subject_key", len(self.subject_records))),
                        "paths": paths,
                        "is_task": bool(item.get("is_task", False)),
                    }
                )

        if not self.subject_records:
            raise RuntimeError("No valid subject records found in train_subject_json.")
        self.set_epoch(0)

    def set_epoch(self, epoch):
        active_samples = []
        if self.selection_mode == "all_sessions":
            for record in self.subject_records:
                for path in record["paths"]:
                    active_samples.append(
                        {
                            "path": path,
                            "is_task": bool(record.get("is_task", False)),
                            "subject_key": record.get("subject_key"),
                        }
                    )
        else:
            if self.selection_mode in {"fixed_one_session", "fixed_session"}:
                rng = random.Random(self.seed)
            else:
                rng = random.Random(self.seed + int(epoch))
            for record in self.subject_records:
                active_samples.append(
                    {
                        "path": rng.choice(record["paths"]),
                        "is_task": bool(record.get("is_task", False)),
                        "subject_key": record.get("subject_key"),
                    }
                )
        if not active_samples:
            raise RuntimeError("Subject-session sampling produced no active samples.")
        self.samples = active_samples
        self.file_paths = [sample["path"] for sample in self.samples]

    def __len__(self):
        return len(self.file_paths)

    def _load_one_with_reason(self, idx, random_crop):
        file_path = self.samples[idx]["path"]
        try:
            arr = _load_series_array(file_path, raw_series_layout=self.raw_series_layout)
            ct = _to_channel_time(
                arr,
                self.expected_channels,
                raw_series_layout=self.raw_series_layout,
            )
            force_full_sequence = (
                self.transform_mode == "train"
                and self.use_augment
                and self.full_sequence_to_augment
            )
            crop_length = ct.shape[1] if force_full_sequence else self.crop_length
            ct = _crop_time(ct, crop_length, random_crop=random_crop)
            if ct is None:
                return None, f"time_length_too_short: expected at least {crop_length}"
            ct = _zscore_per_roi(ct)
            if self.enable_fc:
                ct = _to_fc_matrix(ct)
            return torch.from_numpy(ct).float(), None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    def __getitem__(self, idx):
        n = len(self.file_paths)
        for offset in range(min(n, 16)):
            j = (idx + offset) % n
            data_tensor, _ = self._load_one_with_reason(j, random_crop=self.transform)
            if data_tensor is None:
                continue
            if self.use_augment:
                is_task = bool(self.samples[j].get("is_task", False))
                views = self.augment(data_tensor, is_task=is_task)
                return (views, is_task) if self.return_modality_id else views
            is_task = bool(self.samples[j].get("is_task", False))
            return (data_tensor, is_task) if self.return_modality_id else data_tensor
        raise RuntimeError("Failed to fetch a valid subject-session sample after retries.")


def create_pretrain_dataloaders(config, is_distributed, rank, world_size):
    data_config = config["data"]
    data_backend = str(data_config.get("backend", "files") or "files").strip().lower()
    if data_backend == "lance":
        if config["training"]["probe_val"]:
            raise ValueError("The Lance backend is implemented for pretraining, not probe_val mode.")
        if str(config.model_chose) == "leworld":
            raise ValueError("The Lance backend currently supports LeJEPA/DINO-style pretraining only.")

        lance_cfg = data_config.get("lance", {})
        lance_uri = lance_cfg.get("uri") if isinstance(lance_cfg, dict) else getattr(lance_cfg, "uri", None)
        if not lance_uri:
            raise ValueError("data.lance.uri is required when data.backend=lance")

        subject_sampling_cfg = data_config.get("subject_sampling", {})
        selection_mode = (
            subject_sampling_cfg.get("mode", "all_sessions")
            if bool(subject_sampling_cfg.get("enable", False))
            else "all_sessions"
        )
        use_variable_roi_collate = (
            str(config.model_chose) == "lejepa"
            and bool(data_config.get("variable_roi_pretrain", False))
        )
        collate_fn = (
            _build_variable_roi_collate_fn(tuple(config.model.patch_size))
            if use_variable_roi_collate
            else None
        )

        train_dataset = LanceSubjectSessionDataset(
            uri=lance_uri,
            lance_config=lance_cfg,
            split="train",
            crop_length=data_config["input_seq_len"],
            transform=data_config["transform"],
            args=config,
            transform_mode="train" if data_config["transform"] else "none",
            seed=subject_sampling_cfg.get("seed", config["experiment"]["seed"]),
            selection_mode=selection_mode,
        )
        val_dataset = LanceSubjectSessionDataset(
            uri=lance_uri,
            lance_config=lance_cfg,
            split="val",
            crop_length=data_config["input_seq_len"],
            transform=data_config["transform"],
            args=config,
            transform_mode="train" if data_config["transform"] else "none",
            seed=subject_sampling_cfg.get("seed", config["experiment"]["seed"]),
            selection_mode="all_sessions",
        )

        if is_distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=config["experiment"]["seed"],
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
            )
        else:
            train_sampler = None
            val_sampler = None

        train_loader = _make_pretrain_loader(
            train_dataset,
            data_config,
            train_sampler,
            train_sampler is None,
            True,
            collate_fn=collate_fn,
        )
        val_loader = _make_pretrain_loader(
            val_dataset,
            data_config,
            val_sampler,
            False,
            False,
            collate_fn=collate_fn,
        )
        return train_loader, val_loader, train_sampler

    subject_sampling_cfg = data_config.get("subject_sampling", {})
    use_subject_sampling = (
        bool(subject_sampling_cfg.get("enable", False))
        and bool(data_config.get("train_subject_json", []))
    )
    if not use_subject_sampling:
        return _create_pretrain_dataloaders(config, is_distributed, rank, world_size)

    if config["training"]["probe_val"]:
        raise ValueError("subject_sampling is only implemented for pretraining, not probe_val mode.")

    use_leworld = str(config.model_chose) == "leworld"
    use_variable_roi_collate = (
        str(config.model_chose) == "lejepa"
        and bool(data_config.get("variable_roi_pretrain", False))
        and not bool(config["training"]["probe_val"])
    )
    collate_fn = _build_variable_roi_collate_fn(tuple(config.model.patch_size)) if use_variable_roi_collate else None

    train_dataset = EpochRandomSubjectSessionDataset(
        subject_json_list=data_config.get("train_subject_json"),
        crop_length=data_config["input_seq_len"],
        transform=data_config["transform"],
        args=config,
        transform_mode="train" if data_config["transform"] else "none",
        seed=subject_sampling_cfg.get("seed", config["experiment"]["seed"]),
        selection_mode=subject_sampling_cfg.get("mode", "epoch_random_one_session"),
        use_leworld=use_leworld,
    )

    val_sources = [(path, False) for path in data_config.get("val_list", [])]
    val_sources += [(path, True) for path in data_config.get("val_task_list", [])]
    val_cls = fMRILeWorldDataset if use_leworld else fMRIDataset
    val_dataset = val_cls(
        csv_list=val_sources,
        crop_length=data_config["input_seq_len"],
        transform=False if use_leworld else data_config["transform"],
        args=config,
        transform_mode="none" if use_leworld else ("train" if data_config["transform"] else "none"),
    )

    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config["experiment"]["seed"],
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
    else:
        train_sampler = None
        val_sampler = None

    train_loader = _make_pretrain_loader(
        train_dataset,
        data_config,
        train_sampler,
        train_sampler is None,
        True,
        collate_fn=collate_fn,
    )
    val_loader = _make_pretrain_loader(
        val_dataset,
        data_config,
        val_sampler,
        False,
        False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, train_sampler
