from .pretrain_dataset import (
    fMRIDataset,
    fMRILeWorldDataset,
    fMRILeWorldDatasetWithlabel,
    fMRIDatasetWithlabel,
)
from .adni_dataset import ADNIDataset
from .task_dataset import TaskManifestDataset
from .lance_dataset import LanceSubjectSessionDataset
from torch.utils.data import DataLoader, DistributedSampler
import torch


def _build_patch_valid_mask(roi_mask, time_len, patch_size):
    patch_c = int(patch_size[0])
    patch_t = int(patch_size[1])
    batch_size, roi_count = roi_mask.shape

    valid = roi_mask.unsqueeze(-1).expand(batch_size, roi_count, int(time_len))
    pad_c = (-roi_count) % patch_c
    pad_t = (-int(time_len)) % patch_t
    if pad_c > 0 or pad_t > 0:
        padded = torch.zeros(
            batch_size,
            roi_count + pad_c,
            int(time_len) + pad_t,
            dtype=torch.bool,
        )
        padded[:, :roi_count, :time_len] = valid
        valid = padded

    grid_c = valid.shape[1] // patch_c
    grid_t = valid.shape[2] // patch_t
    valid = valid.contiguous().view(batch_size, grid_c, patch_c, grid_t, patch_t)
    valid = valid.permute(0, 1, 3, 2, 4).reshape(batch_size, grid_c * grid_t, patch_c * patch_t)
    return valid.all(dim=-1)


def _collate_variable_roi_view(view_batch, patch_size, max_roi=None):
    batch_size = len(view_batch)
    time_len = int(view_batch[0].shape[1])
    if max_roi is None:
        max_roi = max(int(view.shape[0]) for view in view_batch)
    dtype = view_batch[0].dtype

    x = torch.zeros(batch_size, max_roi, time_len, dtype=dtype)
    roi_mask = torch.zeros(batch_size, max_roi, dtype=torch.bool)
    for idx, view in enumerate(view_batch):
        if view.ndim != 2:
            raise ValueError(f"Expected view tensor with shape (C, T), got {tuple(view.shape)}")
        if int(view.shape[1]) != time_len:
            raise ValueError(
                f"Variable-ROI collate expects a fixed temporal length per view, got {time_len} and {view.shape[1]}."
            )
        roi_count = int(view.shape[0])
        x[idx, :roi_count] = view
        roi_mask[idx, :roi_count] = True

    return {
        "x": x,
        "roi_mask": roi_mask,
        "patch_valid_mask": _build_patch_valid_mask(roi_mask, time_len, patch_size),
    }


class _VariableROICollate:
    """Pickle-safe variable-ROI collate callable for forkserver/spawn workers."""

    def __init__(self, patch_size):
        self.patch_size = tuple(patch_size)

    def __call__(self, batch):
        first = batch[0]
        if torch.is_tensor(first):
            return _collate_variable_roi_view(batch, self.patch_size)
        if isinstance(first, (list, tuple)):
            num_views = len(first)
            max_roi = max(int(sample[view_idx].shape[0]) for sample in batch for view_idx in range(num_views))
            return [
                _collate_variable_roi_view(
                    [sample[view_idx] for sample in batch],
                    self.patch_size,
                    max_roi=max_roi,
                )
                for view_idx in range(num_views)
            ]
        raise ValueError(f"Unsupported pretrain batch item for variable ROI collation: {type(first).__name__}")


def _build_variable_roi_collate_fn(patch_size):
    return _VariableROICollate(patch_size)


def _make_pretrain_loader(dataset, data_config, sampler, shuffle, drop_last, collate_fn=None):
    num_workers = int(data_config['num_workers'])
    loader_kwargs = {
        'dataset': dataset,
        'batch_size': data_config['batch_size'],
        'sampler': sampler,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': data_config['pin_memory'],
        'persistent_workers': bool(data_config['enable_persistent']) and num_workers > 0,
        'drop_last': drop_last,
        'collate_fn': collate_fn,
    }
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = data_config.get('prefetch_factor', 2)
        multiprocessing_context = data_config.get('multiprocessing_context')
        if multiprocessing_context:
            loader_kwargs['multiprocessing_context'] = str(multiprocessing_context)
    return DataLoader(**loader_kwargs)


def create_pretrain_dataloaders(config, is_distributed, rank, world_size):
    """Create train and validation dataloaders"""
    data_config = config['data']
    use_leworld = str(config.model_chose) == "leworld"
    use_variable_roi_collate = (
        str(config.model_chose) == "lejepa"
        and bool(data_config.get('variable_roi_pretrain', False))
        and not bool(config['training']['probe_val'])
    )
    collate_fn = _build_variable_roi_collate_fn(tuple(config.model.patch_size)) if use_variable_roi_collate else None

    if config['training']['probe_val']:
        dataset_cls = fMRILeWorldDatasetWithlabel if use_leworld else fMRIDatasetWithlabel
        train_dataset = dataset_cls(
            args=config,
            csv_list=data_config['train_list'],
            csv_root=data_config['csv_root'],
            crop_length=data_config['input_seq_len'],
            transform=data_config['transform'],
            dataset_names=data_config.get('datasets'),
        )

        val_dataset = dataset_cls(
            args=config,
            csv_list=data_config['val_list'],
            csv_root=data_config['csv_root'],
            crop_length=data_config['input_seq_len'],
            transform=False,
            dataset_names=data_config.get('datasets'),
        )

    else:
        train_sources = [(path, False) for path in data_config.get('train_list', [])]
        train_sources += [(path, True) for path in data_config.get('train_task_list', [])]
        val_sources = [(path, False) for path in data_config.get('val_list', [])]
        val_sources += [(path, True) for path in data_config.get('val_task_list', [])]
        dataset_cls = fMRILeWorldDataset if use_leworld else fMRIDataset
        subject_sampling_cfg = data_config.get('subject_sampling', {})
        use_subject_sampling = (
            bool(subject_sampling_cfg.get('enable', False))
            and bool(data_config.get('train_subject_json', []))
        )

        # Training dataset
        train_dataset = dataset_cls(
            csv_list=[] if use_subject_sampling else train_sources,
            crop_length=data_config['input_seq_len'],
            transform=data_config['transform'],
            args=config,
            transform_mode="train" if data_config['transform'] else "none",
            subject_session_json=data_config.get('train_subject_json') if use_subject_sampling else None,
            subject_sampling_seed=subject_sampling_cfg.get('seed', config['experiment']['seed']),
        )

        # Validation dataset
        val_dataset = dataset_cls(
            csv_list=val_sources,
            crop_length=data_config['input_seq_len'],
            transform=False if use_leworld else data_config['transform'],
            args=config,
            transform_mode="none" if use_leworld else ("train" if data_config['transform'] else "none"),
        )

    # Create samplers
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config['experiment']['seed']
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None

    # Create dataloaders
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


def create_pretrain_hard_val_dataloader(config, is_distributed, rank, world_size):
    data_config = config['data']
    use_leworld = str(config.model_chose) == "leworld"
    if use_leworld or config['training']['probe_val']:
        return None

    hard_val_cfg = getattr(config.validation, "deterministic_multiview", None)
    if hard_val_cfg is None or not bool(getattr(hard_val_cfg, "enable", False)):
        return None

    data_backend = str(data_config.get('backend', 'files') or 'files').strip().lower()
    if data_backend == 'lance':
        if use_leworld:
            raise ValueError("The Lance backend currently supports LeJEPA/DINO-style pretraining only.")
        lance_cfg = data_config.get('lance', {})
        lance_uri = lance_cfg.get('uri') if isinstance(lance_cfg, dict) else getattr(lance_cfg, 'uri', None)
        if not lance_uri:
            raise ValueError("data.lance.uri is required when data.backend=lance")
        val_dataset = LanceSubjectSessionDataset(
            uri=lance_uri,
            lance_config=lance_cfg,
            split='val',
            crop_length=data_config['input_seq_len'],
            transform=False,
            args=config,
            transform_mode='hard_val',
            seed=data_config.get('subject_sampling', {}).get('seed', config['experiment']['seed']),
            selection_mode='all_sessions',
        )
    else:
        dataset_cls = fMRILeWorldDataset if use_leworld else fMRIDataset
        val_sources = [(path, False) for path in data_config.get('val_list', [])]
        val_sources += [(path, True) for path in data_config.get('val_task_list', [])]
        val_dataset = dataset_cls(
            csv_list=val_sources,
            crop_length=data_config['input_seq_len'],
            transform=False,
            args=config,
            transform_mode="hard_val",
        )

    if is_distributed:
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
    else:
        val_sampler = None

    collate_fn = None
    if str(config.model_chose) == "lejepa" and bool(data_config.get('variable_roi_pretrain', False)):
        collate_fn = _build_variable_roi_collate_fn(tuple(config.model.patch_size))

    return _make_pretrain_loader(val_dataset, data_config, val_sampler, False, False, collate_fn=collate_fn)


def create_zeroshot_dataloader(config, is_distributed, rank, world_size):
    zs_cfg = config.validation.zero_shot
    data_cfg = config['data']

    val_dataset = fMRIDatasetWithlabel(
        args=config,
        csv_list=zs_cfg['val_list'],
        csv_root=zs_cfg['csv_root'],
        crop_length=data_cfg.get('input_seq_len'),
        transform=False,
        dataset_names=zs_cfg.get('datasets'),
    )

    if is_distributed:
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
    else:
        val_sampler = None

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg.get('batch_size'),
        sampler=val_sampler,
        shuffle=False,
        num_workers=data_cfg.get('num_workers'),
        pin_memory=data_cfg.get('pin_memory'),
        prefetch_factor=data_cfg.get('prefetch_factor'),
        drop_last=False
    )
    return val_loader


def create_downstream_dataloaders(config, is_distributed, rank, world_size):
    """Create train, validation, and test dataloaders"""
    data_config = config['data']
    task_config = config['task']
    

    if data_config['mode'] == "multi_site":
        train_dataset = fMRIDatasetWithlabel(
            args=config,
            csv_list=data_config['train_list'],
            csv_root=data_config['csv_root'],
            crop_length=data_config['input_seq_len'],
            transform=False,
            dataset_names=data_config.get('datasets'),
            task_type=task_config['task_type'],
            label_col=task_config.get('label_col'),
            split_name="train",
        )

        val_dataset = fMRIDatasetWithlabel(
            args=config,
            csv_list=data_config['val_list'],
            csv_root=data_config['csv_root'],
            crop_length=data_config['input_seq_len'],
            transform=False,
            dataset_names=data_config.get('datasets'),
            task_type=task_config['task_type'],
            label_col=task_config.get('label_col'),
            split_name="val",
        )

        test_dataset = fMRIDatasetWithlabel(
            args=config,
            csv_list=data_config['test_list'],
            csv_root=data_config['csv_root'],
            crop_length=data_config['input_seq_len'],
            transform=False,
            dataset_names=data_config.get('datasets'),
            task_type=task_config['task_type'],
            label_col=task_config.get('label_col'),
            split_name="test",
        )

    elif data_config['mode'] == "task_manifest":
        train_dataset = TaskManifestDataset(
            manifest_list=data_config['train_list'],
            crop_length=data_config['input_seq_len'],
            args=config,
            task_type=task_config['task_type'],
            label_col=task_config.get('label_col', 'label'),
            label_cols=task_config.get('label_cols'),
        )
        val_dataset = TaskManifestDataset(
            manifest_list=data_config['val_list'],
            crop_length=data_config['input_seq_len'],
            args=config,
            task_type=task_config['task_type'],
            label_col=task_config.get('label_col', 'label'),
            label_cols=task_config.get('label_cols'),
        )
        test_dataset = TaskManifestDataset(
            manifest_list=data_config['test_list'],
            crop_length=data_config['input_seq_len'],
            args=config,
            task_type=task_config['task_type'],
            label_col=task_config.get('label_col', 'label'),
            label_cols=task_config.get('label_cols'),
        )

    elif data_config['mode'] == "adni":
        required_splits = ('train_list', 'val_list', 'test_list')
        missing_splits = [key for key in required_splits if not data_config.get(key)]
        if missing_splits:
            raise ValueError(
                "data.mode='adni' requires user-supplied train_list, val_list, and test_list entries. "
                f"Missing: {', '.join(missing_splits)}"
            )
        train_dataset = ADNIDataset(txt_file=data_config['train_list'][0])
        val_dataset = ADNIDataset(txt_file=data_config['val_list'][0])
        test_dataset = ADNIDataset(txt_file=data_config['test_list'][0])

    else:
        raise ValueError(f"Unsupported downstream data.mode: {data_config['mode']}")

    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config['experiment']['seed']
        )
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    # Only pass prefetch_factor when multiprocessing is enabled.
    loader_kwargs = dict(
        batch_size=data_config['batch_size'],
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
    )
    if data_config['num_workers'] > 0:
        loader_kwargs['prefetch_factor'] = data_config.get('prefetch_factor', 2)

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        drop_last=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    return train_loader, val_loader, test_loader, train_sampler
