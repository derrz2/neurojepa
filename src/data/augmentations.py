import torch
import torch.nn.functional as F
from collections.abc import Sequence


class FMRIDINOAugmentation:
    """ROI-level (C, T) augmentation for contrastive/self-supervised training."""

    def __init__(self, args, target_t=40):
        self.args = args
        self.target_t = int(target_t)
        self.view_mode = str(args.view_mode)
        self.full_sequence_to_augment = bool(getattr(args, "full_sequence_to_augment", False))
        self.task_temporal_policy = str(getattr(args, "task_temporal_policy", "shared_window"))
        self.task_share_temporal_resample_factor = bool(
            getattr(args, "task_share_temporal_resample_factor", True)
        )

        temporal_cfg = getattr(args, "temporal", None)
        self.temp_enable = bool(getattr(temporal_cfg, "enable", False))
        self.temp_prob = float(getattr(temporal_cfg, "prob", 0.0))
        self.temp_scale = tuple(getattr(temporal_cfg, "scale_range", (1.0, 1.0)))

        self.global_crops_scale = tuple(getattr(args, "global_crops_scale", (0.7, 1.0)))
        self.local_crops_scale = tuple(getattr(args, "local_crops_scale", (0.4, 0.8)))
        self.view_crops_scale = tuple(getattr(args, "view_crops_scale", (0.6, 1.0)))
        self.global_channel_crops_scale = tuple(getattr(args, "global_channel_crops_scale", (1.0, 1.0)))
        self.local_channel_crops_scale = tuple(getattr(args, "local_channel_crops_scale", (1.0, 1.0)))
        self.view_channel_crops_scale = tuple(getattr(args, "view_channel_crops_scale", (1.0, 1.0)))

        self.global_output_size = self._resolve_output_size(getattr(args, "global_output_size", target_t))
        self.local_output_size = self._resolve_output_size(getattr(args, "local_output_size", target_t))
        self.view_output_size = self._resolve_output_size(getattr(args, "view_output_size", target_t))
        self.preserve_cropped_channels = bool(getattr(args, "preserve_cropped_channels", False))
        self.allow_variable_roi_batching = bool(getattr(args, "allow_variable_roi_batching", False))

        self.local_crops_number = int(getattr(args, "local_crops_number", 0))
        self.n_views = int(getattr(args, "n_views", 2))

        # ROI-level augment hyperparameters.
        self.roi_mask_prob = float(getattr(args, "roi_mask_prob", 0.10))
        self.time_mask_prob = float(getattr(args, "time_mask_prob", 0.10))
        self.time_mask_span_ratio = tuple(getattr(args, "time_mask_span_ratio", (0.05, 0.15)))
        self.scale_std = float(getattr(args, "scale_std", 0.05))
        self.jitter_std_ratio = float(getattr(args, "jitter_std_ratio", 0.02))
        self._validate_channel_resize_config()

    @staticmethod
    def _resolve_output_size(output_size):
        """Return (out_c, out_t). out_c can be None to keep full channels."""
        if isinstance(output_size, Sequence) and not isinstance(output_size, (str, bytes)):
            if len(output_size) == 1:
                return None, int(output_size[0])
            if len(output_size) >= 2:
                return int(output_size[0]), int(output_size[1])
        return None, int(output_size)

    def _temporal_resample(
        self,
        x: torch.Tensor,
        keep_full_sequence: bool = False,
        fit_to_target: bool = True,
    ) -> torch.Tensor:
        if (not self.temp_enable) or (torch.rand(1).item() >= self.temp_prob):
            return x

        t = x.shape[-1]
        new_t = max(1, int(round(t * torch.empty(1).uniform_(*self.temp_scale).item())))
        if keep_full_sequence:
            new_t = max(self.target_t, new_t)
        out = F.interpolate(x.unsqueeze(0), size=new_t, mode="linear", align_corners=False).squeeze(0)

        if keep_full_sequence:
            return out
        if not fit_to_target:
            return out

        if new_t > self.target_t:
            start = torch.randint(0, new_t - self.target_t + 1, (1,)).item()
            out = out[:, start : start + self.target_t]
        elif new_t < self.target_t:
            pad = self.target_t - new_t
            mode = "reflect" if new_t > 1 and pad < new_t else "replicate"
            out = F.pad(out.unsqueeze(0), (0, pad), mode=mode).squeeze(0)
        return out

    @staticmethod
    def _is_fixed_scale(scale_range) -> bool:
        if not isinstance(scale_range, Sequence) or len(scale_range) < 2:
            return True
        return abs(float(scale_range[0]) - float(scale_range[1])) < 1e-8

    def _validate_channel_resize_config(self) -> None:
        if (not self.preserve_cropped_channels) or self.allow_variable_roi_batching:
            return

        if self.view_mode == "single_view":
            scale_ranges = [("view_channel_crops_scale", self.view_channel_crops_scale)]
        elif self.view_mode == "multi_view":
            scale_ranges = [("view_channel_crops_scale", self.view_channel_crops_scale)]
        elif self.view_mode == "global_local":
            scale_ranges = [
                ("global_channel_crops_scale", self.global_channel_crops_scale),
                ("local_channel_crops_scale", self.local_channel_crops_scale),
            ]
        else:
            scale_ranges = []

        for name, scale_range in scale_ranges:
            if self._is_fixed_scale(scale_range):
                continue
            raise ValueError(
                "preserve_cropped_channels=True requires a fixed channel crop scale for batching. "
                f"Set `{name}` to a fixed value like [0.4, 0.4], got {tuple(scale_range)}."
            )

    def _crop_and_resize_ct(
        self,
        x: torch.Tensor,
        t_scale_range,
        output_size,
        c_scale_range=(1.0, 1.0),
        center_t=None,
        fixed_temporal_window=False,
        temporal_crop_mode="random",
        preserve_cropped_channels=False,
    ):
        c, t = x.shape
        out_c, out_t = output_size
        out_c = c if out_c is None else int(out_c)
        out_t = int(out_t)

        crop_c = max(1, min(c, int(round(c * torch.empty(1).uniform_(*c_scale_range).item()))))
        if temporal_crop_mode == "full":
            crop_t = t
        elif fixed_temporal_window:
            crop_t = max(1, min(t, out_t))
        else:
            crop_t = max(1, min(t, int(round(t * torch.empty(1).uniform_(*t_scale_range).item()))))

        start_c = torch.randint(0, max(1, c - crop_c + 1), (1,)).item()
        if temporal_crop_mode == "full":
            start_t = 0
        elif center_t is None:
            start_t = torch.randint(0, max(1, t - crop_t + 1), (1,)).item()
        else:
            start_min = max(0, int(center_t) - crop_t)
            start_max = min(t - crop_t, int(center_t))
            if start_max < start_min:
                start_t = max(0, min(t - crop_t, int(center_t) - crop_t // 2))
            else:
                start_t = torch.randint(start_min, start_max + 1, (1,)).item()

        crop = x[start_c : start_c + crop_c, start_t : start_t + crop_t]
        target_c = crop.shape[0] if preserve_cropped_channels else out_c
        if crop.shape[0] == target_c and crop.shape[1] == out_t:
            return crop, start_t + crop_t // 2

        resized = F.interpolate(
            crop.unsqueeze(0).unsqueeze(0),
            size=(target_c, out_t),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        return resized, start_t + crop_t // 2

    def _apply_roi_mask(self, x: torch.Tensor, prob: float) -> torch.Tensor:
        if prob <= 0.0:
            return x
        keep = (torch.rand(x.shape[0], device=x.device) > prob)
        if keep.all():
            return x
        y = x.clone()
        y[~keep] = 0.0
        return y

    def _apply_time_mask(self, x: torch.Tensor, prob: float) -> torch.Tensor:
        if prob <= 0.0 or torch.rand(1).item() >= prob:
            return x

        t = x.shape[-1]
        min_ratio, max_ratio = self.time_mask_span_ratio
        span = max(1, int(round(t * torch.empty(1).uniform_(min_ratio, max_ratio).item())))
        if span >= t:
            span = max(1, t - 1)
        start = torch.randint(0, max(1, t - span + 1), (1,)).item()

        y = x.clone()
        y[:, start : start + span] = 0.0
        return y

    def _apply_scaling(self, x: torch.Tensor, std: float) -> torch.Tensor:
        if std <= 0.0:
            return x
        scales = torch.randn((x.shape[0], 1), device=x.device, dtype=x.dtype) * std + 1.0
        return x * scales

    def _apply_jitter(self, x: torch.Tensor, std_ratio: float) -> torch.Tensor:
        if std_ratio <= 0.0:
            return x
        ch_std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        noise = torch.randn_like(x) * (std_ratio * ch_std)
        return x + noise

    def _augment_view(self, x: torch.Tensor, strength: str, is_task: bool = False) -> torch.Tensor:
        if strength == "global1":
            roi_mask_p, time_mask_p = self.roi_mask_prob * 0.8, self.time_mask_prob * 0.8
            scale_std, jitter_std = self.scale_std * 0.6, self.jitter_std_ratio * 0.6
            shift_max = 1
        elif strength == "global2":
            roi_mask_p, time_mask_p = self.roi_mask_prob, self.time_mask_prob
            scale_std, jitter_std = self.scale_std, self.jitter_std_ratio
            shift_max = 2
        else:  # local
            roi_mask_p, time_mask_p = self.roi_mask_prob * 1.2, self.time_mask_prob * 1.2
            scale_std, jitter_std = self.scale_std * 1.1, self.jitter_std_ratio * 1.1
            shift_max = 2

        if is_task:
            time_mask_p = 0.0
            shift_max = 0

        y = self._apply_scaling(x, scale_std)
        y = self._apply_jitter(y, jitter_std)
        y = self._apply_roi_mask(y, roi_mask_p)
        y = self._apply_time_mask(y, time_mask_p)

        if shift_max > 0:
            shift = int(torch.randint(-shift_max, shift_max + 1, (1,)).item())
            y = torch.roll(y, shifts=shift, dims=-1)
        return y

    @staticmethod
    def _zscore_per_roi(x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return (x - mu) / sd

    def __call__(self, image, is_task: bool = False):
        x = torch.as_tensor(image, dtype=torch.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected (C, T), got shape {tuple(x.shape)}")

        task_shared_window = is_task and self.task_temporal_policy == "shared_window"
        fixed_temporal_window = self.full_sequence_to_augment and x.shape[-1] > self.target_t
        temporal_crop_mode = "full" if task_shared_window else "random"
        if task_shared_window and self.task_share_temporal_resample_factor:
            x = self._temporal_resample(x, fit_to_target=False)
        elif not task_shared_window:
            x = self._temporal_resample(x, keep_full_sequence=fixed_temporal_window)
        crops = []

        def view_input():
            if task_shared_window and not self.task_share_temporal_resample_factor:
                return self._temporal_resample(x, fit_to_target=False)
            return x

        if self.view_mode == "single_view":
            crop, _ = self._crop_and_resize_ct(
                view_input(),
                self.view_crops_scale,
                self.view_output_size,
                self.view_channel_crops_scale,
                fixed_temporal_window=fixed_temporal_window,
                temporal_crop_mode=temporal_crop_mode,
                preserve_cropped_channels=self.preserve_cropped_channels,
            )
            return self._zscore_per_roi(crop)

        if self.view_mode == "multi_view":
            for i in range(self.n_views):
                crop, _ = self._crop_and_resize_ct(
                    view_input(),
                    self.view_crops_scale,
                    self.view_output_size,
                    self.view_channel_crops_scale,
                    fixed_temporal_window=fixed_temporal_window,
                    temporal_crop_mode=temporal_crop_mode,
                    preserve_cropped_channels=self.preserve_cropped_channels,
                )
                aug = self._augment_view(crop, "global1" if i % 2 == 0 else "global2", is_task=is_task)
                crops.append(self._zscore_per_roi(aug))

        elif self.view_mode == "global_local":
            centers = []
            for strength in ["global1", "global2"]:
                crop, center = self._crop_and_resize_ct(
                    view_input(),
                    self.global_crops_scale,
                    self.global_output_size,
                    self.global_channel_crops_scale,
                    fixed_temporal_window=fixed_temporal_window,
                    temporal_crop_mode=temporal_crop_mode,
                    preserve_cropped_channels=self.preserve_cropped_channels,
                )
                centers.append(center)
                aug = self._augment_view(crop, strength, is_task=is_task)
                crops.append(self._zscore_per_roi(aug))

            split = self.local_crops_number // 2
            for i in range(self.local_crops_number):
                ref_center = centers[0 if i < split else 1]
                crop, _ = self._crop_and_resize_ct(
                    view_input(),
                    self.local_crops_scale,
                    self.local_output_size,
                    self.local_channel_crops_scale,
                    center_t=ref_center,
                    fixed_temporal_window=fixed_temporal_window,
                    temporal_crop_mode=temporal_crop_mode,
                    preserve_cropped_channels=self.preserve_cropped_channels,
                )
                aug = self._augment_view(crop, "local", is_task=is_task)
                crops.append(self._zscore_per_roi(aug))
        else:
            raise ValueError(f"Unknown view_mode: {self.view_mode}")

        return crops


class FMRIDeterministicMultiViewAugmentation:
    """Deterministic global-local views for scaling-law validation."""

    def __init__(self, args, target_t=40, expected_channels=None):
        self.args = args
        self.target_t = int(target_t)
        self.expected_channels = int(expected_channels) if expected_channels is not None else None
        self.n_global_views = int(getattr(args, "n_global_views", 2))
        self.n_local_views = int(getattr(args, "n_local_views", 4))
        self.global_output_size = FMRIDINOAugmentation._resolve_output_size(
            getattr(args, "global_output_size", (None, target_t))
        )
        self.local_output_size = FMRIDINOAugmentation._resolve_output_size(
            getattr(args, "local_output_size", (None, target_t))
        )
        self.global_temporal_centers = tuple(getattr(args, "global_temporal_centers", (0.33, 0.67)))
        self.local_temporal_centers = tuple(getattr(args, "local_temporal_centers", (0.2, 0.4, 0.6, 0.8)))
        self.use_full_channels = bool(getattr(args, "use_full_channels", True))
        self.preserve_cropped_channels = bool(getattr(args, "preserve_cropped_channels", False))

    @staticmethod
    def _zscore_per_roi(x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return (x - mu) / sd

    @staticmethod
    def _center_to_start(length: int, crop_len: int, center_fraction: float) -> int:
        if crop_len >= length:
            return 0
        center_fraction = float(max(0.0, min(1.0, center_fraction)))
        center = int(round(center_fraction * (length - 1)))
        start = center - crop_len // 2
        return max(0, min(length - crop_len, start))

    def _deterministic_crop_and_resize_ct(self, x: torch.Tensor, output_size, center_fraction: float) -> torch.Tensor:
        c, t = x.shape
        out_c, out_t = output_size
        out_c = c if self.use_full_channels or out_c is None else int(out_c)
        out_t = int(out_t)

        crop_t = min(t, out_t)
        start_t = self._center_to_start(t, crop_t, center_fraction)

        if self.use_full_channels:
            start_c = 0
            crop_c = c
        else:
            crop_c = min(c, out_c)
            start_c = max(0, (c - crop_c) // 2)

        crop = x[start_c : start_c + crop_c, start_t : start_t + crop_t]
        target_c = crop.shape[0] if self.preserve_cropped_channels else out_c
        if crop.shape[0] == target_c and crop.shape[1] == out_t:
            return crop

        return F.interpolate(
            crop.unsqueeze(0).unsqueeze(0),
            size=(target_c, out_t),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)

    def __call__(self, image, is_task: bool = False):
        x = torch.as_tensor(image, dtype=torch.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected (C, T), got shape {tuple(x.shape)}")

        crops = []
        for center_fraction in self.global_temporal_centers[: self.n_global_views]:
            crop = self._deterministic_crop_and_resize_ct(x, self.global_output_size, center_fraction)
            crops.append(self._zscore_per_roi(crop))

        for center_fraction in self.local_temporal_centers[: self.n_local_views]:
            crop = self._deterministic_crop_and_resize_ct(x, self.local_output_size, center_fraction)
            crops.append(self._zscore_per_roi(crop))

        return crops


class FMRITemporalAugmentation:
    """Weak augmentation tuned for temporal predictive pretraining."""

    def __init__(self, args, target_t=40):
        self.args = args
        self.target_t = int(target_t)
        self.roi_mask_prob = float(getattr(args, "roi_mask_prob", 0.02))
        self.time_mask_prob = float(getattr(args, "time_mask_prob", 0.02))
        self.time_mask_span_ratio = tuple(getattr(args, "time_mask_span_ratio", (0.03, 0.08)))
        self.scale_std = float(getattr(args, "scale_std", 0.01))
        self.jitter_std_ratio = float(getattr(args, "jitter_std_ratio", 0.005))
        self.time_shift_max = int(getattr(args, "time_shift_max", 1))

    def _fit_length(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[-1]
        if t == self.target_t:
            return x
        if t > self.target_t:
            return x[:, : self.target_t]
        pad = self.target_t - t
        mode = "reflect" if t > 1 and pad < t else "replicate"
        return F.pad(x.unsqueeze(0), (0, pad), mode=mode).squeeze(0)

    def _apply_scaling(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale_std <= 0.0:
            return x
        scales = torch.randn((x.shape[0], 1), device=x.device, dtype=x.dtype) * self.scale_std + 1.0
        return x * scales

    def _apply_jitter(self, x: torch.Tensor) -> torch.Tensor:
        if self.jitter_std_ratio <= 0.0:
            return x
        ch_std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        noise = torch.randn_like(x) * (self.jitter_std_ratio * ch_std)
        return x + noise

    def _apply_roi_mask(self, x: torch.Tensor) -> torch.Tensor:
        if self.roi_mask_prob <= 0.0:
            return x
        keep = torch.rand(x.shape[0], device=x.device) > self.roi_mask_prob
        if keep.all():
            return x
        y = x.clone()
        y[~keep] = 0.0
        return y

    def _apply_time_mask(self, x: torch.Tensor) -> torch.Tensor:
        if self.time_mask_prob <= 0.0 or torch.rand(1).item() >= self.time_mask_prob:
            return x
        t = x.shape[-1]
        min_ratio, max_ratio = self.time_mask_span_ratio
        span = max(1, int(round(t * torch.empty(1).uniform_(min_ratio, max_ratio).item())))
        span = min(span, max(1, t - 1))
        start = torch.randint(0, max(1, t - span + 1), (1,)).item()
        y = x.clone()
        y[:, start : start + span] = 0.0
        return y

    @staticmethod
    def _zscore_per_roi(x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return (x - mu) / sd

    def __call__(self, image, is_task: bool = False):
        x = torch.as_tensor(image, dtype=torch.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected (C, T), got shape {tuple(x.shape)}")

        y = self._fit_length(x)
        y = self._apply_scaling(y)
        y = self._apply_jitter(y)
        y = self._apply_roi_mask(y)
        y = self._apply_time_mask(y)
        if self.time_shift_max > 0:
            shift = int(torch.randint(-self.time_shift_max, self.time_shift_max + 1, (1,)).item())
            y = torch.roll(y, shifts=shift, dims=-1)
        return self._zscore_per_roi(y)
