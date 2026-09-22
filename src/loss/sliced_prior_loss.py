import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ..utils.utils import gather


class SlicedPriorMatchingLoss(nn.Module):
    """Random 1D projection regularizers for non-uniform feature priors."""

    def __init__(
        self,
        loss_type: str = "wasserstein",
        num_slices: int = 256,
        feature_source: str = "centers",
        target_distribution: str = "student_t",
        target_df: float = 2.0,
        target_scale: float = 1.0,
        gather_distributed: bool = True,
        normalize_slices: bool = True,
        wasserstein_p: int = 1,
        num_bins: int = 8,
        occupancy_temperature: float = 0.25,
        occupancy_powerlaw_exponent: float = 1.0,
        occupancy_max_radius: float = 3.0,
        kurtosis_target: float = 6.0,
        pairwise_max_samples: int = 48,
        variance_floor_weight: float = 0.0,
        variance_floor_target: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.loss_type = str(loss_type).lower()
        self.num_slices = int(num_slices)
        self.feature_source = str(feature_source).lower()
        self.target_distribution = str(target_distribution).lower()
        self.target_df = float(target_df)
        self.target_scale = float(target_scale)
        self.gather_distributed = bool(gather_distributed)
        self.normalize_slices = bool(normalize_slices)
        self.wasserstein_p = int(wasserstein_p)
        self.num_bins = int(num_bins)
        self.occupancy_temperature = float(occupancy_temperature)
        self.occupancy_powerlaw_exponent = float(occupancy_powerlaw_exponent)
        self.occupancy_max_radius = float(occupancy_max_radius)
        self.kurtosis_target = float(kurtosis_target)
        self.pairwise_max_samples = int(pairwise_max_samples)
        self.variance_floor_weight = float(variance_floor_weight)
        self.variance_floor_target = float(variance_floor_target)
        self.eps = float(eps)
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))
        self._generator = None
        self._generator_device = None

        supported = {"wasserstein", "occupancy", "kurtosis", "pairwise", "none"}
        if self.loss_type not in supported:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

    def _get_generator(self, device: torch.device, seed: int) -> torch.Generator:
        if self._generator is None or self._generator_device != device:
            self._generator = torch.Generator(device=device)
            self._generator_device = device
        self._generator.manual_seed(seed)
        return self._generator

    @torch.no_grad()
    def _next_seed(self) -> int:
        step = self.global_step.detach().clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(step, op=dist.ReduceOp.MAX)
        seed = int(step.item())
        self.global_step.add_(1)
        return seed

    def _sample_directions(self, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        seed = self._next_seed()
        generator = self._get_generator(device, seed)
        directions = torch.randn(
            dim,
            self.num_slices,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        return F.normalize(directions, dim=0, eps=self.eps)

    def _select_features(self, view_embeddings: torch.Tensor, n_anchor_views: int) -> torch.Tensor:
        if view_embeddings.ndim != 3:
            raise ValueError(
                f"Expected view_embeddings to have shape (V, B, D), got {tuple(view_embeddings.shape)}."
            )
        if n_anchor_views <= 0 or n_anchor_views > view_embeddings.shape[0]:
            raise ValueError(
                f"Expected 1 <= n_anchor_views <= {view_embeddings.shape[0]}, got {n_anchor_views}."
            )

        if self.feature_source == "centers":
            features = view_embeddings[:n_anchor_views].mean(dim=0)
        elif self.feature_source == "proj":
            features = view_embeddings.reshape(-1, view_embeddings.shape[-1])
        elif self.feature_source == "anchors":
            features = view_embeddings[:n_anchor_views].reshape(-1, view_embeddings.shape[-1])
        elif self.feature_source == "locals":
            if view_embeddings.shape[0] <= n_anchor_views:
                raise ValueError("feature_source='locals' requires local views.")
            features = view_embeddings[n_anchor_views:].reshape(-1, view_embeddings.shape[-1])
        else:
            raise ValueError(f"Unsupported feature_source: {self.feature_source}")

        if self.gather_distributed and dist.is_available() and dist.is_initialized():
            features = gather(features)
        return features.float()

    def _standardize(self, values: torch.Tensor) -> torch.Tensor:
        mean = values.mean(dim=0, keepdim=True)
        std = values.std(dim=0, unbiased=False, keepdim=True).clamp_min(self.eps)
        return (values - mean) / std

    def _sample_target(self, shape, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        with torch.no_grad():
            if self.target_distribution == "student_t":
                distn = torch.distributions.StudentT(df=self.target_df, loc=0.0, scale=self.target_scale)
                target = distn.sample(shape)
            elif self.target_distribution == "laplace":
                distn = torch.distributions.Laplace(loc=0.0, scale=self.target_scale)
                target = distn.sample(shape)
            elif self.target_distribution == "normal":
                target = torch.randn(shape) * self.target_scale
            elif self.target_distribution == "gaussian_scale_mixture":
                scales = torch.exp(torch.randn(shape) * 0.5) * self.target_scale
                target = torch.randn(shape) * scales
            else:
                raise ValueError(f"Unsupported target_distribution: {self.target_distribution}")
        return target.to(device=device, dtype=dtype)

    def _power_law_target(self, size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ranks = torch.arange(1, size + 1, device=device, dtype=dtype)
        target = ranks.pow(-self.occupancy_powerlaw_exponent)
        return target / target.sum().clamp_min(self.eps)

    def _variance_floor(self, raw_proj: torch.Tensor) -> torch.Tensor:
        if self.variance_floor_weight <= 0:
            return raw_proj.new_zeros(())
        proj_std = raw_proj.std(dim=0, unbiased=False)
        floor_loss = F.relu(self.variance_floor_target - proj_std).mean()
        return floor_loss * self.variance_floor_weight

    def _wasserstein_loss(self, proj: torch.Tensor, device: torch.device, dtype: torch.dtype):
        target = self._sample_target(proj.shape, device=device, dtype=dtype)
        proj_sorted = proj.sort(dim=0).values
        target_sorted = target.sort(dim=0).values
        diff = (proj_sorted - target_sorted).abs()
        if self.wasserstein_p == 2:
            diff = diff.square()
        loss = diff.mean()
        metrics = {
            "prior_proj_std": proj.std(dim=0, unbiased=False).mean(),
            "prior_proj_abs_mean": proj.abs().mean(),
        }
        return loss, metrics

    def _occupancy_loss(self, proj: torch.Tensor):
        magnitudes = proj.abs()
        radius = max(self.occupancy_max_radius, self.eps)
        centers = torch.linspace(0.0, radius, self.num_bins, device=proj.device, dtype=proj.dtype)
        scaled = (magnitudes.unsqueeze(-1) - centers.view(1, 1, -1)) / max(self.occupancy_temperature, self.eps)
        assignments = F.softmax(-(scaled.square()), dim=-1)
        occupancy = assignments.mean(dim=0)
        target = self._power_law_target(self.num_bins, device=proj.device, dtype=proj.dtype)
        kl = target.unsqueeze(0) * (
            torch.log(target.unsqueeze(0) + self.eps) - torch.log(occupancy + self.eps)
        )
        loss = kl.sum(dim=-1).mean()
        occupancy_mean = occupancy.mean(dim=0)
        metrics = {
            "prior_center_mass": occupancy_mean[0],
            "prior_tail_mass": occupancy_mean[-1],
            "prior_proj_std": proj.std(dim=0, unbiased=False).mean(),
        }
        return loss, metrics

    def _kurtosis_loss(self, proj: torch.Tensor):
        second = proj.square().mean(dim=0).clamp_min(self.eps)
        fourth = proj.pow(4).mean(dim=0)
        excess_kurtosis = fourth / second.square() - 3.0
        target = torch.full_like(excess_kurtosis, self.kurtosis_target)
        loss = F.mse_loss(excess_kurtosis, target)
        metrics = {
            "prior_kurtosis": excess_kurtosis.mean(),
            "prior_proj_std": proj.std(dim=0, unbiased=False).mean(),
        }
        return loss, metrics

    def _pairwise_loss(self, proj: torch.Tensor, device: torch.device, dtype: torch.dtype):
        sample_count = min(proj.shape[0], self.pairwise_max_samples)
        if sample_count < 2:
            return proj.new_zeros(()), {
                "prior_pairwise_mean": proj.new_zeros(()),
                "prior_proj_std": proj.std(dim=0, unbiased=False).mean(),
            }

        seed = self._next_seed()
        generator = self._get_generator(proj.device, seed)
        indices = torch.randperm(proj.shape[0], device=proj.device, generator=generator)[:sample_count]
        proj_sub = proj[indices]
        target_sub = self._sample_target(proj_sub.shape, device=device, dtype=dtype)

        proj_pairwise = (proj_sub.unsqueeze(1) - proj_sub.unsqueeze(0)).abs()
        target_pairwise = (target_sub.unsqueeze(1) - target_sub.unsqueeze(0)).abs()
        mask = torch.triu(torch.ones(sample_count, sample_count, device=proj.device, dtype=torch.bool), diagonal=1)
        proj_pairs = proj_pairwise[mask]
        target_pairs = target_pairwise[mask]

        proj_sorted = proj_pairs.sort(dim=0).values
        target_sorted = target_pairs.sort(dim=0).values
        diff = (proj_sorted - target_sorted).abs()
        if self.wasserstein_p == 2:
            diff = diff.square()
        loss = diff.mean()
        metrics = {
            "prior_pairwise_mean": proj_pairs.mean(),
            "prior_proj_std": proj.std(dim=0, unbiased=False).mean(),
        }
        return loss, metrics

    def forward(self, view_embeddings: torch.Tensor, n_anchor_views: int):
        features = self._select_features(view_embeddings, n_anchor_views=n_anchor_views)
        directions = self._sample_directions(features.shape[-1], device=features.device, dtype=features.dtype)
        raw_proj = features @ directions
        proj = self._standardize(raw_proj) if self.normalize_slices else raw_proj

        if self.loss_type == "none":
            loss = proj.new_zeros(())
            metrics = {
                "prior_proj_std": raw_proj.std(dim=0, unbiased=False).mean(),
            }
        elif self.loss_type == "wasserstein":
            loss, metrics = self._wasserstein_loss(proj, device=features.device, dtype=features.dtype)
        elif self.loss_type == "occupancy":
            loss, metrics = self._occupancy_loss(proj)
        elif self.loss_type == "kurtosis":
            loss, metrics = self._kurtosis_loss(proj)
        elif self.loss_type == "pairwise":
            loss, metrics = self._pairwise_loss(proj, device=features.device, dtype=features.dtype)
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

        variance_floor = self._variance_floor(raw_proj)
        metrics["prior_variance_floor"] = variance_floor.detach()
        metrics["prior_feature_std"] = features.std(dim=0, unbiased=False).mean()
        total_loss = loss + variance_floor
        return total_loss, metrics
