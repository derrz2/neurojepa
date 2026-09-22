import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypePairLoss(nn.Module):
    """
    Prototype objective for downstream classification.
    Variants:
    - lite: total = L_vi + lambda_div * L_div
    - full: total = L_vi + lambda_view * L_vv + lambda_div * L_div
    """

    def __init__(
        self,
        loss_variant: str = "lite",
        lambda_view: float = 0.5,
        lambda_div: float = 0.0,
        use_diversity: bool = False,
    ):
        super().__init__()
        self.loss_variant = str(loss_variant).lower()
        if self.loss_variant not in {"lite", "full"}:
            raise ValueError(f"Unsupported loss_variant: {loss_variant}")
        self.lambda_view = float(lambda_view)
        self.lambda_div = float(lambda_div)
        self.use_diversity = bool(use_diversity)

    def forward(self, z1: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor, z2: torch.Tensor = None):
        z1 = F.normalize(z1.float(), dim=-1)
        proto = F.normalize(prototypes.float(), dim=-1)
        labels = labels.long().view(-1)

        target_proto = proto[labels]
        l_vi_1 = 1.0 - (z1 * target_proto).sum(dim=-1)

        l_vv = z1.new_zeros(z1.shape[0])
        l_vi = l_vi_1

        if z2 is not None and self.loss_variant == "full":
            z2 = F.normalize(z2.float(), dim=-1)
            l_vi_2 = 1.0 - (z2 * target_proto).sum(dim=-1)
            l_vi = 0.5 * (l_vi_1 + l_vi_2)
            l_vv = 1.0 - (z1 * z2).sum(dim=-1)

        l_vi_mean = l_vi.mean()
        l_vv_mean = l_vv.mean()
        total = l_vi_mean
        if self.loss_variant == "full":
            total = total + self.lambda_view * l_vv_mean

        l_div = z1.new_zeros(())
        if self.use_diversity and proto.shape[0] > 1:
            gram = proto @ proto.t()
            off_diag = gram - torch.diag_embed(torch.diag(gram))
            l_div = F.relu(off_diag).pow(2).sum() / (proto.shape[0] * (proto.shape[0] - 1))
            total = total + self.lambda_div * l_div

        metrics = {
            "l_vi": l_vi_mean.detach(),
            "l_vv": l_vv_mean.detach(),
            "l_div": l_div.detach(),
        }
        return total, metrics
