
import torch
import torch.nn as nn
from dinov2.hub.backbones import _make_dinov2_model


class Dinov2FeatSeq(nn.Module):
    """
    DINOv2 feature wrapper compatible with the existing MEMUCOD interface.
    Forward returns [B, C, H*W], so the existing Unflatten(2, [H, W]) can be reused.
    """

    def __init__(self, arch_name="vit_base", pretrained=True):
        """
        arch_name options: "vit_small", "vit_base", "vit_large", "vit_giant".
        """
        super().__init__()
        self.model = _make_dinov2_model(arch_name=arch_name, pretrained=pretrained)
        self.patch_size = 14
        self.embed_dim = getattr(self.model, "embed_dim", None)
        if self.embed_dim is None:
            self.embed_dim = getattr(self.model, "num_features", None)
        assert self.embed_dim is not None, "Failed to get embed_dim from the DINOv2 model"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor [B,3,H,W]. H and W must be divisible by 14.
        Returns:
            Patch features [B, C, H'*W'], where H'=H/14 and W'=W/14.
        """
        B, _, Himg, Wimg = x.shape
        assert Himg % self.patch_size == 0 and Wimg % self.patch_size == 0, \
            f"Input resolution must be divisible by 14, got {Himg}x{Wimg}"

        x_tok = self.model.prepare_tokens(x)

        for blk in self.model.blocks:
            x_tok = blk(x_tok)

        x_tok = self.model.norm(x_tok)
        x_tok = x_tok[:, 1:, :]
        feats = x_tok.transpose(1, 2).contiguous()
        return feats
