
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

import src.model.dino as DINO


from src.model.dinov2_feat import Dinov2FeatSeq
backbone = Dinov2FeatSeq(arch_name="vit_base", pretrained=True)

from src.model.discriminatorplus import Discriminator


class TrainableMemory(nn.Module):
    """
    Foreground/background trainable memory module.
    It enhances foreground features and suppresses background features while keeping the original interface.
    """

    def __init__(self, feat_dim, mem_size=2024, conf_thres=0.9, ema_alpha=0.99, device=None):
        super().__init__()
        self.mem_size = mem_size
        self.feat_dim = feat_dim
        self.conf_thres = conf_thres
        self.ema_alpha = ema_alpha
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.fg_memory = nn.Parameter(torch.empty(mem_size, feat_dim))
        nn.init.xavier_uniform_(self.fg_memory)

        self.bg_memory = nn.Parameter(torch.empty(mem_size, feat_dim))
        nn.init.xavier_uniform_(self.bg_memory)

        self.register_buffer("fg_conf", torch.zeros(mem_size))
        self.register_buffer("bg_conf", torch.zeros(mem_size))
        self.register_buffer("size", torch.zeros(1, dtype=torch.long))

        self.gate_fg = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 4, 1),
            nn.Sigmoid()
        )
        self.gate_bg = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, feats):
        """
        Args:
            feats: Feature map [B,C,H,W].
        Returns:
            Fused feature map [B,C,H,W].
        """
        B, C, H, W = feats.shape
        feats_flat = feats.permute(0, 2, 3, 1).reshape(-1, C)
        feats_norm = F.normalize(feats_flat, dim=1)

        fg_norm = F.normalize(self.fg_memory, dim=1)
        bg_norm = F.normalize(self.bg_memory, dim=1)

        sim_fg = torch.matmul(feats_norm, fg_norm.t())
        attn_fg = F.softmax(sim_fg / 0.03, dim=1)
        mem_fg = torch.matmul(attn_fg, fg_norm)

        sim_bg = torch.matmul(feats_norm, bg_norm.t())
        attn_bg = F.softmax(sim_bg / 0.03, dim=1)
        mem_bg = torch.matmul(attn_bg, bg_norm)

        g_fg = self.gate_fg(feats_norm)
        g_bg = self.gate_bg(feats_norm)

        fused = feats_norm + g_fg * mem_fg - g_bg * mem_bg

        fused_map = fused.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return fused_map

    @torch.no_grad()
    def update_memory(self, feats, confs):
        """
        Update foreground/background memories by confidence.

        Args:
            feats: Flattened features [N,C].
            confs: Confidence values [N] in [0,1].
        """
        if feats.numel() == 0 or confs.numel() == 0:
            return

        feats = feats.to(self.device)
        confs = confs.to(self.device)

        fg_mask = confs > self.conf_thres
        bg_mask = confs < (1 - self.conf_thres)

        fg_feats = F.normalize(feats[fg_mask], dim=1) if fg_mask.any() else None
        bg_feats = F.normalize(feats[bg_mask], dim=1) if bg_mask.any() else None

        cur_size = int(self.size.item())

        if cur_size < self.mem_size:
            if fg_feats is not None and fg_feats.shape[0] > 0:
                fill_size_fg = min(self.mem_size - cur_size, fg_feats.shape[0])
                self.fg_memory.data[cur_size:cur_size + fill_size_fg] = fg_feats[:fill_size_fg]
                self.fg_conf[cur_size:cur_size + fill_size_fg] = confs[fg_mask][:fill_size_fg]

            if bg_feats is not None and bg_feats.shape[0] > 0:
                fill_size_bg = min(self.mem_size - cur_size, bg_feats.shape[0])
                self.bg_memory.data[cur_size:cur_size + fill_size_bg] = bg_feats[:fill_size_bg]
                self.bg_conf[cur_size:cur_size + fill_size_bg] = confs[bg_mask][:fill_size_bg]

            self.size[0] = min(self.mem_size, cur_size + (
                max(
                    (fg_feats.shape[0] if fg_feats is not None else 0),
                    (bg_feats.shape[0] if bg_feats is not None else 0)
                )
            ))
            return

        if fg_feats is not None and fg_feats.shape[0] > 0:
            k = min(fg_feats.shape[0], self.mem_size)
            low_idx = torch.topk(self.fg_conf, k, largest=False).indices
            self.fg_memory.data[low_idx] = (
                    self.ema_alpha * self.fg_memory.data[low_idx] +
                    (1 - self.ema_alpha) * fg_feats[:k]
            )
            self.fg_conf[low_idx] = confs[fg_mask][:k]

        if bg_feats is not None and bg_feats.shape[0] > 0:
            k = min(bg_feats.shape[0], self.mem_size)
            low_idx = torch.topk(self.bg_conf, k, largest=False).indices
            self.bg_memory.data[low_idx] = (
                    self.ema_alpha * self.bg_memory.data[low_idx] +
                    (1 - self.ema_alpha) * bg_feats[:k]
            )
            self.bg_conf[low_idx] = confs[bg_mask][:k]


class MEMUCOD(nn.Module):
    def __init__(self, patch_size, train_size):
        super(MEMUCOD, self).__init__()
        self.sMod = backbone
        self.flatten = nn.Unflatten(2, torch.Size([train_size // patch_size, train_size // patch_size]))

        self.tMod = nn.Conv2d(in_channels=self.sMod.embed_dim, out_channels=1, kernel_size=1, padding=0)



        self.adaMod = Discriminator(train_size // patch_size)
        self.downSample = nn.Upsample(scale_factor=1 / patch_size, mode='bicubic')
        self.out_size = train_size
        self.sigmoid = nn.Sigmoid()

        self.memory = TrainableMemory(feat_dim=self.sMod.embed_dim, mem_size=2048)

        for p in self.sMod.parameters():
            p.requires_grad = False
        self.sMod.eval()

    @torch.no_grad()
    def extract_feats(self, x):
        """Extract frozen backbone features on the patch grid [B,C,H',W']."""
        feats = self.sMod(x)
        feats = self.flatten(feats)
        return feats

    def forward(self, x_c1, x_c2=None):
        if not x_c2 == None:
            with torch.no_grad():
                feats_c1 = self.sMod(x_c1)
            feats_c1 = self.flatten(feats_c1)

            with torch.no_grad():
                feats_c2 = self.sMod(x_c2)
            feats_c2 = self.flatten(feats_c2)

            feats_c1 = self.memory(feats_c1)
            feats_c2 = self.memory(feats_c2)

            y_c1 = self.tMod(feats_c1)
            y_c2 = self.tMod(feats_c2)

            fc1 = self.sigmoid(y_c1)
            bc1 = -1 * fc1 + 1
            fc2 = self.sigmoid(y_c2)
            bc2 = -1 * fc2 + 1

            def _disc_prob(din):
                out = self.adaMod(din)
                if isinstance(out, tuple):
                    _, pooled = out
                    return torch.sigmoid(pooled)
                else:
                    return out

            vld_fc1 = _disc_prob(torch.cat((self.downSample(x_c1), fc1), 1))
            vld_bc1 = _disc_prob(torch.cat((self.downSample(x_c1), bc1), 1))
            vld_fc2 = _disc_prob(torch.cat((self.downSample(x_c2), fc2), 1))
            vld_bc2 = _disc_prob(torch.cat((self.downSample(x_c2), bc2), 1))

            vld_fc = (vld_fc1 + vld_fc2) / 2
            vld_bc = (vld_bc1 + vld_bc2) / 2

            y_c1 = F.interpolate(y_c1, size=self.out_size, mode='bicubic', align_corners=False)
            y_c2 = F.interpolate(y_c2, size=self.out_size, mode='bicubic', align_corners=False)

            return y_c1, vld_fc, y_c2, vld_bc
        else:
            with torch.no_grad():
                feats = self.sMod(x_c1)
            feats = self.flatten(feats)
            feats = self.memory(feats)
            y = self.tMod(feats)
            y = F.interpolate(y, size=self.out_size, mode='bicubic', align_corners=False)

            return y
