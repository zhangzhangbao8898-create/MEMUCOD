

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from skimage.segmentation import random_walker as _random_walker
except Exception:
    _random_walker = None

from skimage.measure import label as sk_label
from skimage.morphology import dilation, erosion


EPS = 1e-12


def _minmax(x: np.ndarray) -> np.ndarray:
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)


def _center_prior(h: int, w: int, sigma_ratio: float = 0.35) -> np.ndarray:
    """2D Gaussian center prior normalized to [0,1]."""
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    sy, sx = max(h * sigma_ratio, 1.0), max(w * sigma_ratio, 1.0)
    g = np.exp(-((yy - cy) ** 2) / (2 * sy ** 2) - ((xx - cx) ** 2) / (2 * sx ** 2))
    return _minmax(g)


def _feature_edges(f_hw_c: np.ndarray) -> np.ndarray:
    """Feature-edge score from neighboring patch L2 differences."""
    h, w, c = f_hw_c.shape
    dh = np.linalg.norm(f_hw_c[:, 1:, :] - f_hw_c[:, :-1, :], axis=-1)
    dv = np.linalg.norm(f_hw_c[1:, :, :] - f_hw_c[:-1, :, :], axis=-1)
    edge = np.zeros((h, w), dtype=np.float32)
    edge[:, 1:] += dh
    edge[:, :-1] += dh
    edge[1:, :] += dv
    edge[:-1, :] += dv
    return _minmax(edge)


def _feature_contrast(f_hw_c: np.ndarray) -> np.ndarray:
    """Feature contrast as distance from the global feature mean."""
    mu = f_hw_c.reshape(-1, f_hw_c.shape[-1]).mean(axis=0, keepdims=True)
    dist = np.linalg.norm(
        f_hw_c.reshape(-1, f_hw_c.shape[-1]) - mu, axis=1
    ).reshape(f_hw_c.shape[:2])
    return _minmax(dist)


def _feature_objectness(f_hw_c: np.ndarray) -> np.ndarray:
    """Objectness from inverse similarity to the border-background mean."""
    h, w, c = f_hw_c.shape
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    bg_mean = f_hw_c[border].mean(axis=0, keepdims=True)
    sim = np.clip(np.sum(f_hw_c * bg_mean, axis=-1), -1.0, 1.0)
    obj = 1.0 - (sim + 1.0) * 0.5
    return _minmax(obj)


def _smooth_map(m: np.ndarray, k: int = 3, iters: int = 1) -> np.ndarray:
    """Light smoothing with avg_pool, avoiding extra dependencies."""
    if k < 3 or k % 2 == 0 or iters <= 0:
        return m.astype(np.float32)
    t = torch.from_numpy(m.astype(np.float32))[None, None]
    pad = k // 2
    for _ in range(iters):
        t = F.avg_pool2d(t, kernel_size=k, stride=1, padding=pad)
    return t.squeeze(0).squeeze(0).cpu().numpy()


def _postprocess(mask: np.ndarray, morph_erode: int, morph_open: int, keep_lcc: bool) -> np.ndarray:
    """Apply erosion, opening, and optional largest-connected-component filtering."""
    m = mask.astype(np.uint8)
    se = np.ones((3, 3), dtype=bool)
    for _ in range(max(morph_erode, 0)):
        m = erosion(m.astype(bool), se).astype(np.uint8)
    for _ in range(max(morph_open, 0)):
        m = erosion(m.astype(bool), se)
        m = dilation(m, se).astype(np.uint8)
    if keep_lcc and m.sum() > 0:
        lab = sk_label(m, connectivity=1)
        if lab.max() >= 1:
            areas = np.bincount(lab.ravel())[1:]
            keep = 1 + int(areas.argmax())
            m = (lab == keep).astype(np.uint8)
    return m


@torch.no_grad()
def prior_conf_pseudo(
    feats_bchw: torch.Tensor,
    out_size: Optional[Tuple[int, int]] = None,
    w_obj: float = 0.4,
    w_ctr: float = 0.2,
    w_edge: float = 0.2,
    w_cont: float = 0.2,
    tau_fg: float = 0.65,
    tau_bg: float = 0.35,
    use_random_walker: bool = True,
    rw_beta: float = 90.0,
    smooth_k: int = 3,
    smooth_iters: int = 1,
    morph_erode: int = 1,
    morph_open: int = 1,
    keep_lcc: bool = True,
    min_fg: float = 0.02,
    max_fg: float = 0.60,
) -> torch.Tensor:
    """
    Generate SOD pseudo labels from handcrafted priors, confidence seeds, and optional random walker.

    Args:
        feats_bchw: Feature tensor [B,C,Hf,Wf]. It is expected to be normalized outside.
        out_size: Optional output size (Hout,Wout). If None, keep Hf x Wf.
    Returns:
        Pseudo mask tensor [B,1,Hout,Wout] with values in {0.,1.}.
    """
    device = feats_bchw.device
    B, C, Hf, Wf = feats_bchw.shape

    ws = np.array([w_obj, w_ctr, w_edge, w_cont], dtype=np.float32)
    ws = ws / (ws.sum() + EPS)
    w_obj, w_ctr, w_edge, w_cont = ws.tolist()

    out_list = []
    for b in range(B):
        fb = feats_bchw[b].detach().cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(fb, axis=0, keepdims=True) + EPS
        f_hw_c = (fb / norm).transpose(1, 2, 0)

        prior_obj = _feature_objectness(f_hw_c)
        prior_ctr = _center_prior(Hf, Wf, sigma_ratio=0.35)
        prior_edge = _feature_edges(f_hw_c)
        prior_cont = _feature_contrast(f_hw_c)

        prior_obj = _smooth_map(prior_obj, k=smooth_k, iters=smooth_iters)
        prior_ctr = _smooth_map(prior_ctr, k=smooth_k, iters=smooth_iters)
        prior_edge = _smooth_map(prior_edge, k=smooth_k, iters=smooth_iters)
        prior_cont = _smooth_map(prior_cont, k=smooth_k, iters=smooth_iters)

        S = w_obj * prior_obj + w_ctr * prior_ctr + w_edge * prior_edge + w_cont * prior_cont
        S = _minmax(S)

        seed_fg = (S >= tau_fg)
        seed_bg = (S <= tau_bg)
        border = np.zeros((Hf, Wf), dtype=bool)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
        seed_bg = np.logical_or(seed_bg, border)

        tmp = (S >= 0.5).astype(np.uint8)
        fg_ratio = float(tmp.mean())
        if fg_ratio < min_fg or fg_ratio > max_fg:
            S = 1.0 - S
            seed_fg, seed_bg = seed_bg, seed_fg

        can_rw = bool(use_random_walker and (_random_walker is not None) and seed_fg.any() and seed_bg.any())
        if can_rw:
            labels = np.full((Hf, Wf), -1, dtype=np.int32)
            labels[seed_bg] = 0
            labels[seed_fg] = 1

            g = np.clip(S * (1.0 - prior_edge), 0.0, 1.0).astype(np.float32)

            try:
                rw = _random_walker(g, labels, beta=rw_beta, mode='bf')
            except TypeError:
                rw = _random_walker(g, labels, beta=rw_beta)

            mask = (rw == 1).astype(np.uint8)
        else:
            mask = (S > 0.5).astype(np.uint8)

        mask = _postprocess(mask, morph_erode=morph_erode, morph_open=morph_open, keep_lcc=keep_lcc)
        out_list.append(torch.from_numpy(mask)[None, None, ...].float())

    pseudo = torch.cat(out_list, dim=0).to(device)
    if out_size is not None:
        pseudo = F.interpolate(pseudo, size=out_size, mode="nearest")
    return pseudo
