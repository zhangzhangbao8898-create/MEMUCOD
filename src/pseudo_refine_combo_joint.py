
import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def _to_gray(imgs):
    if imgs.shape[1] == 1:
        x = imgs.float()
    else:
        r, g, b = imgs[:, 0:1], imgs[:, 1:2], imgs[:, 2:3]
        x = (0.2989 * r + 0.5870 * g + 0.1140 * b).float()
    if x.max() > 1.5:
        x = x / 255.0
    return x.clamp(0, 1)


def _morph_dilate(mask, k=1):
    if k <= 0:
        return mask
    return F.max_pool2d(mask, kernel_size=2 * k + 1, stride=1, padding=k)


def _get_boundary(mask, ring=1):
    d = _morph_dilate(mask, k=ring)
    return (d - mask).clamp(0, 1)


@torch.no_grad()
def _fft_highpass_map(gray, hp_sigma=0.35):
    B, _, H, W = gray.shape
    X = torch.fft.fft2(gray, norm='ortho')
    X = torch.fft.fftshift(X)
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=gray.device),
        torch.linspace(-1.0, 1.0, W, device=gray.device),
        indexing='ij'
    )
    rr2 = xx ** 2 + yy ** 2
    sigma2 = (hp_sigma ** 2) + 1e-8
    low = torch.exp(-rr2 / (2.0 * sigma2))
    high = (1.0 - low).unsqueeze(0).unsqueeze(0)
    Xh = X * high
    Xh = torch.fft.ifftshift(Xh)
    detail = torch.fft.ifft2(Xh, norm='ortho').real
    mn = detail.amin(dim=(-2, -1), keepdim=True)
    mx = detail.amax(dim=(-2, -1), keepdim=True)
    return ((detail - mn) / (mx - mn + 1e-6)).clamp(0, 1)


def _make_gabor_bank(ksize=9, sigma=2.5, lam=6.0, gamma=0.5, thetas=8, device='cpu', dtype=torch.float32):
    assert ksize % 2 == 1
    half = ksize // 2
    y, x = torch.meshgrid(
        torch.arange(-half, half + 1, device=device, dtype=dtype),
        torch.arange(-half, half + 1, device=device, dtype=dtype),
        indexing='ij'
    )
    bank = []
    for t in range(thetas):
        theta = math.pi * t / thetas
        xt = x * math.cos(theta) + y * math.sin(theta)
        yt = -x * math.sin(theta) + y * math.cos(theta)
        gauss = torch.exp(-(xt ** 2 + (gamma ** 2) * (yt ** 2)) / (2.0 * (sigma ** 2)))
        carrier = torch.cos(2.0 * math.pi * xt / lam)
        g = gauss * carrier
        g = g - g.mean()
        g = g / (g.abs().sum() + 1e-6)
        bank.append(g.unsqueeze(0).unsqueeze(0))
    return torch.cat(bank, dim=0)


@torch.no_grad()
def _ridge_map_with_gabor(detail, ksize=9, sigma=2.5, lam=6.0, gamma=0.5, thetas=8):
    bank = _make_gabor_bank(ksize, sigma, lam, gamma, thetas, device=detail.device, dtype=detail.dtype)
    resp = F.conv2d(detail, bank, padding=ksize // 2)
    max_resp = resp.abs().max(dim=1, keepdim=True)[0]
    mn = max_resp.amin(dim=(-2, -1), keepdim=True)
    mx = max_resp.amax(dim=(-2, -1), keepdim=True)
    return ((max_resp - mn) / (mx - mn + 1e-6)).clamp(0, 1)


@torch.no_grad()
def _sobel_kernels(device, dtype):
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    return kx, ky


@torch.no_grad()
def _gaussian_kernel(ks=7, sigma=2.0, device='cpu', dtype=torch.float32):
    assert ks % 2 == 1
    half = ks // 2
    y, x = torch.meshgrid(
        torch.arange(-half, half + 1, device=device, dtype=dtype),
        torch.arange(-half, half + 1, device=device, dtype=dtype),
        indexing='ij'
    )
    g = torch.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    return g.view(1, 1, ks, ks)


@torch.no_grad()
def _structure_tensor(gray_s, blur_ks=7, blur_sigma=2.0):
    kx, ky = _sobel_kernels(gray_s.device, gray_s.dtype)
    Ix = F.conv2d(gray_s, kx, padding=1)
    Iy = F.conv2d(gray_s, ky, padding=1)
    J11, J12, J22 = Ix * Ix, Ix * Iy, Iy * Iy
    g = _gaussian_kernel(blur_ks, blur_sigma, device=gray_s.device, dtype=gray_s.dtype)
    J11 = F.conv2d(J11, g, padding=blur_ks // 2)
    J12 = F.conv2d(J12, g, padding=blur_ks // 2)
    J22 = F.conv2d(J22, g, padding=blur_ks // 2)
    tmp = torch.sqrt((J11 - J22) ** 2 + 4 * J12 ** 2 + 1e-12)
    lam1 = 0.5 * (J11 + J22 + tmp)
    lam2 = 0.5 * (J11 + J22 - tmp)
    theta = 0.5 * torch.atan2(2 * J12, (J11 - J22 + 1e-12))
    coherence = (lam1 - lam2) / (lam1 + lam2 + 1e-12)
    return theta, coherence.clamp(0, 1)


@torch.no_grad()
def _make_line_bank(L=7, thetas=12, device='cpu', dtype=torch.float32):
    assert L % 2 == 1
    half = L // 2
    y, x = torch.meshgrid(
        torch.arange(-half, half + 1, device=device, dtype=dtype),
        torch.arange(-half, half + 1, device=device, dtype=dtype),
        indexing='ij'
    )
    bank = []
    for t in range(thetas):
        theta = math.pi * t / thetas
        xt = x * math.cos(theta) + y * math.sin(theta)
        yt = -x * math.sin(theta) + y * math.cos(theta)
        line = (yt.abs() <= 0.5).float()
        line = line / (line.sum() + 1e-6)
        bank.append(line.unsqueeze(0).unsqueeze(0))
    return torch.cat(bank, dim=0)


@torch.no_grad()
def refine_pseudo_combo_joint(
    hard, imgs,
    iters=2,
    hp_sigma=0.35, gabor_ks=9,
    thetas=12, L=7,
    w_ridge=0.6, q=0.85
):
    """
    Refine a hard pseudo mask with frequency detail, ridge response, and directional growth.

    Args:
        hard: Hard mask [B,1,S,S] in {0,1}.
        imgs: Image tensor [B,3,H0,W0] or [B,1,H0,W0].
    Returns:
        Refined hard mask [B,1,S,S].
    """
    assert hard.ndim == 4 and hard.shape[1] == 1
    B, _, S, _ = hard.shape

    gray = _to_gray(imgs)
    gray_s = F.interpolate(gray, size=(S, S), mode='bilinear', align_corners=False)

    detail = _fft_highpass_map(gray_s, hp_sigma=hp_sigma)
    ridge = _ridge_map_with_gabor(detail, ksize=gabor_ks)

    theta, coh = _structure_tensor(gray_s, blur_ks=7, blur_sigma=2.0)
    theta_n = (theta + (math.pi / 2)) / math.pi
    ori_idx = (theta_n * thetas).long().clamp(0, thetas - 1)
    bank = _make_line_bank(L=L, thetas=thetas, device=imgs.device, dtype=torch.float32)

    m = hard.clone().float()

    for _ in range(max(1, iters)):
        bd = (_morph_dilate(m, k=1) - m).clamp(0, 1)
        if bd.sum() < 0.5:
            break

        hit = F.conv2d(m, bank, padding=L // 2)
        sel = torch.gather(hit, dim=1, index=ori_idx)

        sel_mn = sel.amin(dim=(-2, -1), keepdim=True)
        sel_mx = sel.amax(dim=(-2, -1), keepdim=True)
        sel_n = (sel - sel_mn) / (sel_mx - sel_mn + 1e-6)

        score = w_ridge * ridge + (1.0 - w_ridge) * torch.sqrt((coh.clamp(0, 1) * sel_n).clamp(0, 1))
        score = score * bd * (F.max_pool2d(m, 3, 1, 1) > 0).float()

        grow = torch.zeros_like(m)
        for b in range(B):
            sb = score[b, 0]
            rb = bd[b, 0] > 0
            if rb.sum() == 0:
                continue
            th = torch.quantile(sb[rb], q)
            grow[b, 0] = (sb >= th).float() * bd[b, 0]

        m = (m + (grow > 0).float()).clamp(0, 1)

    return (m >= 0.5).float()
