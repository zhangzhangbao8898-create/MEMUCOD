import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from data import get_loader
from model.memucod import MEMUCOD
from model.prior_conf_pseudo import prior_conf_pseudo
from utils import AvgMeter, print_network


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _morph_dilate(mask, k=2):
    if k <= 0:
        return mask
    return F.max_pool2d(mask, kernel_size=2 * k + 1, stride=1, padding=k)


def _morph_erode(mask, k=2):
    if k <= 0:
        return mask
    return 1.0 - F.max_pool2d(1.0 - mask, kernel_size=2 * k + 1, stride=1, padding=k)


def _morph_open(mask, k=1):
    if k <= 0:
        return mask
    return _morph_dilate(_morph_erode(mask, k), k)


@torch.no_grad()
def make_trimap_weights(pseudo_bin, erode_k=2, ring_k=2, w_core=1.0, w_unknown=0.3, extra_unknown=None):
    assert pseudo_bin.ndim == 4 and pseudo_bin.shape[1] == 1, f"pseudo_bin shape={pseudo_bin.shape}, expected [B,1,H,W]"
    pseudo = pseudo_bin.float().clamp(0, 1)
    core_fg = _morph_erode(pseudo, k=erode_k)
    core_bg = _morph_erode(1.0 - pseudo, k=erode_k)
    bd = (_morph_dilate(pseudo, k=ring_k) - _morph_erode(pseudo, k=ring_k)).clamp(0, 1)
    unknown = (bd > 0).float()
    if extra_unknown is not None:
        unknown = torch.clamp(unknown + extra_unknown.float(), 0, 1)
    weights = w_core * core_fg + w_core * core_bg + w_unknown * unknown
    return weights.clamp(0, 1), unknown


@torch.no_grad()
def pseudo_from_logits(logits, tau_fg=0.65, tau_bg=0.35, morph_open_k=1):
    assert logits.ndim == 4 and logits.shape[1] == 1, f"logits shape={logits.shape}, expected [B,1,H,W]"
    prob = torch.sigmoid(logits)
    hard = (prob >= 0.5).float()
    if morph_open_k > 0:
        hard = _morph_open(hard, k=morph_open_k).clamp(0, 1)
    uncertain = ((prob > tau_bg) & (prob < tau_fg)).float()
    return hard, uncertain


@torch.no_grad()
def soft_bootstrap_target(logits, hard_pseudo, unknown_mask, beta=0.8):
    assert logits.shape == hard_pseudo.shape == unknown_mask.shape, "soft bootstrap shapes do not match"
    hard = hard_pseudo.float()
    prob_detach = torch.sigmoid(logits).detach()
    soft = beta * hard + (1 - beta) * prob_detach
    return torch.where(unknown_mask > 0, soft, hard)


def weighted_bce_dice(logits, target, weights, dice_weight=1.0, eps=1e-6):
    assert logits.shape == target.shape == weights.shape, "weighted loss input shapes do not match"
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    num = weights.sum() + eps
    bce = (bce * weights).sum() / num
    prob = torch.sigmoid(logits)
    inter = (prob * target * weights).sum()
    denom = (prob * weights).sum() + (target * weights).sum() + eps
    dice = 1.0 - (2.0 * inter + eps) / denom
    return bce + dice_weight * dice


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=5e-3)
parser.add_argument('--batchsize', type=int, default=4)
parser.add_argument('--trainsize', type=int, default=560)
parser.add_argument('--patchsize', type=int, default=14)
parser.add_argument('--reg_weight', type=float, default=1.0)
parser.add_argument('--fuse', type=str, default="gated", choices=["gated", "mix"])
parser.add_argument('--gamma', type=float, default=0.6, help='post ratio when using mix fusion')
opt = parser.parse_args()


model = MEMUCOD(opt.patchsize, opt.trainsize)
print_network(model, 'MEMUCOD')
model = torch.nn.DataParallel(model)
model.to(device)

if hasattr(model, "module"):
    model.module.sMod.eval()
else:
    model.sMod.eval()

segModParams, adaModParams = [], []
for name, p in model.named_parameters():
    if 'adaMod' in name:
        adaModParams.append(p)
    else:
        segModParams.append(p)
optimizer = torch.optim.Adam([{'params': segModParams}, {'params': adaModParams}], opt.lr)

adv_loss = torch.nn.BCEWithLogitsLoss()

salient_image_root = "../data/train/COD600/Imgs"
salient_pseudo_gt_root = "../data/train/COD600/GT"

camouflage_image_root = "../data/train/SOD600/Imgs"
camouflage_pseudo_gt_root = "../data/train/SOD600/GT"
save_path = '../checkpoints/memucod/'
os.makedirs(save_path, exist_ok=True)

train_loader = get_loader(
    salient_image_root,
    salient_pseudo_gt_root,
    camouflage_image_root,
    camouflage_pseudo_gt_root,
    batchsize=opt.batchsize,
    trainsize=opt.trainsize,
)
total_step = len(train_loader)

logging.basicConfig(
    filename=os.path.join(save_path, 'log.log'),
    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
    level=logging.INFO,
    filemode='a',
    datefmt='%Y-%m-%d %I:%M:%S %p',
)
logging.info("MEMUCOD-Train (Dual-Fuse)")


def TRAIN(train_loader, model, optimizer, epoch, save_path):
    optimizer.param_groups[0]['lr'] = opt.lr * 0.2 ** (epoch - 1)
    optimizer.param_groups[1]['lr'] = opt.lr * 0.2 ** (epoch - 1) * 0.1

    model.train()
    total_loss_m = AvgMeter()
    segC1_m = AvgMeter()
    segC2_m = AvgMeter()
    advCF_m = AvgMeter()
    advCB_m = AvgMeter()

    for i, pack in enumerate(train_loader, 1):
        optimizer.zero_grad()

        imgsC1, _, imgsC2, _ = pack
        imgsC1 = Variable(imgsC1).to(device)
        imgsC2 = Variable(imgsC2).to(device)

        preds_c1, prob_cf, preds_c2, prob_cb = model(imgsC1, imgsC2)

        with torch.no_grad():
            target_model = model.module if hasattr(model, "module") else model
            conf_map = torch.sigmoid(preds_c1).detach()
            feats = target_model.flatten(target_model.sMod(imgsC1))
            conf_patch = F.adaptive_avg_pool2d(conf_map, (feats.shape[2], feats.shape[3]))
            conf_patch = conf_patch.flatten()
            feats = feats.permute(0, 2, 3, 1).reshape(-1, feats.shape[1])
            target_model.memory.update_memory(feats, conf_patch)

        assert preds_c1.ndim == 4 and preds_c1.shape[1] == 1 and preds_c2.shape == preds_c1.shape, "invalid preds_* shape"
        S = preds_c1.shape[-1]

        with torch.no_grad():
            extract = getattr(model, "extract_feats", None) or getattr(model.module, "extract_feats")
            f1 = F.normalize(extract(imgsC1), dim=1)
            f2 = F.normalize(extract(imgsC2), dim=1)

            pseudo_pre1 = prior_conf_pseudo(
                f1,
                out_size=(S, S),
                w_obj=0.45,
                w_ctr=0.15,
                w_edge=0.20,
                w_cont=0.20,
                tau_fg=0.65,
                tau_bg=0.35,
                use_random_walker=True,
                rw_beta=90.0,
                smooth_k=3,
                smooth_iters=1,
                morph_erode=1,
                morph_open=1,
                keep_lcc=True,
                min_fg=0.02,
                max_fg=0.60,
            )
            pseudo_pre2 = prior_conf_pseudo(
                f2,
                out_size=(S, S),
                w_obj=0.45,
                w_ctr=0.15,
                w_edge=0.20,
                w_cont=0.20,
                tau_fg=0.65,
                tau_bg=0.35,
                use_random_walker=True,
                rw_beta=90.0,
                smooth_k=3,
                smooth_iters=1,
                morph_erode=1,
                morph_open=1,
                keep_lcc=True,
                min_fg=0.02,
                max_fg=0.60,
            )

        pseudo_post1, uncertain1 = pseudo_from_logits(preds_c1, tau_fg=0.65, tau_bg=0.35, morph_open_k=1)
        pseudo_post2, uncertain2 = pseudo_from_logits(preds_c2, tau_fg=0.65, tau_bg=0.35, morph_open_k=1)

        if opt.fuse == "gated":
            hard1 = torch.where(uncertain1 > 0, pseudo_pre1, pseudo_post1).float()
            hard2 = torch.where(uncertain2 > 0, pseudo_pre2, pseudo_post2).float()
        else:
            hard1 = (opt.gamma * pseudo_post1 + (1 - opt.gamma) * pseudo_pre1).clamp(0, 1)
            hard2 = (opt.gamma * pseudo_post2 + (1 - opt.gamma) * pseudo_pre2).clamp(0, 1)
            hard1 = (hard1 >= 0.5).float()
            hard2 = (hard2 >= 0.5).float()

        from pseudo_refine_combo_joint import refine_pseudo_combo_joint
        hard1 = refine_pseudo_combo_joint(
            hard1,
            imgsC1,
            iters=2,
            hp_sigma=0.35,
            gabor_ks=9,
            thetas=12,
            L=7,
            w_ridge=0.6,
            q=0.85,
        )
        hard2 = refine_pseudo_combo_joint(
            hard2,
            imgsC2,
            iters=2,
            hp_sigma=0.35,
            gabor_ks=9,
            thetas=12,
            L=7,
            w_ridge=0.6,
            q=0.85,
        )

        w1, u1 = make_trimap_weights(hard1, erode_k=2, ring_k=2, w_core=1.0, w_unknown=0.3, extra_unknown=uncertain1)
        w2, u2 = make_trimap_weights(hard2, erode_k=2, ring_k=2, w_core=1.0, w_unknown=0.3, extra_unknown=uncertain2)

        t1 = soft_bootstrap_target(preds_c1, hard1, u1, beta=0.8)
        t2 = soft_bootstrap_target(preds_c2, hard2, u2, beta=0.8)

        segC1_loss = weighted_bce_dice(preds_c1, t1, w1, dice_weight=1.0)
        segC2_loss = weighted_bce_dice(preds_c2, t2, w2, dice_weight=1.0)

        B = preds_c1.size(0)
        advCF_loss = adv_loss(prob_cf, torch.zeros(B, 1, device=device))
        advCB_loss = adv_loss(prob_cb, torch.ones(B, 1, device=device))

        total_loss = segC1_loss + opt.reg_weight * advCF_loss + segC2_loss + opt.reg_weight * advCB_loss
        total_loss.backward()
        optimizer.step()

        total_loss_m.update(total_loss.data, opt.batchsize)
        segC1_m.update(segC1_loss.data, opt.batchsize)
        segC2_m.update(segC2_loss.data, opt.batchsize)
        advCF_m.update(advCF_loss.data, opt.batchsize)
        advCB_m.update(advCB_loss.data, opt.batchsize)

        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}] Step [{:04d}/{:04d}] | Total:{:.4f} | Seg1:{:.4f} Seg2:{:.4f} | AdvF:{:.4f} AdvB:{:.4f}'
                  .format(datetime.now(), epoch, opt.epochs, i, total_step,
                          total_loss_m.show(), segC1_m.show(), segC2_m.show(), advCF_m.show(), advCB_m.show()))
            logging.info('Epoch [{:03d}/{:03d}] Step [{:04d}/{:04d}] | Total:{:.4f} | Seg1:{:.4f} Seg2:{:.4f} | AdvF:{:.4f} AdvB:{:.4f}'
                         .format(epoch, opt.epochs, i, total_step,
                                 total_loss_m.show(), segC1_m.show(), segC2_m.show(), advCF_m.show(), advCB_m.show()))

    torch.save(model.state_dict(), os.path.join(save_path, f'MEMUCOD_{epoch}.pth'))


if __name__ == '__main__':
    print("Let's go! (Dual-Fuse)")
    print(f"=> Device: {device} | Fusion mode: {opt.fuse}")
    for epoch in range(1, opt.epochs + 1):
        TRAIN(train_loader, model, optimizer, epoch, save_path)
    print("Training Done!")
