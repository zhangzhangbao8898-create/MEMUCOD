import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import ObjDatasetTE
from model.memucod import MEMUCOD
from utils import print_network


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=560)
parser.add_argument('--patchsize', type=int, default=14)
opt = parser.parse_args()
assert opt.testsize % opt.patchsize == 0


model = MEMUCOD(opt.patchsize, opt.testsize)
print_network(model, 'MEMUCOD')
ckpt = torch.load("../checkpoints/memucod/MEMUCOD_10.pth")
all_params = {}
for k, v in model.state_dict().items():
    if 'module.' + k in ckpt.keys():
        all_params[k] = ckpt['module.' + k]
model.load_state_dict(all_params)
model.cuda()
model.eval()


dataset_path = "../data/test"
test_datasets = ['CHAMELEON', 'CAMOtest250']
save_root = "../outputs/memucod"
os.makedirs(save_root, exist_ok=True)


def post_process(mask, min_area=100, smooth=True):
    """
    Post-process a binary prediction mask.

    Steps:
      - Fill small holes with morphological closing.
      - Remove small noise with morphological opening.
      - Remove connected components smaller than min_area.
      - Optionally smooth mask boundaries.

    Args:
        mask: Input binary mask as np.uint8 in the range [0, 255].
        min_area: Minimum connected-component area to keep.
        smooth: Whether to apply Gaussian boundary smoothing.
    Returns:
        np.uint8 binary mask in the range [0, 255].
    """
    mask = (mask > 127).astype(np.uint8)


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:

        keep = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= min_area]
        if keep:
            mask = np.isin(labels, keep).astype(np.uint8)
        else:

            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = (labels == largest).astype(np.uint8)


    if smooth:
        mask = cv2.GaussianBlur(mask.astype(np.float32), (3, 3), 0)
        mask = (mask > 0.5).astype(np.uint8)

    return (mask * 255).astype(np.uint8)


for dataset in test_datasets:
    save_path = os.path.join(save_root, dataset)
    os.makedirs(save_path, exist_ok=True)
    print(f"\nProcessing dataset: {dataset}")
    loader = ObjDatasetTE(os.path.join(dataset_path, dataset, "Imgs/"), opt.testsize)
    for _ in tqdm(range(loader.size)):
        image, HH, WW, name = loader.load_data()
        image = image.cuda()
        with torch.no_grad():
            logits = model.forward(image)

        pred = F.interpolate(logits, size=[WW, HH], mode='bilinear', align_corners=False)
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).squeeze().float()
        pred_np = (255 * pred).to(torch.uint8).cpu().numpy()


        processed_mask = post_process(pred_np)


        cv2.imwrite(os.path.join(save_path, name), processed_mask)

print("Test done with post-processing!")
