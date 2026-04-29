
import os
from typing import List, Tuple
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms


IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
GT_EXTS  = ('.png', '.jpg', '.jpeg', '.bmp')

def _is_ext(name: str, exts: Tuple[str, ...]) -> bool:
    n = name.lower()
    return any(n.endswith(e) for e in exts)

def _list_files(root: str, exts: Tuple[str, ...]) -> List[str]:
    """List one-level files under root, without recursion, sorted by file name."""
    assert os.path.isdir(root), f"[Err] Directory does not exist: {root}"
    files = []
    for f in os.listdir(root):
        p = os.path.join(root, f)
        if os.path.isfile(p) and _is_ext(f, exts):
            files.append(p)
    return sorted(files)

def _stem(p: str) -> str:
    """Return the base file name without extension."""
    return os.path.splitext(os.path.basename(p))[0]


class ObjDataset(data.Dataset):
    def __init__(self,
                 image_root_c1: str, pseudoGT_root_c1: str,
                 image_root_c2: str, pseudoGT_root_c2: str,
                 trainsize: int):
        self.trainsize = trainsize


        raw_images_c1     = _list_files(image_root_c1, IMG_EXTS)
        raw_pseudo_gts_c1 = _list_files(pseudoGT_root_c1, GT_EXTS)
        raw_images_c2     = _list_files(image_root_c2, IMG_EXTS)
        raw_pseudo_gts_c2 = _list_files(pseudoGT_root_c2, GT_EXTS)


        c1_img_map = {_stem(p): p for p in raw_images_c1}
        c1_gt_map  = {_stem(p): p for p in raw_pseudo_gts_c1}
        c2_img_map = {_stem(p): p for p in raw_images_c2}
        c2_gt_map  = {_stem(p): p for p in raw_pseudo_gts_c2}

        c1_names = sorted(set(c1_img_map) & set(c1_gt_map))
        c2_names = sorted(set(c2_img_map) & set(c2_gt_map))

        miss_c1_img = sorted(set(c1_gt_map) - set(c1_img_map))
        miss_c1_gt  = sorted(set(c1_img_map) - set(c1_gt_map))
        miss_c2_img = sorted(set(c2_gt_map) - set(c2_img_map))
        miss_c2_gt  = sorted(set(c2_img_map) - set(c2_gt_map))
        if miss_c1_img: print(f"[Warn][C1] {len(miss_c1_img)} masks have no matching image, e.g. {miss_c1_img[:5]}")
        if miss_c1_gt:  print(f"[Warn][C1] {len(miss_c1_gt)} images have no matching mask, e.g. {miss_c1_gt[:5]}")
        if miss_c2_img: print(f"[Warn][C2] {len(miss_c2_img)} masks have no matching image, e.g. {miss_c2_img[:5]}")
        if miss_c2_gt:  print(f"[Warn][C2] {len(miss_c2_gt)} images have no matching mask, e.g. {miss_c2_gt[:5]}")

        self.images_c1     = [c1_img_map[n] for n in c1_names]
        self.pseudo_gts_c1 = [c1_gt_map[n]  for n in c1_names]
        self.images_c2     = [c2_img_map[n] for n in c2_names]
        self.pseudo_gts_c2 = [c2_gt_map[n]  for n in c2_names]


        if len(self.images_c1) != len(self.images_c2):
            m = min(len(self.images_c1), len(self.images_c2))
            print(f"[Warn] Domain sample counts differ; truncating both to {m}")
            self.images_c1, self.pseudo_gts_c1 = self.images_c1[:m], self.pseudo_gts_c1[:m]
            self.images_c2, self.pseudo_gts_c2 = self.images_c2[:m], self.pseudo_gts_c2[:m]

        self.size = len(self.images_c1)
        assert self.size > 0, "[Err] No valid image-mask pairs found; check file stems and extensions"


        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        image_c1 = self._rgb_loader(self.images_c1[index])
        pgt_c1   = self._binary_loader(self.pseudo_gts_c1[index])
        image_c2 = self._rgb_loader(self.images_c2[index])
        pgt_c2   = self._binary_loader(self.pseudo_gts_c2[index])

        image_c1 = self.img_transform(image_c1)
        pgt_c1   = self.gt_transform(pgt_c1)
        image_c2 = self.img_transform(image_c2)
        pgt_c2   = self.gt_transform(pgt_c2)

        return image_c1, pgt_c1, image_c2, pgt_c2

    @staticmethod
    def _rgb_loader(path: str) -> Image.Image:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    @staticmethod
    def _binary_loader(path: str) -> Image.Image:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

def get_loader(image_root_cls_1: str, pseudoGT_root_cls_1: str,
               image_root_cls_2: str, pseudoGT_root_cls_2: str,
               batchsize: int, trainsize: int,
               shuffle: bool = True, num_workers: int = 12, pin_memory: bool = True) -> torch.utils.data.DataLoader:
    dataset = ObjDataset(image_root_cls_1, pseudoGT_root_cls_1,
                         image_root_cls_2, pseudoGT_root_cls_2,
                         trainsize)
    loader = data.DataLoader(dataset=dataset,
                             batch_size=batchsize,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return loader


class ObjDatasetTE:
    def __init__(self, image_root: str, testsize: int):
        self.testsize = testsize
        self.images = _list_files(image_root, IMG_EXTS)
        self.images = sorted(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.size = len(self.images)
        assert self.size > 0, f"[Err] Test image directory is empty: {image_root}"
        self.index = 0

    def __len__(self):
        return self.size

    def rgb_loader(self, path: str) -> Image.Image:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        HH = image.size[0]
        WW = image.size[1]
        image_t = self.transform(image).unsqueeze(0)
        name = os.path.basename(self.images[self.index])
        if name.lower().endswith('.jpg'):
            name = name[: -4] + '.png'

        self.index = (self.index + 1) % self.size
        return image_t, HH, WW, name


class test_in_train:
    def __init__(self, image_root_c1: str, pseudoGT_root_c1: str,
                 image_root_c2: str, pseudoGT_root_c2: str,
                 valsize: int):
        self.valsize = valsize

        raw_images_c1     = _list_files(image_root_c1, IMG_EXTS)
        raw_pseudo_gts_c1 = _list_files(pseudoGT_root_c1, GT_EXTS)
        raw_images_c2     = _list_files(image_root_c2, IMG_EXTS)
        raw_pseudo_gts_c2 = _list_files(pseudoGT_root_c2, GT_EXTS)

        c1_img_map = {_stem(p): p for p in raw_images_c1}
        c1_gt_map  = {_stem(p): p for p in raw_pseudo_gts_c1}
        c2_img_map = {_stem(p): p for p in raw_images_c2}
        c2_gt_map  = {_stem(p): p for p in raw_pseudo_gts_c2}

        c1_names = sorted(set(c1_img_map) & set(c1_gt_map))
        c2_names = sorted(set(c2_img_map) & set(c2_gt_map))

        self.images_c1     = [c1_img_map[n] for n in c1_names]
        self.pseudo_gts_c1 = [c1_gt_map[n]  for n in c1_names]
        self.images_c2     = [c2_img_map[n] for n in c2_names]
        self.pseudo_gts_c2 = [c2_gt_map[n]  for n in c2_names]

        if len(self.images_c1) != len(self.images_c2):
            m = min(len(self.images_c1), len(self.images_c2))
            print(f"[Warn][VAL] Domain sample counts differ; truncating both to {m}")
            self.images_c1, self.pseudo_gts_c1 = self.images_c1[:m], self.pseudo_gts_c1[:m]
            self.images_c2, self.pseudo_gts_c2 = self.images_c2[:m], self.pseudo_gts_c2[:m]

        self.size = len(self.images_c1)
        assert self.size > 0, "[Err][VAL] No valid image-mask pairs found"

        self.img_transform = transforms.Compose([
            transforms.Resize((self.valsize, self.valsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()

        self.index = 0

    def __len__(self):
        return self.size

    @staticmethod
    def rgb_loader(path: str) -> Image.Image:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    @staticmethod
    def binary_loader(path: str) -> Image.Image:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def load_data(self):

        image_c1 = self.rgb_loader(self.images_c1[self.index])
        pgt_c1   = self.binary_loader(self.pseudo_gts_c1[self.index])
        HH_c1, WW_c1 = pgt_c1.size[1], pgt_c1.size[0]
        image_c1_t = self.img_transform(image_c1).unsqueeze(0)
        name_c1 = os.path.basename(self.images_c1[self.index])
        if name_c1.lower().endswith('.jpg'):
            name_c1 = name_c1[: -4] + '.png'


        image_c2 = self.rgb_loader(self.images_c2[self.index])
        pgt_c2   = self.binary_loader(self.pseudo_gts_c2[self.index])
        HH_c2, WW_c2 = pgt_c2.size[1], pgt_c2.size[0]
        image_c2_t = self.img_transform(image_c2).unsqueeze(0)
        name_c2 = os.path.basename(self.images_c2[self.index])
        if name_c2.lower().endswith('.jpg'):
            name_c2 = name_c2[: -4] + '.png'

        self.index = (self.index + 1) % self.size


        return image_c1_t, pgt_c1, name_c1, HH_c1, WW_c1, \
               image_c2_t, pgt_c2, name_c2, HH_c2, WW_c2
