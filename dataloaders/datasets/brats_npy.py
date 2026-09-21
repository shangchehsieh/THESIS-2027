"""Dataset for the pre-processed BraTS2020 `.npy` layout shipped in
`BRATS2020_Training_none_npy/`:

    <root>/vol/<id>_vol.npy   float32  [H, W, D, 4]  (already z-score normalised)
    <root>/seg/<id>_seg.npy   uint8    [H, W, D]     labels {0, 1, 2, 3}  (ET == 3)
    <root>/{train,val,test}.txt          one case id per line

This mirrors the crop / flip augmentation of `brats_acn.Brats2018` but reads
the stacked npy volumes instead of per-modality `.nii.gz` files and does **not**
re-normalise (the arrays are already normalised).
"""
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


def read_split(root, split):
    """Return the list of case ids in `<root>/<split>.txt` that actually have
    an `_vol.npy` file on disk."""
    txt = os.path.join(root, f"{split}.txt")
    with open(txt) as fh:
        ids = [line.strip() for line in fh if line.strip()]
    vol_dir = os.path.join(root, "vol")
    ids = [i for i in ids if os.path.isfile(os.path.join(vol_dir, i + "_vol.npy"))]
    return ids


def make_split_lists(root, limit=-1):
    train_ids = read_split(root, "train")
    val_ids = read_split(root, "val")
    test_ids = read_split(root, "test")
    if limit and limit > 0:
        train_ids = train_ids[:limit]
        val_ids = val_ids[:limit]
        test_ids = test_ids[:limit]
    return train_ids, val_ids, test_ids


class BratsNpy(Dataset):

    def __init__(self, root, ids, crop_size, train=True):
        self.root = root
        self.ids = ids
        self.crop_size = tuple(int(c) for c in crop_size)
        self.train = train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        pid = self.ids[index]
        vol = np.load(os.path.join(self.root, "vol", pid + "_vol.npy"))   # [H, W, D, 4]
        seg = np.load(os.path.join(self.root, "seg", pid + "_seg.npy"))   # [H, W, D]

        x = np.ascontiguousarray(np.transpose(vol, (3, 0, 1, 2)), dtype="float32")  # [4, H, W, D]
        y = seg[np.newaxis].astype("int64")                                         # [1, H, W, D]

        x, y = self.aug_sample(x, y)

        seg = y[0]
        wt = seg > 0
        tc = (seg == 1) | (seg == 3) | (seg == 4)
        et = (seg == 3) | (seg == 4)
        label = np.stack([wt, tc, et], axis=0).astype("float32")   # [3, H, W, D]

        return {
            "image": torch.as_tensor(x.copy(), dtype=torch.float),
            "label": torch.as_tensor(label.copy(), dtype=torch.float),
        }

    def aug_sample(self, x, y):
        if self.train:
            x, y = self._crop(x, y, random_crop=True)
            for axis in (1, 2, 3):
                if random.random() < 0.5:
                    x = np.flip(x, axis=axis)
                    y = np.flip(y, axis=axis)
        else:
            x, y = self._crop(x, y, random_crop=False)
        return np.ascontiguousarray(x), np.ascontiguousarray(y)

    def _crop(self, x, y, random_crop):
        cs = self.crop_size
        h, w, d = x.shape[-3:]

        # zero-pad any axis that is smaller than the requested crop
        ph, pw, pd = max(cs[0] - h, 0), max(cs[1] - w, 0), max(cs[2] - d, 0)
        if ph or pw or pd:
            pad = ((0, 0), (0, ph), (0, pw), (0, pd))
            x = np.pad(x, pad, mode="constant")
            y = np.pad(y, pad, mode="constant")
            h, w, d = x.shape[-3:]

        if random_crop:
            sx = random.randint(0, h - cs[0])
            sy = random.randint(0, w - cs[1])
            sz = random.randint(0, d - cs[2])
        else:
            sx, sy, sz = (h - cs[0]) // 2, (w - cs[1]) // 2, (d - cs[2]) // 2

        x = x[:, sx:sx + cs[0], sy:sy + cs[1], sz:sz + cs[2]]
        y = y[:, sx:sx + cs[0], sy:sy + cs[1], sz:sz + cs[2]]
        return x, y
