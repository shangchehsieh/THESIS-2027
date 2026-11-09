"""Pre-process raw BraTS2020 cases into the npy layout of BRATS2020_Training_none_npy/.

    python tools/preprocess_brats_case.py <case_dir> [<case_dir> ...] [--out DIR]

`sup_128`, `crop` and `normalize` are copied verbatim from
~/PAMSRecon/PAMRecon/preprocess.py (the RFNet-style script that produced the
existing npy data), and the per-case steps below follow its main loop exactly:
stack (flair, t1ce, t1, t2) -> crop to the non-zero bounding box (each axis at
least 128) -> z-score each modality over brain voxels -> label 4 -> 3. That
script cannot be imported (it runs on import with hard-coded /root paths), and
it expects images/ and labels/ sub-folders, so this wrapper takes one case
folder holding all five files (`<id>_<mode>.nii` or `.nii.gz`) instead.
"""
import argparse
import glob
import os

import nibabel as nib
import numpy as np

join = os.path.join
OUT = join(os.path.dirname(os.path.abspath(__file__)), "..", "BRATS2020_Training_none_npy")


# ---- verbatim from PAMSRecon/PAMRecon/preprocess.py ----
def sup_128(xmin, xmax):
    if xmax - xmin < 128:
        print('#' * 100)
        ecart = int((128-(xmax-xmin))/2)
        xmax = xmax+ecart+1
        xmin = xmin-ecart
    if xmin < 0:
        xmax-=xmin
        xmin=0
    return xmin, xmax

def crop(vol):
    if len(vol.shape) == 4:
        vol = np.amax(vol, axis=0)
    assert len(vol.shape) == 3
    x_dim, y_dim, z_dim = tuple(vol.shape)
    x_nonzeros, y_nonzeros, z_nonzeros = np.where(vol != 0)
    x_min, x_max = np.amin(x_nonzeros), np.amax(x_nonzeros)
    y_min, y_max = np.amin(y_nonzeros), np.amax(y_nonzeros)
    z_min, z_max = np.amin(z_nonzeros), np.amax(z_nonzeros)
    x_min, x_max = sup_128(x_min, x_max)
    y_min, y_max = sup_128(y_min, y_max)
    z_min, z_max = sup_128(z_min, z_max)
    return x_min, x_max, y_min, y_max, z_min, z_max

def normalize(vol):
    mask = vol.sum(0) > 0
    for k in range(4):
        x = vol[k, ...]
        y = x[mask]
        x = (x - y.mean()) / y.std()
        vol[k, ...] = x
    return vol
# ---- end verbatim ----


def process(case_dir, tar_path):
    case_id = os.path.basename(os.path.normpath(case_dir))
    print(f"Processing {case_id}")
    load = lambda mode: nib.load(glob.glob(join(case_dir, f'*_{mode}.nii*'))[0]).get_fdata()
    flair, t1ce, t1, t2 = (load(m) for m in ("flair", "t1ce", "t1", "t2"))
    seg = load("seg").astype(np.uint8)

    vol = np.stack((flair, t1ce, t1, t2), axis=0).astype(np.float32)
    x_min, x_max, y_min, y_max, z_min, z_max = crop(vol)
    vol1 = normalize(vol[:, x_min:x_max, y_min:y_max, z_min:z_max])
    vol1 = vol1.transpose(1, 2, 3, 0)  # [H, W, D, 4]
    print(f"Volume shape after processing: {vol1.shape}")

    seg1 = seg[x_min:x_max, y_min:y_max, z_min:z_max]
    seg1[seg1 == 4] = 3

    np.save(join(tar_path, 'vol', f'{case_id}_vol.npy'), vol1)
    np.save(join(tar_path, 'seg', f'{case_id}_seg.npy'), seg1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("case_dirs", nargs="+")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    for sub in ("vol", "seg"):
        os.makedirs(join(args.out, sub), exist_ok=True)
    for d in args.case_dirs:
        process(d, args.out)
