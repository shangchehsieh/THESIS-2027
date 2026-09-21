"""Evaluate a trained checkpoint on the held-out test split.

Mirrors trainer.test() (all 15 modality-drop combinations) but:
  - builds a loader over test.txt, which make_data_loader never constructs
  - NEVER calls trainer.save(), so the baseline results.csv / checkpoints
    in the training run's experiment dir are left untouched
  - prints one clean line per combination, naming the modalities that are
    PRESENT (the upstream log names the dropped channel indices instead)

Usage (see cli/eval_test.sh).
"""
import contextlib
import io
import itertools

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

import trainer.trainer as trainer_module
from dataloaders.datasets.brats_npy import BratsNpy, make_split_lists
from mypath import Path
from trainer import build_trainer

# Channel order of the pre-processed *_vol.npy volumes.
MODALITIES = ["Fl", "T1c", "T1", "T2"]

# PyTorch >=2.6 defaults torch.load(weights_only=True), which rejects the numpy
# scalar stored in best_pred. This checkpoint is one we produced ourselves, so
# loading it in full is safe. Patched here rather than in trainer.py so the
# training code stays untouched.
_torch_load = torch.load


def _load_trusted(*a, **kw):
    kw.setdefault("weights_only", False)
    return _torch_load(*a, **kw)


torch.load = _load_trusted


class _NoBar:
    """Drop-in for tqdm that renders nothing.

    trainer._test() wraps the loader in tqdm, which floods the log with
    progress bars. Swapping the symbol in the trainer module keeps
    trainer.py itself unmodified.
    """

    def __init__(self, iterable, *a, **kw):
        self._it = iterable

    def __iter__(self):
        return iter(self._it)

    def __len__(self):
        return len(self._it)

    def set_description(self, *a, **kw):
        pass

    def close(self):
        pass


trainer_module.tqdm = _NoBar


def present_of(drop):
    """Modality names still available, given the dropped channel indices."""
    return ",".join(m for i, m in enumerate(MODALITIES) if i not in drop)


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(args: DictConfig):
    if args.cuda:
        args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    trainer = build_trainer(args)

    root = Path.getPath('brats3d-acn')
    limit = args.dataset.limit if 'limit' in args.dataset else -1
    _, _, test_ids = make_split_lists(root, limit=limit)
    test_set = BratsNpy(root, test_ids, crop_size=args.dataset.val_size, train=False)
    # _test() reads self.val_loader, so point it at the test split
    trainer.val_loader = DataLoader(
        test_set, batch_size=args.test_batch_size,
        num_workers=args.workers, shuffle=False)

    combos = [subset
              for l in reversed(range(trainer.nchannels))
              for subset in itertools.combinations(range(trainer.nchannels), l)]

    print(f"=== EVALUATING ON TEST SPLIT: {len(test_ids)} cases, "
          f"{len(combos)} modality combinations ===\n")

    rows = []
    for n, subset in enumerate(combos, 1):
        print(f"[{n:2d}/{len(combos)}] Testing with modality: {present_of(subset)}",
              flush=True)
        # _test() prints its own "Testing with modality 0_1 dropped" block;
        # swallow it so only our lines survive.
        with contextlib.redirect_stdout(io.StringIO()):
            dice, dice_class = trainer._test(drop=subset, epoch=0)
        wt, tc, et = dice_class
        print(f"         Average scores: WT: {wt:.4f}, TC: {tc:.4f}, ET: {et:.4f}\n",
              flush=True)
        rows.append((subset, dice, dice_class))

    print("=== per-combination summary ===")
    print(f"  {'modalities used':<16}{'#':>3}{'WT':>9}{'TC':>9}{'ET':>9}")
    print("  " + "-" * 55)
    for subset, dice, dc in rows:
        print(f"  {present_of(subset):<16}{len(MODALITIES) - len(subset):>3}"
              f"{dc[0]:>9.4f}{dc[1]:>9.4f}{dc[2]:>9.4f}")

    wt, tc, et = (np.mean([dc[i] for _, _, dc in rows]) for i in range(3))
    print("  " + "-" * 55)
    print(f"\n=== {len(combos)}-combo mean over {len(test_ids)} test cases ===")
    print(f"Average scores: WT: {wt:.4f}, TC: {tc:.4f}, ET: {et:.4f}")
    print(f"Paper Table III: WT: 0.8960, TC: 0.8450, ET: 0.7030")
    print(f"Delta:          WT: {wt-0.896:+.4f}, TC: {tc-0.845:+.4f}, ET: {et-0.703:+.4f}")


if __name__ == "__main__":
    main()
