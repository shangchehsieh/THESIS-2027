from torch.utils.data import DataLoader

from mypath import Path
from dataloaders.datasets.brats_npy import BratsNpy, make_split_lists


def make_data_loader(args, **kwargs):
    if args.dataset.name == 'brats3d-acn':
        num_channels = 4
        num_class = 3

        root = Path.getPath('brats3d-acn')
        limit = args.dataset.limit if 'limit' in args.dataset else -1
        train_list, val_list, _ = make_split_lists(root, limit=limit)
        print(f"train cases: {len(train_list)}, val cases: {len(val_list)}")

        train_set = BratsNpy(root, train_list, crop_size=args.dataset.crop_size, train=True)
        val_set = BratsNpy(root, val_list, crop_size=args.dataset.val_size, train=False)
    else:
        raise NotImplementedError

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True)
    val_loader = DataLoader(
        val_set,
        batch_size=args.test_batch_size,
        num_workers=args.workers,
        shuffle=False)
    test_loader = None

    return train_loader, val_loader, test_loader, num_class, num_channels
