import os

# Root that holds the datasets. Defaults to this repo directory so the
# pre-processed `BRATS2020_Training_none_npy/` folder is found out of the box.
DATASET_ROOT = os.environ.get(
    "DATASET_ROOT", os.path.dirname(os.path.abspath(__file__))
)


class Path(object):
    @staticmethod
    def getPath(dataset):
        if dataset == 'brats3d-acn':
            path = os.path.join(DATASET_ROOT, 'BRATS2020_Training_none_npy')
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError

        return os.path.realpath(path)
