import numpy as np
from .MyDataset import MyDataset
from torch.utils.data import DataLoader, random_split
from logger import logging
from torchvision.datasets.folder import ImageFolder
from Project import project

def get_dataloaders(
        train_dir,
        test_dir,
        train_transform=None,
        test_transform=None,
        split=(0.5, 0.5),
        batch_size=32,
        *args, **kwargs):
    """
    This function returns the train, val and test dataloaders.
    """
    # create the datasets
    train_ds = MyDataset(project.base_dir / 'train.csv', project.dataset_dir / 'train', transform=train_transform)
    test_ds = MyDataset(project.base_dir / 'sample_submission.csv', project.dataset_dir / 'test', transform=test_transform)
    # now we want to split the train_ds in validation and train
    lengths = np.array(split) * len(train_ds)
    lengths = lengths.astype(int)
    left = len(train_ds) - lengths.sum()
    # we need to add the different due to float approx to int
    lengths[-1] += left

    train_ds, val_ds = random_split(train_ds, lengths.tolist())
    logging.info(f'Train samples={len(train_ds)}, Validation samples={len(val_ds)}, Test samples={len(test_ds)}')

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, *args, **kwargs)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, *args, **kwargs)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, *args, **kwargs)

    return train_dl, val_dl, test_dl

