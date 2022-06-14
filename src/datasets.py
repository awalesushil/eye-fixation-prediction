import torch

from FixationDataset import FixationDataset
from transformations import get_transformer


def get_dataloaders(data_dir, train_batch_size, valid_batch_size):

    transform = get_transformer()

    # Load the datasets
    trainset = FixationDataset(data_dir,"train_images.txt","train_fixations.txt", transform=transform)
    valset = FixationDataset(data_dir,"val_images.txt","val_fixations.txt", transform=transform)
    testset = FixationDataset(data_dir,"test_images.txt", transform=transform, test=True)

    # Prepare Dataloaders
    train = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=2)
    validation = torch.utils.data.DataLoader(valset, batch_size=valid_batch_size, shuffle=False, num_workers=2)
    test = torch.utils.data.DataLoader(testset, batch_size=valid_batch_size, shuffle=False, num_workers=2)

    return train, validation, test

