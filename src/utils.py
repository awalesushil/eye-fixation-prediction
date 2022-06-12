import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms


def read_text_file(filename):
    with open(filename,"r") as f:
        lines = [line.strip() for line in f.readlines()]
    return lines


def plot_sample(sample):
    
    print("Data sample for ", sample["name"])
    raw = sample["raw_image"]
    image = sample["image"]
    fixation = sample["fixation"]

    _, ax = plt.subplots(1, 3, figsize=(10,10))

    raw = raw.swapaxes(0, 1)
    raw = raw.swapaxes(1, 2)

    image = image.swapaxes(0, 1)
    image = image.swapaxes(1, 2)

    ax[0].imshow(raw)
    ax[1].imshow(image)
    ax[2].imshow(fixation.squeeze(),cmap="gray")

    plt.show()


def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    device, dtype = None, None
    if isinstance(sigma, torch.Tensor):
        device, dtype = sigma.device, sigma.dtype
    x = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp(-x.pow(2.0) / (2 * sigma ** 2))
    return gauss / gauss.sum()


plt.rcParams["savefig.bbox"] = 'tight'

def plot(images):
    fig, axs = plt.subplots(1, len(images), squeeze=False, figsize=(20,20))
    
    for i, img in enumerate(images):

        img = img.detach()
        img = transforms.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])