import os
import argparse
import imageio
from tqdm import tqdm

import torch, gc
from torchvision.transforms import ConvertImageDtype

from Model import Model
from datasets import get_dataloaders

gc.collect()
torch.cuda.empty_cache()

_, _, test = get_dataloaders("/export/scratch/CV2/", 8, 8)

def save_prediciton(file_name, pred):
    file_name = file_name.split("-")[-1]
    pred = ConvertImageDtype(torch.uint8)(torch.sigmoid(pred))
    pred = pred.cpu().numpy().squeeze()
    imageio.imwrite("/export/scratch/0awale/predictions/prediction-"+file_name, pred)


def predictions(args):
    model_path = args.model_path
    model = Model()
    device  = torch.device ("cuda:0" if torch.cuda.is_available () else "cpu")
    model = model.to(device)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    if not os.path.exists("/export/scratch/0awale/predictions/"): os.mkdir("/export/scratch/0awale/predictions/")

    for batch in tqdm(test, desc="Test"):
        names, images, raws = batch["name"], batch["image"], batch["raw_image"]
        images = images.to(device)
        predictions = model(images)

        for name, pred in zip(names, predictions):
            save_prediciton(name, pred)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='Model to be used for predictions', type=str)
    args = parser.parse_args()
    predictions(args)
