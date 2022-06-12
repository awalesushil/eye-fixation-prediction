import os
import imageio
from tqdm import tqdm

import torch
from torchvision.transforms import ConvertImageDtype

from Model import Model
from src.datasets import get_dataloaders

_, _, test = get_dataloaders()

def save_prediciton(file_name, pred):
    file_name = file_name.split("-")[-1]
    pred = ConvertImageDtype(torch.uint8)(torch.sigmoid(pred))
    pred = pred.cpu().numpy().squeeze()
    imageio.imwrite("../predictions/prediction-"+file_name, pred)


def predictions(testloader, model_path):
    
    model = Model()
    device  = torch.device ("cuda:0" if torch.cuda.is_available () else "cpu")
    model = model.to(device)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    if not os.path.exists("../predictions"): os.mkdir("../predictions")

    for batch in tqdm(testloader, desc="Test"):
        names, images, raws = batch["name"], batch["image"], batch["raw_image"]
        images = images.to(device)
        predictions = model(images)

        for name, pred in zip(names, predictions):
            save_prediciton(name, pred)

predictions(test)