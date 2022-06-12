import os
import argparse
from tqdm import tqdm

import numpy as np

import torch
from torch.optim import SGD
import torch.nn.functional as F
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
torch.set_grad_enabled(True)

from datasets import get_dataloaders
from Model import Model


def main(args):
    
    epochs = args.epochs
    resume = args.resume
    checkpoint = args.checkpoint
    learning_rate = args.lr
    train_batch_size = args.train_batch_size
    valid_batch_size = args.valid_batch_size
    data_dir = args.data_dir

    if not os.path.exists("../checkpoints"): os.mkdir("../checkpoints")

    PATH = "../checkpoints/"

    train, valid, _ = get_dataloaders(data_dir, train_batch_size, valid_batch_size)

    # Model
    model = Model()
    device  = torch.device ("cuda:0" if torch.cuda.is_available () else "cpu")
    model = model.to(device)

    print(model)

    # Optimizer
    opt = SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Loss function
    def loss_fn(pred, y):
        loss = F.binary_cross_entropy_with_logits(pred, y)
        return loss


    # Training
    writer = SummaryWriter()

    last_epoch = 0
    if resume:
        checkpoint = torch.load(PATH+checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        last_epoch = checkpoint["epoch"]


    for epoch in tqdm(range(last_epoch, epochs+last_epoch), desc="Train"):

        model.train()
        for train_batch in tqdm(train, desc="Epoch: " + str(epoch)):
            
            epoch_loss = []

            inputs, targets = train_batch["image"], train_batch["fixation"]

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            loss = loss_fn(outputs, targets)
            loss.backward()
            
            opt.step()
            opt.zero_grad()
            
            normalized_outputs = torch.sigmoid(outputs)

            epoch_loss.append(loss.item())

        writer.add_scalar("Loss/train", np.mean(epoch_loss), epoch)
        
        torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict()
            }, PATH+"epoch_{}_train_loss_{}.pth".format(epoch, np.mean(epoch_loss)))

        # Plot
        grid = make_grid([normalized_outputs])
        writer.add_images(str(epoch), grid)
                
        model.eval()
        with torch.no_grad():

            epoch_loss = []
            for val_batch in tqdm(valid, desc="Validation"):
                inputs, targets = val_batch["image"], val_batch["fixation"]
            
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                
                valid_loss = loss_fn(outputs, targets).item()
                epoch_loss.append(valid_loss)
            
            print("Epoch {}: validation loss {}".format(epoch, np.mean(epoch_loss)))
            writer.add_scalar("Loss/valid", np.mean(epoch_loss), epoch)

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters to train the model.')
    parser.add_argument('--epochs', default=15, help='Number of epochs to train the model for', type=int)
    parser.add_argument('--lr', default=0.0004, help='Learning Rate', type=float)
    parser.add_argument('--resume', default=False, help='Resume training', type=bool)
    parser.add_argument('--checkpoint', default=None, help='Checkpoint dictionary', type=dict)
    parser.add_argument('--train_batch_size', default=32, help='Batch size for train', type=int)
    parser.add_argument('--valid_batch_size', default=64, help='Batch size for valid/test', type=int)
    parser.add_argument('--data_dir', default="data/", help='Data directory path', type=str)
    args = parser.parse_args()
    main(args)