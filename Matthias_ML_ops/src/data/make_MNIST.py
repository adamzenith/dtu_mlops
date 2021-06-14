# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os


# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def create_MNIST():
    # exchange with the real mnist dataset
    import torch
    from torchvision import datasets, transforms

    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path)
    embed()
    # Define a transform to normalize the data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    # Download and load the training data
    trainset = datasets.MNIST(
        r"C:\Users\matth\Documents\GitHub\dtu_mlops\Matthias_ML_ops\data\raw\MNIST_data",
        download=True,
        train=True,
        transform=transform,
    )
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    
    torch.save(trainloader, r"..\..\data\processed\train.pt")

    # Download and load the test data
    testset = datasets.MNIST(
        r"C:\Users\matth\Documents\GitHub\dtu_mlops\Matthias_ML_ops\data\raw\MNIST_data",
        download=True,
        train=False,
        transform=transform,
    )
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    torch.save(testloader, r"..\..\data\processed\test.pt")

    # Creating predict dataset

    dataset = datasets.MNIST(root=r"..\..\data\raw\MNIST_data/")

    for idx, (img, _) in enumerate(dataset):
        if idx <= 10:
            img.save("../../data/raw/{:05d}.jpg".format(idx))
            # print("yay")

    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
    return testset,trainset


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    create_MNIST()
