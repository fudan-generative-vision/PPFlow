# This source code license is herited from SiT.
# https://github.com/willisma/SiT

"""
Functions for downloading pre-trained PPFlow models
"""
from torchvision.datasets.utils import download_url
import torch
import os


pretrained_models = {'PPF-XL-2.pt', 'PPF-XL-3.pt'}


def find_model(model_name, load_all=False):
    """
    Finds a pre-trained PPFlow model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    if model_name in pretrained_models:  
        return download_model(model_name)
    else:  
        assert os.path.isfile(model_name), f'Could not find PPFlow checkpoint at {model_name}'
        checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
        if "ema" in checkpoint and not load_all:  # supports checkpoints from train.py
            checkpoint = checkpoint["ema"]
        return checkpoint


def download_model(model_name):
    """
    Downloads a pre-trained PPFlow model from the web.
    """
    assert model_name in pretrained_models
    local_path = f'pretrained_models/{model_name}'
    if not os.path.isfile(local_path):
        os.makedirs('pretrained_models', exist_ok=True)
        if model_name == "PPF-XL-2.pt":
            web_path = f'https://www.dropbox.com/scl/fi/uh4o7kra6fnwlrd2lmry0/PPF-XL-2.pt?rlkey=k26p7aw7zwar1a8x0em74vhor&st=upovbj1j&dl=0'
        elif model_name == "PPF-XL-3.pt":
            web_path = f'https://www.dropbox.com/scl/fi/0tnt99s21532sxfg804vq/PPF-XL-3.pt?rlkey=gt7shf2i23jb40pxzv9t0nje3&st=hlbtmvhi&dl=0'
        download_url(web_path, 'pretrained_models', filename=model_name)
    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model
