import os
import torch

def save_checkpoint(model, optim_g, optim_d, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optim_g": optim_g.state_dict(),
        "optim_d": optim_d.state_dict(),
    }, path)

def load_checkpoint(model, optim_g, optim_d, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optim_g.load_state_dict(ckpt["optim_g"])
    optim_d.load_state_dict(ckpt["optim_d"])
    return ckpt["epoch"]
