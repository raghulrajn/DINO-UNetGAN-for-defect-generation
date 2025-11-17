import os
import torch
from PIL import Image
from torchvision import transforms
from config import config
from models.unet import UNetGenerator


def load_generator(ckpt_path, device):
    gen = UNetGenerator(
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        base_channels=config.BASE_CHANNELS,
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    gen_state = {
        k.replace("generator.", ""): v
        for k, v in state_dict.items()
        if k.startswith("generator.")
    }

    gen.load_state_dict(gen_state, strict=True)
    gen.eval()
    return gen

def preprocess_image(path):
    img = Image.open(path).convert("RGB")

    tf = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),              # produces [0,1]
    ])

    tensor = tf(img).unsqueeze(0)
    return torch.clamp(tensor, 0.0, 1.0)

def postprocess_tensor(t):
    t = (t + 1.0) / 2.0          # [-1,1] → [0,1]
    t = torch.clamp(t, 0.0, 1.0)

    img = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype("uint8")
    return Image.fromarray(img)


def main():
    device = torch.device(config.DEVICE)

    ckpt_path = "/content/Project_/prepared_mvtec/best_model.pt"
    gen = load_generator(ckpt_path, device)

    clean_path = "/content/Project_/metal_nut_clean.png"
    style_name = "scratch"
    style_idx = config.STYLE_LABELS[style_name]

    clean_t = preprocess_image(clean_path).to(device)
    style_t = torch.tensor([style_idx], dtype=torch.long, device=device)

    with torch.no_grad():
        fake_defect = gen(clean_t, style_t)

    out_img = postprocess_tensor(fake_defect)
    out_img.save("/content/Project_/fake_defect_output2.png")

    print("Saved fake_defect_output1.png")


if __name__ == "__main__":
    main()
