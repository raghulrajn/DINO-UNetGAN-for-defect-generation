import os
import json
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm
from config import config

def inpaint_clean(img, mask):
    # img: HxWx3, mask: HxW, 0 background, 255 defect
    mask_uint8 = mask.astype(np.uint8)
    clean = cv2.inpaint(img, mask_uint8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return clean

def prepare_category(category_name):
    """
    Prepares one mvtec category, e.g. 'metal_nut'
    Expects mvtec structure:
    category/
        train/good/
        test/good/
        test/<defect_type>/
        ground_truth/<defect_type>/
    """
    raw_root = os.path.join(config.RAW_MVTEC_ROOT, category_name)
    prepared_root = os.path.join(config.PREPARED_ROOT, category_name)
    os.makedirs(os.path.join(prepared_root, "clean"), exist_ok=True)
    os.makedirs(os.path.join(prepared_root, "defective"), exist_ok=True)
    os.makedirs(os.path.join(prepared_root, "masks"), exist_ok=True)

    style_labels = {}  # filename -> style_name

    defect_types = [
        d for d in os.listdir(os.path.join(raw_root, "test"))
        if d != "good"
    ]

    idx = 0
    for defect_type in defect_types:
        img_dir = os.path.join(raw_root, "test", defect_type)
        mask_dir = os.path.join(raw_root, "ground_truth", defect_type)

        img_paths = sorted(glob(os.path.join(img_dir, "*.png")))
        mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))

        for img_path, mask_path in tqdm(
            list(zip(img_paths, mask_paths)),
            desc=f"{category_name} - {defect_type}"
        ):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            clean = inpaint_clean(img, mask)

            base_name = f"{category_name}_{defect_type}_{idx:05d}.png"
            clean_out = os.path.join(prepared_root, "clean", base_name)
            defect_out = os.path.join(prepared_root, "defective", base_name)
            mask_out = os.path.join(prepared_root, "masks", base_name)

            cv2.imwrite(clean_out, clean)
            cv2.imwrite(defect_out, img)
            cv2.imwrite(mask_out, mask)

            style_labels[base_name] = defect_type
            idx += 1

    # save labels
    with open(os.path.join(prepared_root, "style_labels.json"), "w") as f:
        json.dump(style_labels, f, indent=2)

def main():
    categories = [d for d in os.listdir(config.RAW_MVTEC_ROOT)
                  if os.path.isdir(os.path.join(config.RAW_MVTEC_ROOT, d))]
    for cat in categories:
        prepare_category(cat)

if __name__ == "__main__":
    main()
