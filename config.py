import os

class Config:
    RAW_MVTEC_ROOT = r"C:\Users\Raghul\Desktop\Project\mvtec_ad"         
    PREPARED_ROOT = r"C:\Users\Raghul\Downloads\ProjectNew\Project_\prepared_mvtec"           
    LOG_DIR = "logs"
    CHECKPOINT_DIR = r"C:\Users\Raghul\Downloads\ProjectNew\Project_\prepared_mvtec\checkpoints"

    IMAGE_SIZE = 256
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    STYLE_LABELS = {
        "scratch": 0,
        "bent": 1,
        "hole": 2,
        "hole": 3,
    }
    NUM_STYLES = len(STYLE_LABELS)

    IN_CHANNELS = 3
    OUT_CHANNELS = 3
    BASE_CHANNELS = 64
    STYLE_EMB_DIM = 64
    VIT_PATCH_SIZE = 16
    VIT_DIM = 256
    VIT_DEPTH = 6
    VIT_HEADS = 8
    VIT_MLP_DIM = 512

    DINO_IMG_SIZE = 224    # DINO / most ViTs use 224×224
    DINO_FEATURE_DIM = 384

    # Training
    NUM_EPOCHS = 100
    LR_G = 2e-4
    LR_D = 2e-4
    BETAS = (0.5, 0.999)
    LAMBDA_L1 = 10.0
    LAMBDA_ADV = 1.0
    LAMBDA_STYLE = 1.0
    USE_MASKED_L1 = True

    DEVICE = "cpu"  # or "cuda/xpu"

config = Config()
