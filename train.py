import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from config import config
from dataset import MVTecDefectDataset
from models.gan_module import GANModule
from train_utils import save_checkpoint
from torch.utils.tensorboard import SummaryWriter

def main():
    device = torch.device(config.DEVICE)

    # Start with one category (e.g., "metal_nut")
    category = "metal_nut"

    dataset = MVTecDefectDataset(category_name=category)
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    gan = GANModule().to(device)

    optim_g = Adam(gan.generator.parameters(), lr=config.LR_G, betas=config.BETAS)
    optim_d = Adam(gan.discriminator.parameters(), lr=config.LR_D, betas=config.BETAS)

    best_g_loss = float("inf")     # Track best generator (total) loss
    best_epoch = -1

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(config.CHECKPOINT_DIR, "tensorboard"))
    # -------------------------
    #        TRAINING
    # -------------------------
    total_steps = 0
    for epoch in range(1, config.NUM_EPOCHS + 1):

        gan.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        epoch_g_total = 0.0
        num_batches = 0
        for batch in pbar:

            # ========= GENERATOR STEP =========
            optim_g.zero_grad()
            g_loss, g_losses_dict, fake_detached = gan.generator_step(batch, device)
            g_loss.backward()
            optim_g.step()

            # ========= DISCRIMINATOR STEP =========
            optim_d.zero_grad()
            d_loss, d_losses_dict = gan.discriminator_step(batch, fake_detached, device)
            d_loss.backward()
            optim_d.step()

            # Running statistics for best checkpoint
            epoch_g_total += g_losses_dict["g_total"].item()
            num_batches += 1
            step = total_steps
            num_batches += 1
            total_steps += 1
            writer.add_scalar("Batch/G_total", g_losses_dict["g_total"].item(), step)
            writer.add_scalar("Batch/D_total", d_losses_dict["d_total"].item(), step)
            writer.add_scalar("Batch/G_adv", g_losses_dict["g_adv"].item(), step)
            writer.add_scalar("Batch/G_l1", g_losses_dict["g_l1"].item(), step)
            writer.add_scalar("Batch/G_style", g_losses_dict["g_style"].item(), step)
            writer.add_scalar("Batch/D_style", d_losses_dict["d_style"].item(), step)
            pbar.set_postfix({
                "g_total": f"{g_losses_dict['g_total'].item():.3f}",
                "d_total": f"{d_losses_dict['d_total'].item():.3f}",
            })

        # -------------------------
        #   END OF EPOCH
        # -------------------------
        avg_g_loss = epoch_g_total / num_batches
        writer.add_scalar("Epoch/G_avg", avg_g_loss, epoch)
        writer.add_scalar("Epoch/D_last", d_losses_dict["d_total"].item(), epoch)
        avg_g_loss = epoch_g_total / num_batches

        # Save best generator checkpoint
        if avg_g_loss < best_g_loss:
            best_g_loss = avg_g_loss
            best_epoch = epoch

            best_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
            save_checkpoint(gan, optim_g, optim_d, epoch, best_path)

            print(f"✓ Saved BEST model at epoch {epoch} (avg G loss = {avg_g_loss:.4f})")

        # Save periodic checkpoints (every 25 epochs)
        if epoch % 25 == 0:
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"epoch_{epoch:03d}.pt")
            save_checkpoint(gan, optim_g, optim_d, epoch, ckpt_path)
            print(f"✓ Saved periodic checkpoint: epoch {epoch}")

    print(f"\nTraining complete. Best epoch: {best_epoch}, Best G loss: {best_g_loss:.4f}")

    # for epoch in range(1, config.NUM_EPOCHS + 1):

    #     gan.train()
    #     pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    #     epoch_g_total = 0.0
    #     num_batches = 0

    #     for i, batch in enumerate(pbar):

    #         # ========= GENERATOR STEP =========
    #         optim_g.zero_grad()
    #         g_loss, g_losses_dict, fake_detached = gan.generator_step(batch, device)
    #         g_loss.backward()
    #         optim_g.step()

    #         # ========= DEBUG HOOK =========
    #         if epoch == 1 and i == 0:
    #             clean = batch["clean"].to(device)
    #             target_defect = batch["defective"].to(device)
    #             mask = batch["mask"].to(device) if "mask" in batch else None

    #             print("Clean batch min/max:", clean.min().item(), clean.max().item())
    #             print("Defect batch min/max:", target_defect.min().item(), target_defect.max().item())
    #             if mask is not None:
    #                 print("Mask batch min/max:", mask.min().item(), mask.max().item())

    #             print("Fake (generator output) min/max:",
    #                 fake_detached.min().item(), fake_detached.max().item())

    #             # exit right after debugging to inspect values
    #             exit()

    #         # ========= DISCRIMINATOR STEP =========
    #         optim_d.zero_grad()
    #         d_loss, d_losses_dict = gan.discriminator_step(batch, fake_detached, device)
    #         d_loss.backward()
    #         optim_d.step()

    #         # Running statistics for best checkpoint
    #         epoch_g_total += g_losses_dict["g_total"].item()
    #         num_batches += 1

    #         pbar.set_postfix({
    #             "g_total": f"{g_losses_dict['g_total'].item():.3f}",
    #             "d_total": f"{d_losses_dict['d_total'].item():.3f}",
    #         })


if __name__ == "__main__":
    main()
