import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.multiprocessing import spawn
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

import numpy as np

from autoencoder import *


def main_worker(gpu, ngpus_per_node, args):
    rank = gpu
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=ngpus_per_node, rank=rank
    )
    torch.cuda.set_device(gpu)

    # ── Models ──
    autoencoder = SpatialVAE(in_ch=1, latent_ch=3).cuda(gpu)
    discriminator = PatchDiscriminator(in_ch=1).cuda(gpu)

    autoencoder = nn.parallel.DistributedDataParallel(autoencoder, device_ids=[gpu])
    discriminator = nn.parallel.DistributedDataParallel(discriminator, device_ids=[gpu])

    # ── Distributed DataLoaders ──
    train_sampler = DistributedSampler(
        args["train_dataset"], num_replicas=ngpus_per_node, rank=rank, shuffle=True
    )
    train_loader = DataLoader(
        args["train_dataset"],
        batch_size=args["batch_size"],
        sampler=train_sampler,
        num_workers=args["num_workers"],
        pin_memory=True,
    )

    test_loader = None
    if args["test_dataset"] is not None:
        test_loader = DataLoader(
            args["test_dataset"],
            batch_size=args["batch_size"],
            shuffle=False,
            num_workers=args["num_workers"],
            pin_memory=True,
        )

    # ── Optimizers ──
    opt_ae = torch.optim.AdamW(
        autoencoder.parameters(), lr=args["lr_ae"], betas=(0.5, 0.9), weight_decay=1e-4
    )
    opt_disc = torch.optim.AdamW(
        discriminator.parameters(),
        lr=args["lr_disc"],
        betas=(0.5, 0.9),
        weight_decay=1e-4,
    )

    sched_ae = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ae, args["epochs"])
    sched_disc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_disc, args["epochs"])

    # ── Loss ──
    recon_criterion = SharpReconLoss().cuda(gpu)

    for epoch in range(args["epochs"]):
        train_sampler.set_epoch(epoch)  # ensure shuffle differs per epoch
        autoencoder.train()
        discriminator.train()
        use_disc = epoch >= args["disc_start_epoch"]

        kl_weight = args["max_kl_weight"] * min(1.0, epoch / args["kl_warmup_epochs"])

        metrics = {"recon": 0.0, "kl": 0.0, "g_loss": 0.0, "d_loss": 0.0}

        for images, _ in train_loader:
            images = images.cuda(gpu, non_blocking=True)
            recon, mean, logvar, _ = autoencoder(images)

            # ── Discriminator ──
            if use_disc:
                real_pred = discriminator(images)
                d_real = F.relu(1.0 - real_pred).mean()

                fake_pred = discriminator(recon.detach())
                d_fake = F.relu(1.0 + fake_pred).mean()

                d_loss = 0.5 * (d_real + d_fake)
                opt_disc.zero_grad()
                d_loss.backward()
                opt_disc.step()
                metrics["d_loss"] += d_loss.item()

            # ── Autoencoder ──
            recon_loss, loss_dict = recon_criterion(recon, images)
            kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
            ae_loss = recon_loss + kl_weight * kl_loss

            if use_disc:
                fake_pred = discriminator(recon)
                g_loss = -fake_pred.mean()
                ae_loss += args["disc_weight"] * g_loss
                metrics["g_loss"] += g_loss.item()

            opt_ae.zero_grad()
            ae_loss.backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 1.0)
            opt_ae.step()

            metrics["recon"] += loss_dict["l1"]
            metrics["kl"] += kl_loss.item()

        sched_ae.step()
        sched_disc.step()
        n = len(train_loader)

        # ── Validation ──
        val_str = ""
        if test_loader is not None:
            autoencoder.eval()
            val_loss = 0
            with torch.no_grad():
                for images, _ in test_loader:
                    images = images.cuda(gpu, non_blocking=True)
                    recon, _, _, _ = autoencoder(images)
                    val_loss += F.l1_loss(recon, images).item()
            val_str = f" | Val: {val_loss/len(test_loader):.4f}"

        disc_str = (
            f" | D: {metrics['d_loss']/n:.4f} | G: {metrics['g_loss']/n:.4f}"
            if use_disc
            else ""
        )

        # Only rank 0 prints to avoid duplicates
        if rank == 0:
            torch.save(autoencoder.module.state_dict(), "spatial_autoencoder.pth")
            print(
                f"Epoch {epoch+1}/{args['epochs']} | Recon: {metrics['recon']/n:.4f} | "
                f"KL: {metrics['kl']/n:.4f} | KL_w: {kl_weight:.7f}{disc_str}{val_str}"
            )

        # torch.save(autoencoder, './spatial_autoencoder.pth')


def train_ddp(
    train_dataset,
    test_dataset=None,
    batch_size=64,
    epochs=80,
    lr_ae=1e-4,
    lr_disc=4e-4,
    max_kl_weight=1e-4,
    kl_warmup_epochs=30,
    disc_start_epoch=30,
    disc_weight=0.1,
    num_workers=2,
):

    ngpus = torch.cuda.device_count()
    args = dict(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        epochs=epochs,
        lr_ae=lr_ae,
        lr_disc=lr_disc,
        max_kl_weight=max_kl_weight,
        kl_warmup_epochs=kl_warmup_epochs,
        disc_start_epoch=disc_start_epoch,
        disc_weight=disc_weight,
        num_workers=num_workers,
    )
    spawn(main_worker, nprocs=ngpus, args=(ngpus, args))


if __name__ == "__main__":
    # load data
    data_file = "data_96_bw.npz"
    data = np.load(data_file)

    train_tensor = torch.from_numpy(data["train"].astype(np.float32))
    test_tensor = torch.from_numpy(data["test"].astype(np.float32))

    train_tensor_dataset = TensorDataset(
        train_tensor, torch.zeros(len(train_tensor), dtype=torch.long)
    )
    test_tensor_dataset = TensorDataset(
        test_tensor, torch.zeros(len(test_tensor), dtype=torch.long)
    )

    train_loader = DataLoader(
        train_tensor_dataset, batch_size=64, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_tensor_dataset, batch_size=64, shuffle=False, num_workers=2
    )

    # load train_dataset, test_dataset
    train_ddp(train_tensor_dataset, test_dataset=test_tensor_dataset)
