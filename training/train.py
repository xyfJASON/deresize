import os
import tqdm
import logging
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset import ImageDataset, collate_fn
from deresize.helper import get_model_and_processor
from scheduler import CosineWarmupLR
from distributed import (
    is_dist_avail_and_initialized, init_distributed_mode, wait_for_everyone,
    get_rank, get_local_rank, get_world_size, is_main_process,
)


def get_parser():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--dataroot", type=str, required=True)
    # model
    parser.add_argument("--model_name", type=str, required=True, choices=["siglip", "clip", "mae"])
    # training
    parser.add_argument("--num_steps", type=int, default=100000)
    parser.add_argument("--total_batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--checkpoint_freq", type=int, default=10000)
    parser.add_argument("--exp_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    # PARSE ARGS
    args = get_parser().parse_args()

    # INITIALIZE DISTRIBUTED MODE
    device = init_distributed_mode()
    print(f'Process {get_rank()} using device: {device}', flush=True)
    wait_for_everyone()

    # SETUP LOGGING
    logger = logging.getLogger('exp')
    logger.setLevel(logging.INFO if is_main_process() else logging.ERROR)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO if is_main_process() else logging.ERROR)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

    # SET SEED
    np.random.seed(args.seed + get_rank())
    torch.manual_seed(args.seed + get_rank())
    torch.cuda.manual_seed_all(args.seed + get_rank())

    # CREATE EXPERIMENT DIRECTORY
    exp_dir = args.exp_dir or f"./runs/{args.model_name}"
    writer = None
    if is_main_process():
        os.makedirs(exp_dir)
        os.makedirs(os.path.join(exp_dir, "tensorboard"))
        os.makedirs(os.path.join(exp_dir, "checkpoints"))
        writer = SummaryWriter(os.path.join(exp_dir, "tensorboard"))
        with open(os.path.join(exp_dir, "args.yaml"), "w") as f:
            for k, v in vars(args).items():
                f.write(f"{k}: {v}\n")
    logger.info(f"Experiment directory: {exp_dir}")
    wait_for_everyone()

    # LOAD DATA
    dataset = ImageDataset(args.dataroot)
    logger.info(f"Dataset size: {len(dataset)}")

    # BUILD DATALOADER
    assert args.total_batch_size % get_world_size() == 0
    bspp = args.total_batch_size // get_world_size()  # batch size per process
    datasampler = DistributedSampler(dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=True)
    dataloader = DataLoader(
        dataset, batch_size=bspp, collate_fn=collate_fn, sampler=datasampler,
        drop_last=True, num_workers=4, pin_memory=True, prefetch_factor=2,
    )
    logger.info(f"Batch size per process: {bspp}")
    logger.info(f"Total batch size: {args.total_batch_size}")

    # BUILD MODEL
    model, processor = get_model_and_processor(args.model_name)
    model.requires_grad_(False)
    model.head.requires_grad_(True)
    model.to(device)
    logger.info(f"# Total params: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"# Optimizable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # BUILD OPTIMIZER AND SCHEDULER
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineWarmupLR(
        optimizer,
        warmup_steps=args.warmup_steps,
        training_steps=args.num_steps,
    )

    # PREPARE FOR DISTRIBUTED TRAINING
    if is_dist_avail_and_initialized():
        model = DDP(model, device_ids=[get_local_rank()], output_device=get_local_rank())
    model_wo_ddp = model.module if is_dist_avail_and_initialized() else model
    wait_for_everyone()

    # TRAINING LOOP
    logger.info("Starting training...")
    model.train()
    step, epoch = 0, 0
    while step < args.num_steps:
        pbar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch}", disable=not is_main_process())
        for images, ars in pbar:
            # GET INPUTS
            inputs = processor(images=images, return_tensors="pt")
            inputs["pixel_values"] = inputs["pixel_values"].to(device)
            ars = ars.to(device)
            # MODEL FORWARD
            outputs = model(**inputs).squeeze(1)
            # LOSS
            loss = F.mse_loss(outputs, torch.log(ars))
            # OPTIMIZE
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            # LOGGING
            if is_main_process():
                pbar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)
                # SAVE CHECKPOINT
                if (step + 1) % args.checkpoint_freq == 0:
                    torch.save({
                        "model": model_wo_ddp.head.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "step": step,
                        "epoch": epoch,
                    }, os.path.join(exp_dir, "checkpoints", f"step_{step+1}.pt"))
            step += 1
            if step >= args.num_steps:
                break
        epoch += 1
    logger.info("Training completed.")


if __name__ == "__main__":
    main()
