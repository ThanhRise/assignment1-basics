"""
Train Triton MoE Transformer
==============================
Distributed training script for TritonGroupedGEMMMoE-based Transformer LM.
Inherits the same training infrastructure (CausalLMTrainer, DDP, WandB, etc.)
from train.py but swaps the dense model for the Triton MoE variant.

Usage (single GPU):
  torchrun --nproc_per_node=1 -m cs336_basics.train_triton_moe

Usage (multi-GPU):
  torchrun --nproc_per_node=4 -m cs336_basics.train_triton_moe

Config overrides (Hydra):
  torchrun --nproc_per_node=1 -m cs336_basics.train_triton_moe \
      model.num_experts=32 model.num_experts_per_tok=8 training.batch_size=4
"""

import os
import logging
import torch
import torch.distributed as dist
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from cs336_basics.model import TransformerLM_TritonMoE
from cs336_basics.loss import crossEntropyLoss
from cs336_basics.optimizer import AdamWOptimizer
from cs336_basics.utils import load_checkpoint, save_checkpoint
from cs336_basics.data import ShardedDataset
from cs336_basics.trainer import CausalLMTrainer


def setup_logger(log_dir="logs", log_file="training_moe.log"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger("MoETrainer")
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_path, mode='a')

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


logger = setup_logger()


@hydra.main(version_base=None, config_path="../conf", config_name="config_moe")
def main(cfg: DictConfig):

    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if local_rank == 0:
        logger.info(f"Distributed process group initialized. Using device cuda:{local_rank}.")
        logger.info(f"MoE Config: {cfg.model.num_experts} experts, top-{cfg.model.num_experts_per_tok}")

    if local_rank == 0 and cfg.logging.use_wandb:
        # Login with API key from config or env var
        api_key = cfg.logging.get("wandb_api_key", "") or os.environ.get("WANDB_API_KEY", "")
        if api_key:
            wandb.login(key=api_key)
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(
            project=cfg.logging.wandb_project,
            name=cfg.logging.run_name,
            config=config_dict,
        )
        logger.info("Wandb successfully initialized.")

    # ── Data ──────────────────────────────────────────────────
    train_data = ShardedDataset(cfg.dataset.train_data, cfg.model.context_length)
    valid_data = ShardedDataset(cfg.dataset.valid_data, cfg.model.context_length)

    if local_rank == 0:
        logger.info(f"Data loaded: {len(train_data):,} training samples, {len(valid_data):,} validation samples.")

    train_sampler = DistributedSampler(train_data, shuffle=True)
    valid_sampler = DistributedSampler(valid_data, shuffle=False)

    train_loader = DataLoader(
        train_data, batch_size=cfg.training.batch_size,
        sampler=train_sampler, num_workers=cfg.training.num_workers, drop_last=True,
    )
    valid_loader = DataLoader(
        valid_data, batch_size=cfg.training.batch_size,
        sampler=valid_sampler, num_workers=cfg.training.num_workers, drop_last=False,
    )

    # ── Model ─────────────────────────────────────────────────
    vocab_size = train_data.meta["vocab_size"]
    model = TransformerLM_TritonMoE(
        vocab_size=vocab_size,
        context_length=cfg.model.context_length,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        d_model=cfg.model.d_model,
        d_ff=cfg.model.d_ff,
        rope_theta=cfg.model.theta,
        num_experts=cfg.model.num_experts,
        num_experts_per_tok=cfg.model.num_experts_per_tok,
    )

    if local_rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        num_expert_params = sum(
            p.numel() for name, p in model.named_parameters()
            if any(k in name for k in ("w1", "w2", "w3"))
        )
        logger.info(
            f"Initialized TritonMoE model: {num_params:,} total params, "
            f"{num_expert_params:,} expert params ({num_expert_params/num_params*100:.1f}%)"
        )

    model = model.to(local_rank)
    criterion = crossEntropyLoss
    optimizer = AdamWOptimizer(
        model.parameters(), lr=cfg.training.lr,
        betas=(cfg.training.beta1, cfg.training.beta2),
        weight_decay=cfg.training.weight_decay,
    )

    if local_rank == 0:
        logger.info(f"Optimizer configured. Resuming from checkpoint: {cfg.model.from_checkpoint}")

    if cfg.model.from_checkpoint:
        iteration = load_checkpoint(cfg.model.from_checkpoint, model, optimizer)
    else:
        iteration = 0

    # MoE: not all expert weights receive gradients every step (only top-K selected),
    # so DDP must be told to expect unused parameters.
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    os.makedirs(cfg.dataset.checkpoint_dir, exist_ok=True)

    # ── Train ─────────────────────────────────────────────────
    trainer = CausalLMTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        global_step=iteration,
        local_rank=local_rank,
        cfg=cfg,
    )

    if local_rank == 0:
        logger.info("Starting CausalLMTrainer training loop (Triton MoE)...")
    trainer.train()

    if local_rank == 0:
        logger.info("Training completed.")

    if local_rank == 0 and cfg.logging.use_wandb:
        wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
