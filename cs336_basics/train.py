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

from cs336_basics.model import TransformerLM
from cs336_basics.loss import crossEntropyLoss
from cs336_basics.optimizer import AdamWOptimizer
from cs336_basics.utils import load_checkpoint, save_checkpoint
from cs336_basics.data import ShardedDataset
from cs336_basics.trainer import CausalLMTrainer

def setup_logger(log_dir="logs", log_file="training.log"):

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger("MyTrainer")
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_path, mode='a')

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
logger = setup_logger()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):

    dist.init_process_group(backend="nccl")
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if local_rank == 0:
        logger.info(f"Distributed process group initialized. Using device cuda:{local_rank}.")

    if local_rank==0 and cfg.logging.use_wandb:
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(
            project=cfg.logging.wandb_project,
            name=cfg.logging.run_name,
            config=config_dict
        )
        logger.info("Wandb successfully initialized.")


    train_data = ShardedDataset(cfg.dataset.train_data, cfg.model.context_length)
    valid_data = ShardedDataset(cfg.dataset.valid_data, cfg.model.context_length)
    
    if local_rank == 0:
        logger.info(f"Data loaded: {len(train_data):,} training samples, {len(valid_data):,} validation samples.")

    train_sampler = DistributedSampler(train_data, shuffle=True)
    valid_sampler = DistributedSampler(valid_data, shuffle=False)
    
    train_loader = DataLoader(train_data, batch_size=cfg.training.batch_size, sampler=train_sampler, num_workers=cfg.training.num_workers, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=cfg.training.batch_size, sampler=valid_sampler, num_workers=cfg.training.num_workers, drop_last=False)

    vocab_size = train_data.meta["vocab_size"]
    model = TransformerLM(vocab_size=vocab_size, context_length=cfg.model.context_length, num_layers=cfg.model.num_layers, num_heads=cfg.model.num_heads, d_model=cfg.model.d_model, d_ff=cfg.model.d_ff, rope_theta=cfg.model.theta)
    
    if local_rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Initialized TransformerLM model with {num_params:,} parameters.")
        
    model = model.to(local_rank)
    criterion = crossEntropyLoss
    optimizer = AdamWOptimizer(model.parameters(), lr=cfg.training.lr, betas=(cfg.training.beta1, cfg.training.beta2), weight_decay=cfg.training.weight_decay)
    
    if local_rank == 0:
        logger.info(f"Optimizer configured. Resuming from checkpoint: {cfg.model.from_checkpoint}")
        
    if cfg.model.from_checkpoint:
        iteration = load_checkpoint(cfg.model.from_checkpoint, model, optimizer)
    else: iteration = 0
    model = DDP(model, device_ids=[local_rank])
    os.makedirs(cfg.dataset.checkpoint_dir, exist_ok=True)

    trainer = CausalLMTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        global_step=iteration,
        local_rank=local_rank,
        cfg=cfg
    )
   
    if local_rank == 0:
        logger.info("Starting CausalLMTrainer training loop...")
    trainer.train()

    if local_rank==0:
        logger.info("Training completed.")
        
    if local_rank==0 and cfg.logging.use_wandb:
        wandb.finish()


    dist.destroy_process_group()

if __name__ == "__main__":
    main()