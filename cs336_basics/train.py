import os 
import logging
import torch
import torch.distributed as dist
import wandb
import hydra
from omegaconf import OmegaConf
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


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    dist.init_process_group(backend="nccl")
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    if local_rank==0 and cfg.logging.use_wandb:
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(
            project=cfg.logging.wandb_project,
            name=cfg.loggin.run_name,
            config=config_dict
        )
        logger.info("Wandb successfully initialized.")


    train_data = ShardedDataset(cfg.train_data, cfg.context_length)
    valid_data = ShardedDataset(cfg.valid_data, cfg.context_length)

    train_sampler = DistributedSampler(train_data, shuffle=True)
    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, sampler=train_sampler, num_workers=cfg.num_workers, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    model = TransformerLM(vocab_size=cfg.vocab_size, context_length=cfg.context_length, num_layers=cfg.num_layers, num_heads=cfg.num_heads, d_model=cfg.d_model, d_ff=cfg.d_ff, rope_theta=cfg.theta)
    criterion = crossEntropyLoss
    optimizer = AdamWOptimizer(model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)
    if cfg.from_checkpoint:
        iteration = load_checkpoint(cfg.from_checkpoint, model, optimizer)
    else: iteration = 0
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    os.makedirs(cfg.checkponts, exist_ok=True)

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
   
    trainer.train()

    if local_rank==0 and cfg.logging.use_wandb:
        wandb.finish()


    dist.destroy_process_group()

if __name__ == "__main__":
    main()