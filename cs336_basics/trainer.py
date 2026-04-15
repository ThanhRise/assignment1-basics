import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
from omegaconf import DictConfig

from cs336_basics.lr_schedule import lr_cosine_schedule
from cs336_basics.utils import gradient_clipping, save_checkpoint
import logging
logger = logging.getLogger(__name__)

def get_infinite_batches(dataloader, epoch=0):
    if hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
        dataloader.sampler.set_epoch(epoch)
    for batch in dataloader:
        yield batch


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class CausalLMTrainer:
    def __init__(self,
                 model: torch.nn.Module, 
                 train_loader: torch.utils.data.DataLoader,
                 valid_loader: torch.utils.data.DataLoader, 
                 criterion,
                 optimizer,
                 global_step: int,
                 local_rank,
                 cfg: DictConfig):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.local_rank = local_rank
        self.cfg = cfg

        self.global_step = global_step
        self.current_epoch = global_step // self.cfg.iters_per_epoch
        self.max_step = self.cfg.iters_per_epoch * self.cfg.max_epoch

    def train_epoch(self):
        pass

    def train_iter_per_epoch(self, data_iterator):
        self.model.train()
        total_loss = 0
        step_trained_in_current_epoch = self.global_step % self.cfg.iters_per_epoch

        for step in range(self.cfg.iters_per_epoch):
            batch = next(data_iterator)
            if step < step_trained_in_current_epoch:
                continue
            current_lr = lr_cosine_schedule(self.global_step, self.cfg.lr, self.cfg.min_lr, self.cfg.warmup, self.max_step)
            set_learning_rate(self.optimizer, current_lr)
            input_ids = batch['input_ids'].to(self.local_rank)
            labels = batch['labels'].to(self.local_rank)

            self.optimizer.zero_grad()
            logits = self.model(input_ids)
            loss = self.criterion(logits, labels)
            loss.backward()

            gradient_clipping(self.model.parameters(), self.cfg.max_l2_norm)
            self.optimizer.step()

            self.global_step+=1
            total_loss += loss.item()
            if self.local_rank==0 and self.global_step % self.cfg.log_interval == 0:
                logger.info(
                    f"Step: [{self.global_step}/{self.max_step}] | Loss: {loss.item():0.4f} | Lr: {current_lr}"
                )
                if self.use_wandb:
                    wandb.log({
                        "train/step_loss": loss.item(),
                        "train/learning_rate": current_lr,
                        "global_step": self.global_step
                    })
                
        return total_loss / (self.cfg.iters_per_epoch / step_trained_in_current_epoch)
    
    @torch.no_grad()
    def evaluation(self):
        self.model.eval()
        total_loss = 0
        for batch in tqdm(self.valid_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(self.local_rank)
            labels = batch['lables'].to(self.local_rank)

            logits = self.model(input_ids)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
        
        return total_loss / len(self.valid_loader)


    def train(self):
        for epoch in range(self.current_epoch, self.cfg.max_epoch):
            self.current_epoch = epoch
            data_iterator = iter(get_infinite_batches(self.train_loader, epoch))
            self.train_iter_per_epoch(data_iterator)
            if self.local_rank == 0:
                save_checkpoint(self.model, self.optimizer, self.current_epoch * self.cfg.iters_per_epoch)
                loss_eval = self.evaluation()
                logger.info(f"Evaluation at epoch {epoch} | loss_eval: {loss_eval}")
                if self.use_wandb:
                    wandb.log({
                        "validation/epoch": epoch,
                        "validation/loss": loss_eval
                    })
        
