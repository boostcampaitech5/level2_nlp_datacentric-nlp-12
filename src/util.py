import logging
from argparse import Namespace

import torch
import wandb
from wandb import AlertLevel

log = logging.getLogger(__name__)


def start_wandb(config: Namespace, run_name: str) -> None:
    if not config.use_wandb:
        return

    wandb.init(
        entity=config.wandb['entity'],
        project=config.wandb['project_name'],
        name=run_name,
        config=config,
    )
    wandb.alert(title='start', level=AlertLevel.INFO, text=f'{run_name}')

def finish_wandb(config: Namespace, run_name: str) -> None:
    if config.use_wandb:
        wandb.alert(title='finished', level=AlertLevel.INFO, text=f'{run_name}')

def calc_accuracy(X, Y):
    """Define metric."""
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy() / max_indices.size()[0]
    return train_acc
