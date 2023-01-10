import logging

import torch

# Tensorboard
try:
    from pytorch_lightning.loggers import TensorBoardLogger

    tb_available = True
except:
    logging.warning('Tensorboard is not available')
    tb_available = False
    TensorBoardLogger = None

# WandB
try:
    from pytorch_lightning.loggers import WandbLogger
    import wandb

    wandb_available = True
except:
    logging.warning('WandB is not available')
    wandb_available = False
    WandbLogger = None

# Aim
try:
    from aim.pytorch_lightning import AimLogger
    from aim import Image as AimImage

    aim_available = True
except:
    logging.warning('Aim is not available')
    aim_available = False
    AimLogger = None


def log_pl_image(pl_module, image: torch.Tensor, name='train_image'):
    # Image must be in range 0..1

    for logger in pl_module.loggers:
        print('Log image', pl_module.global_step)  # To avoid segmentation fault
        if tb_available and isinstance(logger, TensorBoardLogger):
            logger.experiment.add_image(name, image, pl_module.global_step)
        elif wandb_available and isinstance(logger, WandbLogger):
            logger.experiment.log({name: [wandb.Image(image)]}, step=pl_module.global_step)
        elif aim_available and isinstance(logger, AimLogger):
            aim_image = AimImage(image)
            logger.experiment.track(aim_image, name=name, step=pl_module.global_step)