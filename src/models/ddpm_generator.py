import logging
import os.path
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from omegaconf import OmegaConf
from pytorch_lightning.utilities import rank_zero_only

from src.models.components.utils.init_net import init_net

from src.optimizers.optimizer_constructor import DefaultOptimizerConstructor
from src.utils.instantiate import instantiate
from src.utils.load_pl_dict import load_dict
from src.utils.logging import log_pl_image
from tqdm import tqdm


class DDPMModule(pl.LightningModule):

    def __init__(self, model_config, training_config=None) -> None:
        super(DDPMModule, self).__init__()
        self.config = model_config
        self.training_config = training_config
        self.save_hyperparameters('model_config', 'training_config')

        self.netG = self.create_generator()
        logging.info(self.netG)
        self.register_buffer('mean', torch.tensor(self.config.norm.mean).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor(self.config.norm.std).view(1, -1, 1, 1))

        self.img_size = self.config.img_size
        self.example_input_array = [torch.randn(1, 3, self.img_size, self.img_size), 0]

        # Generate steps
        num_steps = self.config.scheduling.num_steps
        start_beta = self.config.scheduling.start_beta
        end_beta = self.config.scheduling.end_beta
        betas = np.linspace(start=start_beta, stop=end_beta, num=num_steps, dtype=np.float32)
        alphas = 1 - betas
        tilda_alphas = np.cumprod(alphas)
        sqrt_tildas_alphas = np.sqrt(tilda_alphas)  # Used in prediction and sampling
        sqrt_one_minus_tildas_alphas = np.sqrt(1 - tilda_alphas)  # Used in prediction and sampling
        sigmas = np.sqrt(betas)
        self.register_buffer('sqrt_alphas', torch.tensor(np.sqrt(alphas)).view(-1, 1, 1, 1))
        self.register_buffer('one_minus_alphas', torch.tensor(1 - alphas).view(-1, 1, 1, 1))
        self.register_buffer('one_minus_tildas_alphas', torch.tensor(1 - tilda_alphas).view(-1, 1, 1, 1))
        self.register_buffer('sqrt_tildas_alphas', torch.tensor(sqrt_tildas_alphas).view(-1, 1, 1, 1))
        self.register_buffer('sqrt_one_minus_tildas_alphas', torch.tensor(sqrt_one_minus_tildas_alphas).view(-1, 1, 1, 1))
        self.register_buffer('sigmas', torch.tensor(sigmas).view(-1, 1, 1, 1))
        self.register_buffer('num_steps', torch.tensor(num_steps, dtype=torch.int32).view(-1, 1, 1, 1))

        # Steps embedding
        steps_embed = np.linspace(start=-1, stop=1, num=num_steps, dtype=np.float32)
        self.register_buffer('steps_embed', torch.tensor(steps_embed).view(-1, 1, 1, 1))
        ################

        if self.training_config is not None:
            self.automatic_optimization = False
            self.call_count = defaultdict(int)

            self.target_loss = instantiate(self.training_config.losses.target_loss)

            # Weights block
            self.target_weight = self.training_config.losses.weights.target_weight

            self.register_buffer('pl_mean', torch.zeros([]))

            self.register_buffer('cur_nimg', torch.zeros([], dtype=torch.long))
            self.register_buffer('num_iters', torch.zeros([], dtype=torch.long))

            self.offset = 0

    def create_generator(self):
        netG = instantiate(self.config.netG)
        if self.training_config.initialization.pretrain_checkpoint_G:
            load_dict(netG, 'netG', self.training_config.initialization.pretrain_checkpoint_G)
        else:
            init_net(netG, **self.training_config.initialization.init_G)
        return netG

    def backward_mapping(self, real_tensor):
        return (real_tensor * self.std + self.mean)

    def forward_mapping(self, real_tensor):
        return (real_tensor - self.mean) / self.std

    def forward(self, input, step):
        return self.netG(input, step)

    def training_step(self, batch, batch_idx):
        image = batch['image']

        bs = image.shape[0]
        gradient_accumulation = max(bs // self.training_config.inner_batch_size, 1)
        assert bs % self.training_config.inner_batch_size == 0 and bs >= self.training_config.inner_batch_size
        inner_batch_size = self.training_config.inner_batch_size

        # Gen noise input
        device = image.device

        # Uniform steps
        bin_size = self.num_steps // bs
        steps = torch.tensor([self.offset + i * bin_size for i in range(bs)], dtype=torch.long, device=device)
        self.offset += 1
        if self.offset + bin_size * (bs - 1) >= self.num_steps:
            self.offset = 0
        ###############

        eps = torch.randn(image.shape, device=device)
        sqrt_tildas_alphas = self.sqrt_tildas_alphas[steps]
        sqrt_one_minus_tildas_alphas = self.sqrt_one_minus_tildas_alphas[steps]
        steps_embed = self.steps_embed[steps]
        noised_input = sqrt_tildas_alphas * image + sqrt_one_minus_tildas_alphas * eps
        just_input = sqrt_tildas_alphas * image

        for i in range(gradient_accumulation):
            noised_input_small = noised_input[i * inner_batch_size: (i + 1) * inner_batch_size]
            just_input_small = just_input[i * inner_batch_size: (i + 1) * inner_batch_size]
            noise = eps[i * inner_batch_size: (i + 1) * inner_batch_size]
            sqrt_one_minus_tildas_alphas_small = sqrt_one_minus_tildas_alphas[
                                                 i * inner_batch_size: (i + 1) * inner_batch_size]
            steps_embed_small = steps_embed[i * inner_batch_size: (i + 1) * inner_batch_size]

            ######################
            # Optimize Generator #
            ######################
            # This correct else does not converge

            pred_eps = self(noised_input_small, steps_embed_small)

            loss_G = self.generator_loss(noise, pred_eps, noised_input_small)

            if i == 0:
                with torch.no_grad():
                    denoised = noised_input_small - sqrt_one_minus_tildas_alphas_small * pred_eps
                    self.log_images(just_input_small, noised_input, denoised.detach())
            self.log('loss_G', loss_G, prog_bar=True)
            self.manual_backward(loss_G / gradient_accumulation)

        ########## Run optimization ################
        self.optimize_G()
        #########################################

        self.schedule_G()

        self.cur_nimg += bs
        self.log('cur_nimg', self.cur_nimg, prog_bar=True)
        self.num_iters += 1

    def on_fit_start(self) -> None:
        # Create validation folder
        if not os.path.exists('output'):
            self.val_folder = Path('output')
            self.val_folder.mkdir(exist_ok=True, parents=True)

    def validation_step(self, batch, batch_nb):
        xT = torch.randn(batch.shape[0], 3, self.img_size, self.img_size).type_as(batch)
        xt = xT
        for t in tqdm(reversed(range(0, self.num_steps)), total=int(self.num_steps.item())):
            z = torch.randn_like(xt) if t > 0 else 0
            sqrt_alphat = self.sqrt_alphas[t]
            one_minus_alphas = self.one_minus_alphas[t]
            steps_embed = self.steps_embed[t]
            sqrt_one_minus_tildas_alphas = self.sqrt_one_minus_tildas_alphas[t]
            sigma = self.sigmas[t]

            xt = 1 / sqrt_alphat * (xt - one_minus_alphas / sqrt_one_minus_tildas_alphas *
                                    self(xt, steps_embed)) + sigma * z

        grid = xt
        grid = self.backward_mapping(grid)

        torchvision.utils.save_image(grid, str(self.val_folder / (str(batch_nb) + '.png')), nrow=min(batch.shape[0], 8))

        return {}

    @rank_zero_only
    def log_images(self, *images):
        # tensors [self.real, self.fake, preds, self.cartoon]
        images = list(images)
        if self.check_count('img_log_freq', self.training_config.logging.img_log_freq) or self.global_step in (0, 1):
            out_image = torch.cat(images, dim=0)

            grid = torchvision.utils.make_grid(out_image, nrow=len(images[0]))
            grid = self.backward_mapping(grid)[0]
            grid = torch.clamp(grid, 0.0, 1.0)

            log_pl_image(self, grid)

    def generator_loss(self, noise, pred_eps, noised_input_small):

        target_loss = self.target_loss(pred_eps, noise)

        self.log('train_G/target_loss', target_loss)
        target_weight = self.target_weight

        errG = target_loss * target_weight

        return errG

    @rank_zero_only
    def check_count(self, key, freq, update_count=True):
        if update_count:
            self.call_count[key] += 1

            if self.call_count[key] >= freq:
                self.call_count[key] = 0
                return True
            else:
                return False
        else:
            return self.call_count[key] + 1 >= freq

    def requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def schedule_G(self):
        sch_G = self.lr_schedulers()
        sch_G.step()

    def optimize_G(self):
        g_opt = self.optimizers()
        grad_clip = self.training_config.optimizing.grad_clip.G_opt

        if grad_clip is not None:
            self.clip_gradients(optimizer=g_opt, gradient_clip_val=grad_clip.value,
                                gradient_clip_algorithm=grad_clip.algorithm)

        g_opt.step()
        g_opt.zero_grad(set_to_none=True)

    def get_scheduler(self, optimizer, scheduler_params):
        if scheduler_params is not None:
            args = OmegaConf.to_container(scheduler_params, resolve=True)
            if 'SequentialLR' in args['_target_']:
                for inner_scheduler_args in args['schedulers']:
                    inner_scheduler_args['optimizer'] = optimizer
            args['optimizer'] = optimizer
            return instantiate(args)
        else:
            return None

    def configure_optimizers(self):
        opt_params = self.training_config.optimizing.optimizers

        optimizer_G = DefaultOptimizerConstructor(opt_params.optimizer_G,
                                                  opt_params.paramwise_cfg.optimizer_G)(self.netG)

        interval = self.training_config.optimizing.schedulers.interval
        assert interval == 'step', 'Another interval is not  supported'
        sched_params = self.training_config.optimizing.schedulers
        return [optimizer_G], [
            {'scheduler': self.get_scheduler(optimizer_G, sched_params.scheduler_G),
             'interval': interval},
        ]
