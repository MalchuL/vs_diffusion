import copy
import logging
import math
import os.path
from collections import defaultdict
from itertools import chain
from pathlib import Path

import pytorch_lightning as pl
import torch
import torchvision
from omegaconf import OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from torch import autograd

from src.metrics.metric_collection import get_alignment_metrics
from src.models.components.ada_layers.augments_factory import get_augments
from src.models.components.ada_layers.torch_utils import misc
from src.models.components.autoencoder.distributions import DiagonalGaussianDistribution

from src.models.components.autoencoder.abstract_autoencoder import AbstractAutoEncoder
import torch.nn.functional as F
from src.models.components.losses.gan_loss import GANLoss
from src.models.components.losses.tv_loss import TVLoss
from src.models.components.utils.init_net import init_net
from src.optimizers.optimizer_constructor import DefaultOptimizerConstructor
from src.utils.instantiate import instantiate
from src.utils.load_pl_dict import load_dict
from src.utils.logging import log_pl_image


class KlAutoEncoderModule(pl.LightningModule):

    def __init__(self, model_config, training_config=None) -> None:
        super(KlAutoEncoderModule, self).__init__()
        self.config = model_config
        self.training_config = training_config
        self.save_hyperparameters('model_config', 'training_config')

        self.autoencoder: AbstractAutoEncoder = self.create_autoencoder()
        self.autoencoder_ema = copy.deepcopy(self.autoencoder).eval()

        logging.info(self.autoencoder)
        self.register_buffer('mean', torch.tensor(self.config.norm.mean).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor(self.config.norm.std).view(1, -1, 1, 1))

        self.img_size = self.config.img_size
        self.example_input_array = torch.randn(1, 3, self.img_size, self.img_size)

        self.embed_dim = self.config.embedding.embed_dim

        # AE embedding
        self.double_z = self.autoencoder.get_double_z()
        z_multiplier = 2 if self.double_z else 1
        z_channels = self.autoencoder.get_encoded_dim()
        self.quant_conv = torch.nn.Conv2d(z_channels * z_multiplier, self.embed_dim * z_multiplier, 1)
        self.post_quant_conv = torch.nn.Conv2d(self.embed_dim, z_channels, 1)
        self.distribution = DiagonalGaussianDistribution()

        self.register_buffer('cur_nimg', torch.zeros([], dtype=torch.long))
        self.register_buffer('num_iters', torch.zeros([], dtype=torch.long))

        if self.training_config is not None:
            self.use_sigmoid = self.training_config.use_sigmoid

            self.automatic_optimization = False
            self.call_count = defaultdict(int)

            self.netD = self.create_discriminator()
            logging.info(self.netD)

            adv_inner_loss = instantiate(self.training_config.losses.adv_criterion)
            self.adv_apply_sigmoid = self.training_config.losses.adv_apply_sigmoid
            self.adv_loss = GANLoss(criterion=adv_inner_loss,
                                    is_logit=not self.use_sigmoid or not self.adv_apply_sigmoid)

            self.content_loss = instantiate(self.training_config.losses.content_loss)

            logging.info(self.content_loss)
            self.clip_loss = instantiate(self.training_config.losses.clip_loss)
            self.tv_loss = TVLoss()
            self.target_loss = instantiate(self.training_config.losses.target_loss)

            # Weights block
            weights = self.training_config.losses.weights
            self.content_weight = weights.content_weight
            self.clip_weight = weights.clip_weight
            self.adv_weight = weights.adv_weight
            self.global_D_weight = weights.global_D_weight
            self.tv_weight = weights.tv_weight
            self.target_weight = weights.target_weight
            self.kl_weight = weights.kl_weight

            self.register_buffer('pl_mean', torch.zeros([]))

            # ADA
            self.ada_target = self.training_config.ada.ada_target
            self.register_buffer('ada_stats', torch.zeros([]))
            self.ada_gamma = self.training_config.ada.ada_gamma
            self.ada_interval = self.training_config.ada.ada_interval
            self.ada_kimg = self.training_config.ada.ada_kimg

            self.augment_pipe = get_augments(self.training_config.ada.name)

            self.valid_metrics = get_alignment_metrics(prefix='val/')  # For ema

    def create_autoencoder(self):
        autoencoder = instantiate(self.config.autoencoder)
        if self.training_config.initialization.pretrain_checkpoint_autoencoder:
            load_dict(autoencoder, 'autoencoder', self.training_config.initialization.pretrain_checkpoint_autoencoder)
        else:
            init_net(autoencoder, **self.training_config.initialization.init_autoencoder)
        return autoencoder

    def create_discriminator(self):
        netD = instantiate(self.training_config.netD)
        init_net(netD, **self.training_config.initialization.init_D)
        return netD

    def backward_mapping(self, real_tensor):
        return (real_tensor * self.std + self.mean)

    def encode(self, x):
        encoder = self.autoencoder.encoder if self.training else self.autoencoder_ema.encoder
        h = encoder(x)
        z = self.quant_conv(h)
        return z

    def decode(self, z):
        decoder = self.autoencoder.decoder if self.training else self.autoencoder_ema.decoder
        z = self.post_quant_conv(z)
        dec = decoder(z)
        return dec

    def forward(self, input, sample_posterior=False):
        z = self.encode(input)
        if self.double_z:
            mean, logvar = torch.chunk(z, 2, dim=1)
        else:
            mean = z
            logvar = 0

        z = self.distribution.sample(mean, logvar, determenistic=not sample_posterior)
        dec = self.decode(z)
        return dec, (mean, logvar)


    def apply_sigmoid(self, data):
        if self.use_sigmoid:
            return torch.sigmoid(data)

    def apply_sigmoid_to_adv(self, *tensors):
        if self.adv_apply_sigmoid and not self.use_sigmoid:
            return tuple(torch.sigmoid(tensor) if tensor is not None else None for tensor in tensors)
        else:
            return tensors

    def training_step(self, batch, batch_idx):
        if self.augment_pipe is not None:
            self.augment_pipe.train().requires_grad_(False)

        real = batch['image']

        bs = real.shape[0]
        gradient_accumulation = max(bs // self.training_config.inner_batch_size, 1)
        assert bs % self.training_config.inner_batch_size == 0 and bs >= self.training_config.inner_batch_size
        inner_batch_size = self.training_config.inner_batch_size

        temp = {}
        for i in range(gradient_accumulation):
            real_small = real[i * inner_batch_size: (i + 1) * inner_batch_size]

            ######################
            # Optimize Generator #
            ######################
            # This correct else does not converge
            self.autoencoder.train()
            self.netD.train()

            self.requires_grad(self.autoencoder, True)
            self.requires_grad(self.netD, False)

            fake, mean_logvar = self(real_small)

            loss_autoencoder = self.generator_loss(real_small, fake, mean_logvar, temp)

            preds = temp['preds']
            if i == 0:
                with torch.no_grad():
                    self.log_images(real_small, fake.detach(), preds=preds)
            self.log('loss_autoencoder', loss_autoencoder, prog_bar=True)
            self.manual_backward(loss_autoencoder / gradient_accumulation)

            ##########################
            # Optimize Discriminator #
            ##########################
            # This correct
            self.netD.train()
            self.autoencoder.eval()

            self.requires_grad(self.autoencoder, False)
            self.requires_grad(self.netD, True)

            loss_D = self.discriminator_loss(fake.detach(), real_small, temp)

            self.log('loss_D', loss_D, prog_bar=True)
            self.manual_backward(loss_D / gradient_accumulation)

        ########## Run optimization ################
        self.optimize_autoencoder()
        self.optimize_D()
        #########################################

        for i in range(gradient_accumulation):
            real_small = real[i * inner_batch_size: (i + 1) * inner_batch_size]
            ##########################
            # Reg generator
            ##########################
            g_pl_every = self.training_config.losses.pl_loss.g_pl_every
            if g_pl_every > 0 and self.check_count('G_reg_interval', g_pl_every,
                                                   update_count=i == gradient_accumulation - 1):
                self.autoencoder.train()
                self.requires_grad(self.autoencoder, True)
                G_reg_loss = self.generator_reg(real_small) * g_pl_every
                self.log('G_reg_loss', G_reg_loss, prog_bar=True)
                self.manual_backward(G_reg_loss / gradient_accumulation)

            ##########################
            # Reg discriminator
            ##########################
            d_reg_every: int = self.training_config.losses.r1_loss.d_reg_every
            if d_reg_every > 0 and self.check_count('D_reg_interval', d_reg_every,
                                                    update_count=i == gradient_accumulation - 1):
                self.netD.train()
                self.requires_grad(self.netD, True)
                D_reg_loss = self.discriminator_reg(real_small) * d_reg_every
                self.log('D_reg_loss', D_reg_loss, prog_bar=True)
                self.manual_backward(D_reg_loss / gradient_accumulation)

            #########################

        ########## Run optimization ################

        self.optimize_autoencoder()
        self.optimize_D()

        ########################################

        self.schedule_autoencoder()
        self.schedule_D()

        self.cur_nimg += bs
        self.log('cur_nimg', self.cur_nimg, prog_bar=True)
        self.num_iters += 1

    def on_validation_start(self) -> None:
        # Move metrics to cpu
        self.valid_metrics.cpu()

    def on_fit_start(self) -> None:
        # Create validation folder
        if not os.path.exists('output'):
            self.val_folder = Path('output')
            self.val_folder.mkdir(exist_ok=True, parents=True)

    def validation_step(self, batch, batch_nb):

        self.autoencoder.eval()
        real = batch['image']

        fake_ema, _ = self(real)

        # Compute metrics
        denormalized_cartoon = self.backward_mapping(real)
        denormalized_fake_ema = self.backward_mapping(fake_ema)

        self.valid_metrics.update(denormalized_fake_ema.cpu(), denormalized_cartoon.cpu())

        grid = torchvision.utils.make_grid(torch.cat([real, fake_ema], dim=0))
        grid = grid * torch.tensor(self.config.norm.std, dtype=grid.dtype, device=grid.device).view(-1, 1,
                                                                                                    1) + torch.tensor(
            self.config.norm.mean, dtype=grid.dtype, device=grid.device).view(-1, 1, 1)

        torchvision.utils.save_image(grid, str(self.val_folder / (str(batch_nb) + '.png')), nrow=1)

        return {}

    def update_aug_probs(self, D_logits):
        # Execute ADA heuristic.
        if self.check_count('ada_interval', self.ada_interval) and self.augment_pipe is not None:
            if self.use_sigmoid:
                D_logits = torch.logit(D_logits)
            # print('update_aug_probs', self.global_step)
            signs = D_logits.sign().mean()
            batch_size = D_logits.shape[0]
            self.ada_stats = self.ada_stats * self.ada_gamma + (1 - self.ada_gamma) * signs
            adjust = torch.sign(self.ada_stats - self.ada_target) * (batch_size * self.ada_interval) / (
                    self.ada_kimg * 1000)
            self.augment_pipe.p.copy_((self.augment_pipe.p + adjust).max(misc.constant(0, device=self.device)))
        if self.augment_pipe is not None:
            self.log('augment/augment_pipe_p', self.augment_pipe.p)
        self.log('augment/ada_stats', self.ada_stats)

    def on_validation_epoch_end(self) -> None:
        for metrics in [self.valid_metrics]:
            output = metrics.compute()
            self.log_dict(output)
            metrics.reset()

    @rank_zero_only
    def log_images(self, *images, preds=None, name='train_image'):
        # tensors [self.real, self.fake, preds, self.cartoon]
        images = list(images)
        if self.check_count('img_log_freq', self.training_config.logging.img_log_freq) or self.global_step in (0, 1):
            if preds is not None:
                if not self.training_config.use_sigmoid:
                    preds = torch.sigmoid(preds)
                preds = F.upsample_bilinear(preds, size=images[0].shape[2:4])
                preds = (preds - self.mean) / self.std
                images += [preds]
            out_image = torch.cat(images, dim=0)

            grid = torchvision.utils.make_grid(out_image, nrow=len(images[0]))
            grid = self.backward_mapping(grid)[0]
            grid = torch.clamp(grid, 0.0, 1.0)

            log_pl_image(self, grid, name=name)

    @staticmethod
    def read_discriminator_output(pred):
        if isinstance(pred, (tuple, list)):
            return pred
        else:
            return (pred, None)

    @staticmethod
    def read_generator_output(pred):
        if isinstance(pred, (tuple, list)):
            return pred[0]
        else:
            pred

    def generator_loss(self, real, fake, mean_logvar, out=None):
        real = self.backward_mapping(real)
        if self.augment_pipe is not None:
            augment_fake = self.augment_pipe(fake)
        else:
            augment_fake = fake
        augment_fake = self.backward_mapping(augment_fake)
        fake = self.backward_mapping(fake)

        generated_pred, generated_pred_stat = self.read_discriminator_output(self.netD(augment_fake))
        if out is not None:
            with torch.no_grad():
                out['preds'] = generated_pred.mean(1, keepdim=True)

        generated_pred, generated_pred_stat = self.apply_sigmoid_to_adv(generated_pred,
                                                                        generated_pred_stat)  # Apply sigmoid to input to adv_loss

        content_loss = self.content_loss(fake, real)  # We apply content loss for aligned images in dataset
        self.log('train_autoencoder/content_loss', content_loss)
        content_loss = content_loss * self.content_weight

        fake_loss = self.adv_loss(generated_pred, True)
        if generated_pred_stat is None:
            fake_stat_loss = 0
        else:
            fake_stat_loss = self.adv_loss(generated_pred_stat, True)
        clip_loss = self.clip_loss(fake)
        tv_loss = self.tv_loss(fake)
        target_loss = self.target_loss(fake, real)

        mean, logvar = mean_logvar
        kl_loss = self.distribution.kl(mean, logvar)
        kl_weight = self.kl_weight

        self.log('train_autoencoder/kl_loss', kl_loss, prog_bar=True)
        self.log('train_autoencoder/clip_loss', clip_loss)
        self.log('train_autoencoder/tv_loss', tv_loss)
        self.log('train_autoencoder/target_loss', target_loss)
        self.log('train_autoencoder/G_fake_loss', fake_loss)
        self.log('train_autoencoder/G_fake_stat_loss', fake_stat_loss)
        adv_weight = self.adv_weight

        tv_weight = self.tv_weight

        target_weight = self.target_weight

        global_D_weight = self.global_D_weight if generated_pred_stat is not None else 0

        errG = (fake_loss * (1 - global_D_weight) + fake_stat_loss * (
            global_D_weight)) * adv_weight + content_loss + clip_loss * self.clip_weight + \
               tv_loss * tv_weight + target_loss * target_weight + kl_loss * kl_weight

        return errG

    def consistency_output_loss(self, real, model):
        if self.consistency_loss_weight > 0:
            consistency_loss = self.consistency_loss(real, model)
            self.log('train_autoencoder/consistency_loss', consistency_loss)
            return consistency_loss * self.consistency_loss_weight
        else:
            return 0.0

    def discriminator_loss(self, fake, cartoon, out=None):

        tensor_dict = {'fake': (fake, False),
                       'cartoon': (cartoon, True)}
        patch_losses = []
        global_losses = []
        for img_name, (img_tensor, is_cartoon) in tensor_dict.items():
            if self.augment_pipe is not None:
                img_tensor = self.augment_pipe(img_tensor)
            # Backward map normalization to [0..1]
            img_tensor = self.backward_mapping(img_tensor)
            img_pred_patch, img_pred_stat = self.apply_sigmoid_to_adv(*self.netD(img_tensor))

            if img_name == 'cartoon':
                self.update_aug_probs(img_pred_patch)

            if out is not None and img_name == 'fake':
                out['preds'] = img_pred_patch
            # Patch loss
            img_patch_loss = self.adv_loss(img_pred_patch, is_cartoon)
            patch_losses.append(img_patch_loss)

            if not self.use_sigmoid:
                img_pred_patch = torch.sigmoid(img_pred_patch)
            for group_layer in range(img_pred_patch.shape[1]):
                self.log(f'train_D/patch_{img_name}_D_probs_{group_layer}_group',
                         1 - img_pred_patch[:, group_layer].mean() if is_cartoon else img_pred_patch[:,
                                                                                      group_layer].mean())

            # Global loss
            if img_pred_stat is None:
                global_losses.append(0)
            else:
                img_global_loss = self.adv_loss(img_pred_stat, is_cartoon)
                global_losses.append(img_global_loss)

        patch_loss = sum(patch_losses)
        global_loss = sum(global_losses)
        self.log('train_D/patch_D_loss', patch_loss)
        self.log('train_D/global_D_loss', global_loss)
        global_D_weight = self.global_D_weight if global_loss > 0 else 0
        return (patch_loss * (1 - global_D_weight) + global_loss * (global_D_weight)) * self.adv_weight

    @staticmethod
    def d_r1_loss(real_pred, real_img):
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
        grad_penalty = grad_real.pow(2).mean()

        return grad_penalty

    def generator_reg(self, real):
        pl_weight = self.training_config.losses.pl_loss.pl_weight
        pl_batch_shrink = self.training_config.losses.pl_loss.pl_batch_shrink
        pl_decay = self.training_config.losses.pl_loss.pl_decay

        real_input = real.clone().detach()
        real_input.requires_grad = True

        batch_size = round(real_input.shape[0] // pl_batch_shrink)
        gen_img, _ = self(real_input[:batch_size])
        pl_noise = torch.randn_like(gen_img) / math.sqrt(gen_img.shape[2] * gen_img.shape[3])

        pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[real_input], create_graph=True,
                                       only_inputs=True)[0]
        pl_lengths = pl_grads.square().mean().sqrt()
        pl_mean = self.pl_mean.lerp(pl_lengths.mean(), pl_decay)
        self.pl_mean.copy_(pl_mean.detach())
        pl_penalty = (pl_lengths - pl_mean).square()

        self.log('train_autoencoder/pl_loss', pl_penalty, prog_bar=False)
        loss_autoencoderpl = pl_penalty * pl_weight
        loss_autoencoderpl = (gen_img[:, 0, 0, 0] * 0 + loss_autoencoderpl).mean()
        return loss_autoencoderpl

    def discriminator_reg(self, real_img):
        r1 = self.training_config.losses.r1_loss.r1

        real_img = self.backward_mapping(real_img)
        real_img.requires_grad = True

        real_pred, real_pred_stat = self.netD(real_img)
        r1_loss = self.d_r1_loss(real_pred, real_img) * (1 - self.global_D_weight) + \
                  self.d_r1_loss(real_pred_stat, real_img) * (self.global_D_weight)

        self.netD.zero_grad()
        full_loss = r1 / 2 * r1_loss + 0 * real_pred.mean()
        self.log('train_D/full_r1_loss', full_loss, prog_bar=True)
        self.log('train_D/r1_loss_simple', r1_loss, prog_bar=True)

        return full_loss

    def on_train_batch_end(self, outputs, batch, batch_idx: int):

        with torch.inference_mode():
            ema_kimg = self.training_config.ema.ema_kimg
            ema_rampup = self.training_config.ema.ema_rampup
            ema_nimg = ema_kimg * 1000
            ema_nimg = min(ema_nimg, self.cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (1 / max(ema_nimg, 1e-8))
            for p_ema, p in zip(self.autoencoder_ema.parameters(), self.autoencoder.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(self.autoencoder_ema.buffers(), self.autoencoder.buffers()):
                b_ema.copy_(b.float().lerp(b_ema.float(), ema_beta).type_as(b))

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

    def schedule_autoencoder(self):
        sch_autoencoder, _ = self.lr_schedulers()
        sch_autoencoder.step()

    def schedule_D(self):
        _, sch_D = self.lr_schedulers()
        sch_D.step()

    def optimize_D(self):
        _, d_opt = self.optimizers()

        grad_clip = self.training_config.optimizing.grad_clip.D_opt

        if grad_clip is not None:
            self.clip_gradients(optimizer=d_opt, gradient_clip_val=grad_clip.value,
                                gradient_clip_algorithm=grad_clip.algorithm)
        d_opt.step()
        d_opt.zero_grad(set_to_none=True)

    def optimize_autoencoder(self):
        autoencoder_opt, _ = self.optimizers()
        grad_clip = self.training_config.optimizing.grad_clip.G_opt

        if grad_clip is not None:
            self.clip_gradients(optimizer=autoencoder_opt, gradient_clip_val=grad_clip.value,
                                gradient_clip_algorithm=grad_clip.algorithm)

        autoencoder_opt.step()
        autoencoder_opt.zero_grad(set_to_none=True)

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
        optimizer_autoencoder = DefaultOptimizerConstructor(opt_params.optimizer_autoencoder,
                                                            opt_params.paramwise_cfg.optimizer_autoencoder)(
            self.autoencoder)
        optimizer_autoencoder.add_param_group(
            {'params': chain(self.quant_conv.parameters(), self.post_quant_conv.parameters())})
        optimizer_D = DefaultOptimizerConstructor(opt_params.optimizer_D,
                                                  opt_params.paramwise_cfg.optimizer_D)(self.netD)
        interval = self.training_config.optimizing.schedulers.interval
        assert interval == 'step', 'Another interval is not  supported'
        sched_params = self.training_config.optimizing.schedulers
        return [optimizer_autoencoder, optimizer_D], [
            {'scheduler': self.get_scheduler(optimizer_autoencoder, sched_params.scheduler_autoencoder),
             'interval': interval},
            {'scheduler': self.get_scheduler(optimizer_D, sched_params.scheduler_D),
             'interval': interval}]
