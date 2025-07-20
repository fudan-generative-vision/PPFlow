# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from models import PPFlow_models
from download import find_model
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from train_utils import parse_transport_args
import wandb_utils


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def initialize_layer_weights(source_weight, target_layer, repeat_factor, scale_factor=1.0, dim=1):
    """
    Initializes the weights of a target layer based on a source patchify/unpatchify weight tensor.

    Args:
        source_weight (torch.Tensor): The source weight tensor.
        target_layer (torch.nn.Module): The target layer to be initialized.
        repeat_factor (int): The number of times to repeat the weights.
        scale_factor (float, optional): The scaling factor for the source weights. Defaults to 1.0.
        dim (int, optional): The dimension along which to repeat the weights. Defaults to 1.
    """
    with torch.no_grad():
        # Scale and repeat the source weight
        target_weight = source_weight / scale_factor
        initialized_weight = target_weight.repeat_interleave(repeat_factor, dim=dim)
        # Copy the initialized weights to the target layer
        target_layer.weight.data.copy_(initialized_weight)

def initialize_layer_bias(source_bias, target_layer, repeat_factor=1, dim=0):
    """
    Initializes the bias of a target layer based on a source patchify/unpatchify bias tensor.

    Args:
        source_bias (torch.Tensor): The source bias tensor.
        target_layer (torch.nn.Module): The target layer to be initialized.
        repeat_factor (int, optional): The number of times to repeat the bias. Defaults to 1.
        dim (int, optional): The dimension along which to repeat the bias. Defaults to 0.
    """
    with torch.no_grad():
        if repeat_factor > 1:
            initialized_bias = source_bias.repeat_interleave(repeat_factor, dim=dim)
        else:
            initialized_bias = source_bias
        # Copy the initialized bias to the target layer
        target_layer.bias.data.copy_(initialized_bias)

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new PPFlow model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    local_batch_size = int(args.global_batch_size // dist.get_world_size())

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model
        experiment_name = f"{experiment_index:03d}-{model_string_name}-" \
                        f"{args.path_type}-{args.prediction}-{args.loss_weight}"
        experiment_dir = f"{args.results_dir}/{experiment_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        entity = os.environ["ENTITY"]
        project = os.environ["PROJECT"]
        if args.wandb:
            wandb_utils.initialize(args, entity, experiment_name, project)
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = PPFlow_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )

    # Note that parameter initialization is done within the PPFlow constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    if args.ckpt is not None:
        ckpt_path = args.ckpt
        state_dict = find_model(ckpt_path, load_all=True)
        model.load_state_dict(state_dict["model"])
        ema.load_state_dict(state_dict["ema"])
        opt.load_state_dict(state_dict["opt"])
        for state in opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        args = state_dict["args"]
    
    if args.initialize_from is not None:
        ckpt_path = args.initialize_from
        state_dict = find_model(ckpt_path)
        model.load_state_dict(state_dict, strict=False)
        ema.load_state_dict(state_dict, strict=False)

        # ======== 1. Patchify ========
        # Get source tensors
        source_proj_weight = state_dict["x_embedder.proj.weight"]
        source_proj_bias = state_dict["x_embedder.proj.bias"]

        # Initialize proj1
        initialize_layer_weights(source_proj_weight, model.x_embedder.proj1, repeat_factor=4, scale_factor=4.0, dim=1)
        initialize_layer_bias(source_proj_bias, model.x_embedder.proj1) # Bias is not repeated
        
        if args.patch_level == 3:
            # Initialize proj2
            initialize_layer_weights(source_proj_weight, model.x_embedder.proj2, repeat_factor=2, scale_factor=2.0, dim=1)
            initialize_layer_bias(source_proj_bias, model.x_embedder.proj2) # Bias is not repeated


        # ======== 2. Unpatchify ========
        # Get source tensors
        source_linear_weight = state_dict["final_layer.linear.weight"]
        source_linear_bias = state_dict["final_layer.linear.bias"]

        # Initialize linear1
        initialize_layer_weights(source_linear_weight, model.final_layer.linear1, repeat_factor=4, dim=0)
        initialize_layer_bias(source_linear_bias, model.final_layer.linear1, repeat_factor=4, dim=0)

        if args.patch_level == 3:
            # Initialize linear2
            initialize_layer_weights(source_linear_weight, model.final_layer.linear2, repeat_factor=2, dim=0)
            initialize_layer_bias(source_linear_bias, model.final_layer.linear2, repeat_factor=2, dim=0)

    requires_grad(ema, False)
    
    model = DDP(model.to(device), device_ids=[rank])
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )  # default: velocity; 
    transport_sampler = Sampler(transport)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"PPFlow Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    if args.ckpt is None:
        update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    # Labels to condition the model with (feel free to change):
    ys = torch.randint(1000, size=(local_batch_size,), device=device)
    use_cfg = any(cfg_scale >= 1.0 for cfg_scale in args.cfg_scales)
    # Create sampling noise:
    n = ys.size(0)
    zs = torch.randn(n, 4, latent_size, latent_size, device=device)

    # Setup classifier-free guidance:
    if use_cfg:
        zs = torch.cat([zs, zs], 0)
        y_null = torch.tensor([1000] * n, device=device)
        ys = torch.cat([ys, y_null], 0)
        sample_model_kwargs = dict(y=ys, cfg_scales=args.cfg_scales)
        model_fn = ema.forward_with_cfg
    else:
        sample_model_kwargs = dict(y=ys)
        model_fn = ema.forward

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            model_kwargs = dict(y=y)
            loss_dict = transport.training_losses(model, x, args.patch_level, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                if args.wandb:
                    wandb_utils.log(
                        { "train loss": avg_loss, "train steps/sec": steps_per_sec },
                        step=train_steps
                    )
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save SiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()
            
            if train_steps % args.sample_every == 0 and train_steps > 0:
                logger.info("Generating EMA samples...")
                sample_fn = transport_sampler.sample_ode() # default to ode sampling
                samples = sample_fn(zs, model_fn, **sample_model_kwargs)[-1]
                dist.barrier()

                if use_cfg: #remove null samples
                    samples, _ = samples.chunk(2, dim=0)
                samples = vae.decode(samples / 0.18215).sample
                out_samples = torch.zeros((args.global_batch_size, 3, args.image_size, args.image_size), device=device)
                dist.all_gather_into_tensor(out_samples, samples)
                if args.wandb:
                    wandb_utils.log_image(out_samples, train_steps)
                logging.info("Generating EMA samples done.")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train PPFlow-XL-2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(PPFlow_models.keys()), default="PPFlow_XL_2")
    parser.add_argument("--patch-level", type=int, choices=[2,3], default=2)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--sample-every", type=int, default=10_000)
    parser.add_argument("--cfg-scales", type=float, nargs='+', default=[4.0, 4.0],
                        help="A list of CFG scales for different time steps. Example: --cfg-scales 1.0 3.0 --cfg-scales 1.0 3.5 3.75")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a custom PPFlow checkpoint")
    parser.add_argument("--initialize_from", type=str, default="SiT-XL-2-256.pt",
                        help="Optional path to a pretrained SiT checkpoint")

    parse_transport_args(parser)
    args = parser.parse_args()
    main(args)
