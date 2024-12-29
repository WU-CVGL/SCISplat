import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import wandb
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# import torch.distributed as dist
import tqdm
import tyro
import viser
import nerfview
from pose_viewer import PoseViewer
from datasets.colmap_dataparser import ColmapParser
from datasets.colmap import Dataset
from datasets.traj import generate_interpolated_path
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils import (
    PoseOptModule,
    AppearanceOptModule,
    CameraOptModule,
    set_random_seed,
)
from gsplat import quat_scale_to_covar_preci
from gsplat.rendering import rasterization
from gsplat.relocation import compute_relocation
from gsplat.cuda_legacy._torch_impl import scale_rot_to_cov3d
from simple_trainer import create_splats_with_optimizers
from trajectory_evaluation import (
    plot_trajectories2D,
    align_umeyama,
    compute_absolute_error_translation,
    fig_to_array,
    transformation_to_tum_format,
    save_to_tum_file,
)
import pypose as pp
from schedulers import get_exponential_decay_scheduler


@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # if debug mode disable wandb logging
    debug: bool = False
    # Path to the .pt file. If provide, it will skip training and render a video
    ckpt: Optional[str] = None

    # whether to perform novel view evaluation
    novel_view: bool = False

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 1  # cannot downsample w/ SCI
    # How much to scale the camera origins by
    scale_factor: float = 1.0
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8  # not used in our case
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # downsample rate for pointcloud
    downsample_rate: int = 1
    # downsample threshold for initial pointcloud
    downsample_cap: int = 1000

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(
        default_factory=lambda: [
            # 1,
            # 5,
            # 10,
            # 50,
            100,
            # 150,
            # 200,
            # 250,
            # 300,
            500,
            1_000,
            1_500,
            2_000,
            2_500,
            3_000,
            3_500,
            4_000,
            4_500,
            5_000,
            5_500,
            6_000,
            6_500,
            7_000,
            7_500,
            8_000,
            8_500,
            9_000,
            9_500,
            10_000,
            12_000,
            15_000,
            16_500,
            17_000,
            18_500,
            20_000,
            23_000,
            25_000,
            27_000,
            30_000,
        ]
    )
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.5
    # Initial scale of GS
    init_scale: float = 0.1
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Maximum number of GSs.
    cap_max: int = 1_000_000
    # MCMC samping noise learning rate
    noise_lr: float = 5e5
    # Opacity regularization
    opacity_reg: float = 0.01
    # Scale regularization
    scale_reg: float = 0.01

    # warmup steps for learning means3d
    means3d_warmup_steps: int = 0
    # warmup steps for learning scales
    scales_warmup_steps: int = 0
    # warmup steps for quats
    quats_warmup_steps: int = 0
    # warmup steps for learning opacities
    opacities_warmup_steps: int = 0

    # Start refining GSs after this iteration
    refine_start_iter: int = 500
    # Stop refining GSs after this iteration
    refine_stop_iter: int = 25_000
    # Refine GSs every this steps
    refine_every: int = 100

    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use absolute gradient for pruning. This typically requires larger --grow_grad2d, e.g., 0.0008 or 0.0006
    absgrad: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # Whether to refine upon GT poses
    pose_refine: bool = False
    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Final learning rate for camera optimization
    pose_opt_lr_final: float = 5e-6
    # Warmup steps for pose learning rate schedule
    pose_opt_lr_warmup_steps: int = 0
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0
    # pose optimization gradient clipping (0.4 works for cozy2room bezier)
    pose_grad_clip: float = 0.0
    # pose optimization gradient accumulation
    pose_grad_accum: int = 1
    # different trajectory representations
    traj_mode: str = "bezier"
    # degree of Bezier curve
    bezier_degree: int = 8
    # initial noise for pose
    initial_noise: float = 1e-5

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)
        self.refine_start_iter = int(self.refine_start_iter * factor)
        self.refine_stop_iter = int(self.refine_stop_iter * factor)
        self.refine_every = int(self.refine_every * factor)


class Runner:
    """Engine for training and testing."""

    def __init__(self, cfg: Config) -> None:
        set_random_seed(42)

        self.cfg = cfg
        self.device = "cuda"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        # store all rendered images
        self.archive_dir = f"{cfg.result_dir}/archive"
        os.makedirs(self.archive_dir, exist_ok=True)
        # directory to store the best
        self.best_dir = f"{cfg.result_dir}/best"
        os.makedirs(self.best_dir, exist_ok=True)

        self.best_psnr = 0

        # Tensorboard
        # self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")
        wandb.init(  # Set the project where this run will be logged
            project="SCI-gsplat",
            name=os.path.normpath(os.path.basename(cfg.result_dir)),
            # Track hyperparameters and run metadata
            config=cfg,
            dir="/tmp/wandb",
            mode="disabled" if cfg.debug else "online",
        )

        # Load data: Training data should contain initial points and colors.
        # self.parser = Parser(
        #     data_dir=cfg.data_dir,
        #     factor=cfg.data_factor,
        #     normalize=False, #
        #     test_every=cfg.test_every,
        # )

        self.parser = ColmapParser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            scale_factor=cfg.scale_factor,
            downsample_rate=cfg.downsample_rate,
            downsample_cap=cfg.downsample_cap,
        )

        self.num_imgs = len(self.parser.image_names)

        self.trainset = Dataset(
            self.parser,
            split="all",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
        )
        self.valset = Dataset(self.parser, split="all")

        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
        )
        print("Model initialized. Number of GS:", len(self.splats["means3d"]))

        self.pose_optimizers = []
        if cfg.pose_opt:
            # self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            # self.pose_adjust.zero_init()
            # self.pose_optimizers = [
            #     torch.optim.Adam(
            #         self.pose_adjust.parameters(),
            #         lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
            #         weight_decay=cfg.pose_opt_reg,
            #     )
            # ]

            self.pose_adjust = PoseOptModule(
                self.parser.camtoworlds,
                cfg.traj_mode,
                cfg.bezier_degree,
                cfg.initial_noise,
                cfg.pose_refine,
                cfg.pose_noise,
            ).to(self.device)
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr
                    * math.sqrt(
                        cfg.batch_size
                    ),  # NOTE: don't know if scaling should be applied
                    weight_decay=cfg.pose_opt_reg,
                )
            ]

            wandb.watch(self.pose_adjust, log_freq=100)

        # if cfg.pose_noise > 0.0:
        # self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
        # self.pose_perturb.random_init(cfg.pose_noise)

        self.app_optimizers = []
        if cfg.app_opt:
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(
            self.device
        )

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            # self.viewer = nerfview.Viewer(
            #     server=self.server,
            #     render_fn=self._viewer_render_fn,
            #     mode="training",
            # )
            self.viewer = PoseViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means3d"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=self.cfg.absgrad,
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            **kwargs,
        )
        return render_colors, render_alphas, info

    def train(self):
        cfg = self.cfg
        device = self.device

        # Dump cfg.
        with open(f"{cfg.result_dir}/cfg.json", "w") as f:
            json.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        # schedulers = [
        #     # means3d has a learning rate schedule, that end at 0.01 of the initial value
        #     torch.optim.lr_scheduler.ExponentialLR(
        #         self.optimizers[0], gamma=0.01 ** (1.0 / max_steps)
        #     ),
        # ]

        means3d_init_lr = self.optimizers[0].param_groups[0]["lr"]
        means3d_final_lr = 0.01 * means3d_init_lr
        means3d_scheduler = get_exponential_decay_scheduler(
            self.optimizers[0],
            means3d_init_lr,
            means3d_final_lr,
            max_steps,
            lr_pre_warmup=0,
            warmup_steps=cfg.means3d_warmup_steps,
        )

        schedulers = [
            # means3d has a learning rate schedule, that end at 0.01 of the initial value
            means3d_scheduler
        ]

        if cfg.scales_warmup_steps > 0:
            # try to add scheduler also for other attributes of gaussians
            scales_init_lr = self.optimizers[1].param_groups[0]["lr"]
            scales_scheduler = get_exponential_decay_scheduler(
                self.optimizers[1],
                scales_init_lr,
                max_steps=max_steps,
                lr_pre_warmup=0,
                warmup_steps=cfg.scales_warmup_steps,
            )

            schedulers.append(scales_scheduler)

        if cfg.quats_warmup_steps > 0:
            # try to add scheduler also for other attributes of gaussians
            quats_init_lr = self.optimizers[2].param_groups[0]["lr"]
            quats_scheduler = get_exponential_decay_scheduler(
                self.optimizers[2],
                quats_init_lr,
                max_steps=max_steps,
                lr_pre_warmup=0,
                warmup_steps=cfg.quats_warmup_steps,
            )

            schedulers.append(quats_scheduler)

        if cfg.opacities_warmup_steps > 0:
            opacities_init_lr = self.optimizers[3].param_groups[0]["lr"]
            opacities_scheduler = get_exponential_decay_scheduler(
                self.optimizers[3],
                opacities_init_lr,
                max_steps=max_steps,
                lr_pre_warmup=0,
                warmup_steps=cfg.opacities_warmup_steps,
            )
            schedulers.append(opacities_scheduler)

        if cfg.pose_opt:
            if cfg.pose_refine:
                # pose optimization has a learning rate schedule
                # default decay schedule
                # pose_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                #     self.pose_optimizers[0], gamma=0.01**(1.0 / max_steps))
                # NOTE: better for hotdog scene
                pose_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.005 ** (1.0 / max_steps)
                )

                schedulers.append(pose_scheduler)
            else:
                # pose_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                #         self.pose_optimizers[0], gamma=0.0005 ** (1.0 / max_steps))
                # schedulers.append(pose_scheduler)

                pose_scheduler = get_exponential_decay_scheduler(
                    self.pose_optimizers[0],
                    cfg.pose_opt_lr,
                    cfg.pose_opt_lr_final,
                    max_steps,
                    lr_pre_warmup=0,
                    warmup_steps=cfg.pose_opt_lr_warmup_steps,
                )
                schedulers.append(pose_scheduler)

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.num_imgs,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )

        self._init_viewer_state()

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            for data in trainloader:
                camtoworlds = camtoworlds_gt = data["camtoworld"].to(
                    device
                )  # [N, 4, 4]
                Ks = data["K"].to(device)  # [N, 3, 3]
                pixels = data["image"].to(device) / 255.0  # [N, H, W, 3]
                num_train_rays_per_step = (
                    pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
                )
                image_ids = data["image_id"].to(device)
                if cfg.depth_loss:
                    points = data["points"].to(device)  # [1, M, 2]
                    depths_gt = data["depths"].to(device)  # [1, M]

                height, width = pixels.shape[1:3]

                if cfg.pose_opt:
                    camtoworlds = self.pose_adjust.get_poses().to(torch.float32)

                # sh schedule
                sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

                # forward
                renders, alphas, info = self.rasterize_splats(
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=sh_degree_to_use,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    image_ids=image_ids,
                    render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                )
                if renders.shape[-1] == 4:
                    colors, depths = renders[..., 0:3], renders[..., 3:4]
                else:
                    colors, depths = renders, None

                if cfg.random_bkgd:
                    bkgd = torch.rand(1, 3, device=device)
                    colors = colors + bkgd * (1.0 - alphas)

                if "mask" in data:
                    mask = data["mask"].to(device)
                    assert mask.shape[:2] == pixels.shape[:2] == colors.shape[:2]
                    pixels = pixels * mask
                    colors = colors * mask

            if hasattr(self.parser, "comp_img"):
                comp_gt_img = self.parser.comp_img.to(device)
            else:
                comp_gt_img = pixels.sum(0).unsqueeze(0)

            comp_img = colors.sum(0).unsqueeze(0)

            # loss
            l1loss = F.l1_loss(comp_img, comp_gt_img)
            ssimloss = 1.0 - self.ssim(
                comp_gt_img.permute(0, 3, 1, 2), comp_img.permute(0, 3, 1, 2)
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

            if cfg.depth_loss:
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depths_gt  # [1, M]
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda

            loss = (
                loss
                + cfg.opacity_reg
                * torch.abs(torch.sigmoid(self.splats["opacities"])).mean()
            )
            loss = (
                loss
                + cfg.scale_reg * torch.abs(torch.exp(self.splats["scales"])).mean()
            )

            loss.backward()

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            if cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                # self.writer.add_scalar("train/loss", loss.item(), step)
                # self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                # self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                # self.writer.add_scalar(
                #     "train/num_GS", len(self.splats["means3d"]), step
                # )
                # self.writer.add_scalar("train/mem", mem, step)
                # # monitor pose learning rate
                # self.writer.add_scalar("train/poseLR", pose_scheduler.get_last_lr()[0], step)

                # log w/ wandb
                wandb.log({"train/loss": loss.item()}, step=step)
                wandb.log({"train/l1loss": l1loss.item()}, step=step)
                wandb.log({"train/ssimloss": ssimloss.item()}, step=step)
                wandb.log({"train/num_GS": len(self.splats["means3d"])}, step=step)
                wandb.log({"train/mem": mem}, step=step)
                wandb.log(
                    {"learning_rate/means3d": schedulers[0].get_last_lr()[0]}, step=step
                )
                if cfg.pose_opt:
                    wandb.log(
                        {"learning_rate/pose": pose_scheduler.get_last_lr()[0]},
                        step=step,
                    )
                if cfg.scales_warmup_steps > 0:
                    wandb.log(
                        {"learning_rate/scales": scales_scheduler.get_last_lr()[0]},
                        step=step,
                    )
                if cfg.quats_warmup_steps > 0:
                    wandb.log(
                        {"learning_rate/quats": quats_scheduler.get_last_lr()[0]},
                        step=step,
                    )
                if cfg.opacities_warmup_steps > 0:
                    wandb.log(
                        {
                            "learning_rate/opacities": opacities_scheduler.get_last_lr()[
                                0
                            ]
                        },
                        step=step,
                    )

                # monitor ATE
                if cfg.pose_opt and hasattr(self.parser, "camtoworlds_gt"):
                    self.visualize_traj(step)

                if cfg.depth_loss:
                    # self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                    wandb.log({"train/depthloss": depthloss.item()}, step=step)

                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    # self.writer.add_image("train/render", canvas, step)
                    # NOTE: this does not work well with wandb yet
                    wandb.log({"render": wandb.Image(canvas)}, step=step)

                # self.writer.flush()

            # edit GSs
            if step < cfg.refine_stop_iter:
                if step > cfg.refine_start_iter and step % cfg.refine_every == 0:
                    num_relocated_gs = self.relocate_gs()
                    print(f"Step {step}: Relocated {num_relocated_gs} GSs.")

                    num_new_gs = self.add_new_gs(cfg.cap_max)
                    print(
                        f"Step {step}: Added {num_new_gs} GSs. Now having {len(self.splats['means3d'])} GSs."
                    )

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            # optimize
            for optimizer in self.optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if cfg.pose_opt:
                if (
                    (step + 1) % cfg.pose_grad_accum == 0
                ):  # omit the last incomplete accumulation round
                    # scale the gradient of params in pose_adjust
                    for param in self.pose_adjust.parameters():
                        param.grad /= cfg.pose_grad_accum

                    if cfg.pose_grad_clip != 0:
                        torch.nn.utils.clip_grad_value_(
                            self.pose_adjust.parameters(), clip_value=cfg.pose_grad_clip
                        )

                    # optimizing poses (support gradient clipping w/ value, gradient accumulation)
                    for optimizer in self.pose_optimizers:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            for scheduler in schedulers:
                scheduler.step()

            # add noise to GSs
            last_lr = schedulers[0].get_last_lr()[0]
            self.add_noise_to_gs(last_lr)

            # save checkpoint
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means3d"]),
                }
                print("Step: ", step, stats)
                with open(f"{self.stats_dir}/train_step{step:04d}.json", "w") as f:
                    json.dump(stats, f)
                torch.save(
                    {
                        "step": step,
                        "splats": self.splats.state_dict(),
                    },
                    f"{self.ckpt_dir}/ckpt_{step}.pt",
                )

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps] or step == max_steps - 1:
                self.eval(step)
                # self.render_traj(step)
                self.render_traj_all(step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)
                # Update camera poses, this only works for the SCI case for now
                self.viewer.update_camera_poses(camtoworlds)

    @torch.no_grad()
    def relocate_gs(self, min_opacity: float = 0.005) -> int:
        dead_mask = torch.sigmoid(self.splats["opacities"]) <= min_opacity
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = (~dead_mask).nonzero(as_tuple=True)[0]
        num_gs = len(dead_indices)
        if num_gs <= 0:
            return num_gs

        # Sample for new GSs
        eps = torch.finfo(torch.float32).eps
        probs = torch.sigmoid(self.splats["opacities"])[alive_indices]
        probs = probs / (probs.sum() + eps)
        sampled_idxs = torch.multinomial(probs, num_gs, replacement=True)
        sampled_idxs = alive_indices[sampled_idxs]
        new_opacities, new_scales = compute_relocation(
            opacities=torch.sigmoid(self.splats["opacities"])[sampled_idxs],
            scales=torch.exp(self.splats["scales"])[sampled_idxs],
            ratios=torch.bincount(sampled_idxs)[sampled_idxs] + 1,
        )
        new_opacities = torch.clamp(new_opacities, max=1.0 - eps, min=min_opacity)
        self.splats["opacities"][sampled_idxs] = torch.logit(new_opacities)
        self.splats["scales"][sampled_idxs] = torch.log(new_scales)

        # Update splats and optimizers
        for k in self.splats.keys():
            self.splats[k][dead_indices] = self.splats[k][sampled_idxs]
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        p_state[key][sampled_idxs] = 0
                p_new = torch.nn.Parameter(self.splats[name])
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[name] = p_new
        torch.cuda.empty_cache()
        return num_gs

    @torch.no_grad()
    def add_new_gs(self, cap_max: int, min_opacity: float = 0.005) -> int:
        current_num_points = len(self.splats["means3d"])
        target_num = min(cap_max, int(1.05 * current_num_points))
        num_gs = max(0, target_num - current_num_points)
        if num_gs <= 0:
            return num_gs

        # Sample for new GSs
        eps = torch.finfo(torch.float32).eps
        probs = torch.sigmoid(self.splats["opacities"])
        probs = probs / (probs.sum() + eps)
        sampled_idxs = torch.multinomial(probs, num_gs, replacement=True)
        new_opacities, new_scales = compute_relocation(
            opacities=torch.sigmoid(self.splats["opacities"])[sampled_idxs],
            scales=torch.exp(self.splats["scales"])[sampled_idxs],
            ratios=torch.bincount(sampled_idxs)[sampled_idxs] + 1,
        )
        new_opacities = torch.clamp(new_opacities, max=1.0 - eps, min=min_opacity)
        self.splats["opacities"][sampled_idxs] = torch.logit(new_opacities)
        self.splats["scales"][sampled_idxs] = torch.log(new_scales)

        # Update splats and optimizers
        for k in self.splats.keys():
            self.splats[k] = torch.cat([self.splats[k], self.splats[k][sampled_idxs]])
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        v = p_state[key]
                        v_new = torch.zeros(
                            (len(sampled_idxs), *v.shape[1:]), device=self.device
                        )
                        p_state[key] = torch.cat([v, v_new])
                p_new = torch.nn.Parameter(self.splats[name])
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[name] = p_new
        torch.cuda.empty_cache()
        return num_gs

    @torch.no_grad()
    def add_noise_to_gs(self, last_lr):
        opacities = torch.sigmoid(self.splats["opacities"])
        scales = torch.exp(self.splats["scales"])
        actual_covariance, _ = quat_scale_to_covar_preci(
            self.splats["quats"],
            scales,
            compute_covar=True,
            compute_preci=False,
            triu=False,
        )

        def op_sigmoid(x, k=100, x0=0.995):
            return 1 / (1 + torch.exp(-k * (x - x0)))

        noise = (
            torch.randn_like(self.splats["means3d"])
            * (op_sigmoid(1 - opacities)).unsqueeze(-1)
            * cfg.noise_lr
            * last_lr
        )
        noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
        self.splats["means3d"].add_(noise)

    @torch.no_grad()
    def visualize_traj(self, step: int):
        # get ground truth trajectory
        camtoworlds_gt = self.parser.camtoworlds_gt

        # get estimated trajectory
        camtoworlds = self.pose_adjust.get_poses().to(torch.float32).cpu().numpy()

        # align them
        traj_gt = camtoworlds_gt[:, :3, -1]
        traj = camtoworlds[:, :3, -1]

        # log their 3D trajectories to writer as an image
        s, R, t = align_umeyama(traj_gt, traj)
        traj_aligned = (s * (R @ traj.T)).T + t

        # plot them post alignment
        fig = plot_trajectories2D(traj_gt, traj_aligned)
        img_pil, img_array = fig_to_array(fig)
        # self.writer.add_image('train/trajectories', img_array, step, dataformats='HWC')
        # wandb.log({"train/trajectories": fig}, step=step)
        wandb.log(
            {"trajectories": wandb.Image(img_pil)}, step=step
        )  # NOTE: cannot have / before media
        # wandb.log({"train/trajectories": wandb.Image(img_array)})
        plt.close(fig)

        # compute ATE log to writer
        # NOTE: this quantity is fine for now, but needs to be double-checked if reported in paper
        ate = compute_absolute_error_translation(traj_gt, traj_aligned)
        # self.writer.add_scalar("train/ATE", ate.item(), step)
        wandb.log({"train/ATE": ate.item()}, step=step)

    @torch.no_grad()
    def eval(self, step: int):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )

        if self.cfg.pose_opt:
            camtoworlds_all = self.pose_adjust.get_poses().to(
                torch.float32
            )  # (8, 4, 4)

        # hardcoded novel views according to Yunhao Li
        if self.cfg.novel_view:
            camtoworlds_all_se3 = self.pose_adjust.pose_control_knots
            Tstart = camtoworlds_all_se3[0]
            T2 = camtoworlds_all_se3[1]
            T3 = camtoworlds_all_se3[2]
            T4 = camtoworlds_all_se3[3]
            T5 = camtoworlds_all_se3[4]
            T6 = camtoworlds_all_se3[5]
            T7 = camtoworlds_all_se3[6]
            Tend = camtoworlds_all_se3[-1]
            if "cozy2room" in cfg.result_dir:
                Tnovel = 0.8 * Tstart + 0.2 * Tend
            elif "factory" in cfg.result_dir:
                Tnovel = 0.6 * Tstart + 0.4 * Tend
            elif "tanabata" in cfg.result_dir:
                Tnovel = 0.3 * Tstart + 0.7 * Tend
            elif "vender" in cfg.result_dir:
                Tnovel = 0.5 * Tstart + 0.5 * Tend
            elif "hotdog" in cfg.result_dir:
                # Tnovel = 0.5 * T5 + 0.5 * T6
                # Tnovel = 0.5 * T3 + 0.5 * T4
                Tnovel = 0.5 * T4 + 0.5 * T5
            elif "airplants" in cfg.result_dir:
                # import pdb; pdb.set_trace()
                Tnovel = (T2 + T3 + T4 + T5 + T6)/5
            else:
                raise ValueError("NVS evaluation not supported for this scene!")
            camtoworlds_novel = (
                pp.se3(Tnovel).matrix().to(torch.float32).unsqueeze(0).to(device)
            )

        ellipse_time = 0
        metrics = {"psnr": [], "ssim": [], "lpips": []}

        rgb_buffer = []
        depth_buffer = []

        for i, data in enumerate(valloader):
            if self.cfg.pose_opt:
                camtoworlds = camtoworlds_all[i].unsqueeze(0).to(device)  # (1, 4, 4)
            else:
                camtoworlds = data["camtoworld"].to(device)

            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)
            depths = renders[..., 3:4]
            depths = (depths - depths.min()) / (depths.max() - depths.min())

            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            rgb_buffer.append(colors.squeeze(0).cpu().numpy())
            depth_buffer.append(depths.squeeze(0).repeat(1, 1, 3).cpu().numpy())

            # write images
            canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}.png", (canvas * 255).astype(np.uint8)
            )

            color_save = colors.squeeze(0).cpu().numpy()
            imageio.imwrite(
                f"{self.archive_dir}/val_{step}_{i:04d}.png",
                (color_save * 255).astype(np.uint8),
            )

            pixels = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
            colors = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
            metrics["psnr"].append(self.psnr(colors, pixels))
            metrics["ssim"].append(self.ssim(colors, pixels))
            metrics["lpips"].append(self.lpips(colors, pixels))

        if cfg.novel_view:
            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds_novel,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            color_save_novel_view = torch.clamp(renders[..., 0:3], 0.0, 1.0)
            color_save_novel_view = color_save_novel_view.squeeze(0).cpu().numpy()
            imageio.imwrite(
                f"{self.archive_dir}/val_{step}_novel_view.png",
                (color_save_novel_view * 255).astype(np.uint8),
            )

        ellipse_time /= len(valloader)

        psnr = torch.stack(metrics["psnr"]).mean()
        ssim = torch.stack(metrics["ssim"]).mean()
        lpips = torch.stack(metrics["lpips"]).mean()

        # if psnr exceed best one so far
        # write rgb and depth images to disk all at once
        if psnr.item() > self.best_psnr:
            print(f"Saving best results so far at step {step} with PSNR {psnr}dB.")
            self.best_psnr = psnr.item()
            for i, (rgb, depth) in enumerate(zip(rgb_buffer, depth_buffer)):
                imageio.imwrite(
                    f"{self.best_dir}/rgb_{i:04d}.png", (rgb * 255).astype(np.uint8)
                )
                imageio.imwrite(
                    f"{self.best_dir}/depth_{i:04d}.png",
                    (depth * 255).astype(np.uint8),
                )
            # also save novel view render
            imageio.imwrite(
                f"{self.best_dir}/novel_view.png",
                (color_save_novel_view * 255).astype(np.uint8),
            )

            # also save current trajectory
            N = camtoworlds_all.shape[0]
            timestamps = np.linspace(0, N - 1, num=N)
            tum_poses = transformation_to_tum_format(
                camtoworlds_all.cpu().numpy(), timestamps
            )
            save_to_tum_file(tum_poses, f"{self.best_dir}/poses.txt")

        print(
            f"PSNR: {psnr.item():.3f}, SSIM: {ssim.item():.4f}, LPIPS: {lpips.item():.3f} "
            f"Time: {ellipse_time:.3f}s/image "
            f"Number of GS: {len(self.splats['means3d'])}"
        )
        # save stats as json
        stats = {
            "psnr": psnr.item(),
            "ssim": ssim.item(),
            "lpips": lpips.item(),
            "ellipse_time": ellipse_time,
            "num_GS": len(self.splats["means3d"]),
        }
        with open(f"{self.stats_dir}/val_step{step:04d}.json", "w") as f:
            json.dump(stats, f)
        # save stats to tensorboard
        for k, v in stats.items():
            # self.writer.add_scalar(f"val/{k}", v, step)
            wandb.log({f"val/{k}": v}, step=step)
        # self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        # TODO: poses should be replaced if optimized
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds = self.parser.camtoworlds[5:-5]
        camtoworlds = generate_interpolated_path(camtoworlds, 1)  # [N, 3, 4]
        camtoworlds = np.concatenate(
            [
                camtoworlds,
                np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds), axis=0),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds = torch.from_numpy(camtoworlds).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        canvas_all = []
        for i in tqdm.trange(len(camtoworlds), desc="Rendering trajectory"):
            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds[i : i + 1],
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[0, ..., 0:3], 0.0, 1.0)  # [H, W, 3]
            depths = renders[0, ..., 3:4]  # [H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())

            # write images
            canvas = torch.cat(
                [colors, depths.repeat(1, 1, 3)], dim=0 if width > height else 1
            )
            canvas = (canvas.cpu().numpy() * 255).astype(np.uint8)
            canvas_all.append(canvas)

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for canvas in canvas_all:
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def render_traj_all(self, step: int):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        if self.cfg.pose_opt:
            camtoworlds = (
                self.pose_adjust.get_poses().to(torch.float32).detach().cpu().numpy()
            )
        else:
            camtoworlds = self.parser.camtoworlds[:]

        camtoworlds = generate_interpolated_path(camtoworlds, 12)  # [N, 3, 4]
        camtoworlds = np.concatenate(
            [
                camtoworlds,
                np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds), axis=0),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds = torch.from_numpy(camtoworlds).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        canvas_all = []
        for i in tqdm.trange(len(camtoworlds), desc="Rendering trajectory"):
            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds[i : i + 1],
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[0, ..., 0:3], 0.0, 1.0)  # [H, W, 3]
            depths = renders[0, ..., 3:4]  # [H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())

            # write images
            canvas = torch.cat(
                [colors, depths.repeat(1, 1, 3)], dim=0 if width > height else 1
            )
            canvas = (canvas.cpu().numpy() * 255).astype(np.uint8)
            canvas_all.append(canvas)

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for canvas in canvas_all:
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,  # active all SH degrees
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()

    def _init_viewer_state(self) -> None:
        """Initializes viewer scene with given train dataset"""
        if not cfg.disable_viewer:
            assert self.viewer and self.trainset
            self.viewer.init_scene(train_dataset=self.trainset, train_state="training")


def main(cfg: Config):
    runner = Runner(cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpt = torch.load(cfg.ckpt, map_location=runner.device)
        for k in runner.splats.keys():
            runner.splats[k].data = ckpt["splats"][k]
        runner.eval(step=ckpt["step"])
        runner.render_traj(step=ckpt["step"])
    else:
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    cfg.adjust_steps(cfg.steps_scaler)
    main(cfg)
