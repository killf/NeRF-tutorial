import torch
import numpy as np
import torch.nn.functional as F
from pytorch3d.renderer import ImplicitRenderer, NDCMultinomialRaysampler, MonteCarloRaysampler, EmissionAbsorptionRaymarcher, PerspectiveCameras, ray_bundle_to_ray_points
from tensorboardX import SummaryWriter
import torchvision
import datetime
import os

from model import DirectVoxGO
from utils import *


class Solver:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.raysampler_grid = NDCMultinomialRaysampler(image_height=512, image_width=512, n_pts_per_ray=128, min_depth=0.1, max_depth=3.0)
        self.raysampler_mc = MonteCarloRaysampler(min_x=-1.0, max_x=1.0, min_y=-1.0, max_y=1.0, n_rays_per_image=1024, n_pts_per_ray=128, min_depth=0.1, max_depth=3.0)

        self.raymarcher = EmissionAbsorptionRaymarcher()
        self.renderer_grid = ImplicitRenderer(raysampler=self.raysampler_grid, raymarcher=self.raymarcher)
        self.renderer_mc = ImplicitRenderer(raysampler=self.raysampler_mc, raymarcher=self.raymarcher)

        self.data = np.load("data/cow.npz")
        self.data_count = len(self.data["images"])
        self.batch_size = 8

        self.data_R = torch.from_numpy(self.data["R"]).to(self.device)
        self.data_T = torch.from_numpy(self.data["T"]).to(self.device)
        self.data_images = torch.from_numpy(self.data["images"]).type(torch.float32) / 255
        self.data_silhouettes = torch.from_numpy(self.data["silhouette"]).unsqueeze(dim=-1)

        xyz_min, xyz_max = self.compute_bbox_by_camera_frustum()
        self.coarse_model = DirectVoxGO(xyz_min, xyz_max).to(self.device)
        self.coarse_optimizer = torch.optim.Adam(self.coarse_model.parameters(), lr=5e-2)

        self.fine_model = None
        self.fine_optimizer = None

        self.start_epoch = 1
        self.epochs = 2000
        self.global_step = 0
        self.output_dir = "output"
        self.log_dir = os.path.join(self.output_dir, "log")
        self.log_file = os.path.join(self.output_dir, "log.txt")
        self.logger = SummaryWriter(self.log_dir)
        os.makedirs(self.log_dir, exist_ok=True)

        self.load_checkpoint()

    def train(self):
        for epoch in range(10):
            self.train_epoch(epoch, self.coarse_model, self.coarse_optimizer)

        xyz_min, xyz_max = self.compute_bbox_by_coarse_geo(self.coarse_model, 1e-3)
        print('compute_bbox_by_coarse_geo: xyz_min', xyz_min)
        print('compute_bbox_by_coarse_geo: xyz_max', xyz_max)

        self.fine_model = DirectVoxGO(xyz_min, xyz_max, stage="fine").to(self.device)
        self.fine_optimizer = torch.optim.Adam(self.fine_model.parameters(), lr=1e-3)

        self.fine_model.density.copy_from(self.coarse_model.density)
        for epoch in range(self.start_epoch, self.epochs):
            self.train_epoch(epoch, self.fine_model, self.fine_optimizer)
            self.save_checkpoint(epoch)

    def train_epoch(self, epoch, model, optimizer):
        t, c = Timer(), Counter()
        indices = torch.randperm(self.data_count)
        for i in range(0, self.data_count, self.batch_size):
            batch_idx = indices[i:i + self.batch_size]
            batch_cameras = PerspectiveCameras(R=self.data_R[batch_idx], T=self.data_T[batch_idx], focal_length=2., device=self.device)

            target_images = self.data_images[batch_idx].to(self.device)
            target_silhouettes = self.data_silhouettes[batch_idx].to(self.device)

            rendered_images_silhouettes, sampled_rays = self.renderer_mc(cameras=batch_cameras, volumetric_function=model)
            rendered_images, rendered_silhouettes = rendered_images_silhouettes.split([3, 1], dim=-1)

            silhouettes_at_rays = sample_images_at_mc_locs(target_silhouettes, sampled_rays.xys)
            sil_loss = huber(rendered_silhouettes, silhouettes_at_rays).abs().mean()

            colors_at_rays = sample_images_at_mc_locs(target_images, sampled_rays.xys)
            color_loss = huber(rendered_images, colors_at_rays).abs().mean()

            loss = color_loss + sil_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss, sil_loss, color_loss = float(loss.cpu()), float(sil_loss.cpu()), float(color_loss.cpu())
            batch_time = t.elapsed_time()
            c.append(loss=loss, sil_loss=sil_loss, color_loss=color_loss, batch_time=batch_time)
            eta = calculate_eta(self.data_count - i, c.batch_time)
            self.log(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] "
                     f"[epoch={epoch}/{self.epochs}] "
                     f"step={i + 1}/{self.data_count} "
                     f"loss={loss:.4f}/{c.loss:.4f} "
                     f"sil_loss={sil_loss:.4f}/{c.sil_loss:.4f} "
                     f"color_loss={color_loss:.4f}/{c.color_loss:.4f} "
                     f"batch_time={c.batch_time:.4f} "
                     f"| ETA {eta}",
                     end="\r",
                     to_file=True)

            self.logger.add_scalar("train/loss", float(loss), global_step=self.global_step)
            self.logger.add_scalar("train/sil_loss", float(sil_loss), global_step=self.global_step)
            self.logger.add_scalar("train/color_loss", float(color_loss), global_step=self.global_step)
            if self.global_step % 100 == 0:
                with torch.no_grad():
                    with torch.no_grad():
                        camera = PerspectiveCameras(R=self.data_R[batch_idx][:1], T=self.data_T[batch_idx][:1], focal_length=2., device=self.device)
                        rendered_image_silhouette, _ = self.renderer_grid(cameras=camera, volumetric_function=model.batched_forward)
                        rendered_image, rendered_silhouette = (rendered_image_silhouette[0].split([3, 1], dim=-1))

                    rendered_image = rendered_image.permute(2, 0, 1)
                    rendered_silhouette = rendered_silhouette.permute(2, 0, 1)

                    self.logger.add_image("train/rendered_image", rendered_image, global_step=self.global_step)
                    self.logger.add_image("train/rendered_silhouette", rendered_silhouette, global_step=self.global_step)

            self.logger.flush()
            self.global_step += 1

            t.restart()
        print()

    def save_checkpoint(self, epoch):
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "coarse_model": self.coarse_model.state_dict(),
            "coarse_optimizer": self.coarse_optimizer.state_dict(),
            "fine_model": self.fine_model.state_dict(),
            "fine_optimizer": self.fine_optimizer.state_dict()
        }

        os.makedirs(self.output_dir, exist_ok=True)
        torch.save(state, os.path.join(self.output_dir, "latest.pth"))
        # torch.save(state, os.path.join(self.output_dir, f"model_{epoch:04}.pth"))

    def load_checkpoint(self):
        file = os.path.join(self.output_dir, "latest.pth")
        if not os.path.exists(file):
            return

        state = torch.load(file)
        self.start_epoch = state["epoch"] + 1
        self.global_step = state["global_step"]
        self.coarse_model.load_state_dict(state["coarse_model"])
        self.coarse_optimizer.load_state_dict(state["coarse_optimizer"])
        self.fine_model.load_state_dict(state["fine_model"])
        self.fine_optimizer.load_state_dict(state["fine_optimizer"])

    def log(self, msg, end='\n', to_file=True):
        print(msg, end=end, flush=True)
        if to_file:
            print(msg, end='\n', flush=True, file=open(self.log_file, "a+"))

    @torch.no_grad()
    def compute_bbox_by_camera_frustum(self):
        indices, points, batch_size = torch.arange(self.data_count), [], 256
        for i in range(0, self.data_count, batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_cameras = PerspectiveCameras(R=self.data_R[batch_idx], T=self.data_T[batch_idx], focal_length=2., device=self.device)
            batch_rays = self.raysampler_grid(batch_cameras)

            min_points = batch_rays.origins + batch_rays.directions * self.raysampler_grid._min_depth
            max_points = batch_rays.origins + batch_rays.directions * self.raysampler_grid._max_depth
            batch_points = torch.stack([min_points, max_points], -2).reshape(-1, 3)
            batch_points = torch.stack([batch_points.min(0).values, batch_points.max(0).values], 0)

            points.append(batch_points)

        points = torch.concatenate(points, dim=0)
        xyz_min, xyz_max = points.min(0).values, points.max(0).values
        return xyz_min, xyz_max

    @torch.no_grad()
    def compute_bbox_by_coarse_geo(self, model: DirectVoxGO, threshold):
        interp = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, model.world_size[0], device=self.device),
            torch.linspace(0, 1, model.world_size[1], device=self.device),
            torch.linspace(0, 1, model.world_size[2], device=self.device),
        ), -1)

        dense_xyz = model.xyz_min * (1 - interp) + model.xyz_max * interp
        density = model.density(dense_xyz)
        alpha = model.activate_density(density)

        active_xyz = dense_xyz[alpha.squeeze(-1) > threshold]
        xyz_min, xyz_max = active_xyz.amin(0), active_xyz.amax(0)

        return xyz_min, xyz_max


if __name__ == '__main__':
    solver = Solver()
    solver.train()
