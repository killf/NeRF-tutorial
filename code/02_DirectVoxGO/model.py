import torch
from torch import nn
import torch.nn.functional as F
from pytorch3d.renderer import ray_bundle_to_ray_points, RayBundle
import numpy as np


class HarmonicEmbedding(torch.nn.Module):
    def __init__(self, n_harmonic_functions=60, omega0=0.1):
        """
        Given an input tensor `x` of shape [minibatch, ... , dim],
        the harmonic embedding layer converts each feature
        in `x` into a series of harmonic features `embedding`
        as follows:
            embedding[..., i*dim:(i+1)*dim] = [
                sin(x[..., i]),
                sin(2*x[..., i]),
                sin(4*x[..., i]),
                ...
                sin(2**(self.n_harmonic_functions-1) * x[..., i]),
                cos(x[..., i]),
                cos(2*x[..., i]),
                cos(4*x[..., i]),
                ...
                cos(2**(self.n_harmonic_functions-1) * x[..., i])
            ]

        Note that `x` is also premultiplied by `omega0` before
        evaluating the harmonic functions.
        """
        super().__init__()
        self.register_buffer(
            'frequencies',
            omega0 * (2.0 ** torch.arange(n_harmonic_functions)),
        )

    def forward(self, x):
        """
        Args:
            x: tensor of shape [..., dim]
        Returns:
            embedding: a harmonic embedding of `x`
                of shape [..., n_harmonic_functions * dim * 2]
        """
        embed = (x[..., None] * self.frequencies).view(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)


class DenseGrid(nn.Module):
    def __init__(self, channels, world_size, xyz_min, xyz_max):
        super(DenseGrid, self).__init__()
        if isinstance(world_size, torch.Tensor):
            world_size = tuple(world_size.tolist())

        self.channels = channels
        self.world_size = world_size
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.grid = nn.Parameter(torch.zeros([1, channels, *world_size[::-1]]))

    def forward(self, xyz):
        """
        xyz: global coordinates to query
        """
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3)
        index = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)) * 2 - 1
        out = F.grid_sample(self.grid, index, mode='bilinear', align_corners=True)
        out = out.reshape(self.channels, -1).T.reshape(*shape, self.channels)
        return out

    @torch.no_grad()
    def copy_from(self, other):
        interp = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, self.world_size[0], device=self.grid.device),
            torch.linspace(0, 1, self.world_size[1], device=self.grid.device),
            torch.linspace(0, 1, self.world_size[2], device=self.grid.device),
        ), -1)

        dense_xyz = self.xyz_min * (1 - interp) + self.xyz_max * interp
        values = other(dense_xyz)
        values = values.permute(3, 2, 1, 0).unsqueeze(0)
        self.grid.data[:] = values


class DirectVoxGO(nn.Module):
    def __init__(self, xyz_min, xyz_max, num_voxels=1024000, num_voxels_base=1024000, alpha_init=1e-6, fast_color_threshold=1e-7, stage="coarse"):
        super().__init__()

        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.fast_color_threshold = fast_color_threshold

        self.num_voxels_base = num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1 / 3)

        self.alpha_init = alpha_init
        self.register_buffer('act_shift', torch.FloatTensor([np.log(1 / (1 - alpha_init) - 1)]))
        print('dvgo: set density bias shift to', self.act_shift)

        self._set_grid_resolution(num_voxels)

        self.density = DenseGrid(1, self.world_size, self.xyz_min, self.xyz_max)
        if stage == "coarse":
            self.color = DenseGrid(3, self.world_size, self.xyz_min, self.xyz_max)
            self.color_net = None
        else:
            self.color = DenseGrid(12, self.world_size, self.xyz_min, self.xyz_max)
            self.color_net = nn.Sequential(
                nn.Linear(12 + 27, 128), nn.ReLU(inplace=True),
                nn.Linear(128, 128), nn.ReLU(inplace=True),
                nn.Linear(128, 3)
            )
            nn.init.constant_(self.color_net[-1].bias, 0)
            self.register_buffer('view_frequencies', torch.FloatTensor([(2 ** i) for i in range(4)]))

    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1 / 3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        print('dvgo: voxel_size      ', self.voxel_size)
        print('dvgo: world_size      ', self.world_size)
        print('dvgo: voxel_size_base ', self.voxel_size_base)
        print('dvgo: voxel_size_ratio', self.voxel_size_ratio)

    def activate_density(self, density, interval=1.):
        """
        alpha = 1 - exp(-softplus(density + shift) * interval)
              = 1 - exp(-log(1 + exp(density + shift)) * interval)
              = 1 - exp(log(1 + exp(density + shift)) ^ (-interval))
              = 1 - (1 + exp(density + shift)) ^ (-interval)
        """
        return 1 - (1 + torch.exp(density + self.act_shift)) ** (-interval)

    def forward(self, ray_bundle: RayBundle, **kwargs):
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)

        interval = torch.norm(ray_bundle.directions, p=2, dim=-1).unsqueeze(-1).unsqueeze(-1)

        density = self.density(rays_points_world)
        density = self.activate_density(density, interval)

        color = self.color(rays_points_world)
        if self.color_net is None:
            color = torch.sigmoid(color)
        else:
            view_directions_emb = (ray_bundle.directions.unsqueeze(-1) * self.view_frequencies).flatten(-2)
            view_directions_emb = torch.cat([ray_bundle.directions, view_directions_emb.sin(), view_directions_emb.cos()], -1)
            if view_directions_emb.ndim == 2:
                view_directions_emb = view_directions_emb.unsqueeze(-2).repeat(1, color.shape[-2], 1)
            elif view_directions_emb.ndim == 3:
                view_directions_emb = view_directions_emb.unsqueeze(-2).repeat(1, 1, color.shape[-2], 1)
            color = torch.cat([color, view_directions_emb], -1)
            color = self.color_net(color)
            color = torch.sigmoid(color)

        return density, color

    def batched_forward(self, ray_bundle: RayBundle, n_batches: int = 16, **kwargs):
        """
        This function is used to allow for memory efficient processing
        of input rays. The input rays are first split to `n_batches`
        chunks and passed through the `self.forward` function one at a time
        in a for loop. Combined with disabling PyTorch gradient caching
        (`torch.no_grad()`), this allows for rendering large batches
        of rays that do not all fit into GPU memory in a single forward pass.
        In our case, batched_forward is used to export a fully-sized render
        of the radiance field for visualization purposes.

        Args:
            ray_bundle: A RayBundle object containing the following variables:
                origins: A tensor of shape `(minibatch, ..., 3)` denoting the
                    origins of the sampling rays in world coords.
                directions: A tensor of shape `(minibatch, ..., 3)`
                    containing the direction vectors of sampling rays in world coords.
                lengths: A tensor of shape `(minibatch, ..., num_points_per_ray)`
                    containing the lengths at which the rays are sampled.
            n_batches: Specifies the number of batches the input rays are split into.
                The larger the number of batches, the smaller the memory footprint
                and the lower the processing speed.

        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacity of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.

        """

        # Parse out shapes needed for tensor reshaping in this function.
        n_pts_per_ray = ray_bundle.lengths.shape[-1]
        spatial_size = [*ray_bundle.origins.shape[:-1], n_pts_per_ray]

        # Split the rays to `n_batches` batches.
        tot_samples = ray_bundle.origins.shape[:-1].numel()
        batches = torch.chunk(torch.arange(tot_samples), n_batches)

        # For each batch, execute the standard forward pass.
        batch_outputs = [
            self.forward(
                RayBundle(
                    origins=ray_bundle.origins.view(-1, 3)[batch_idx],
                    directions=ray_bundle.directions.view(-1, 3)[batch_idx],
                    lengths=ray_bundle.lengths.view(-1, n_pts_per_ray)[batch_idx],
                    xys=None,
                )
            ) for batch_idx in batches
        ]

        # Concatenate the per-batch rays_densities and rays_colors
        # and reshape according to the sizes of the inputs.
        rays_densities, rays_colors = [
            torch.cat(
                [batch_output[output_i] for batch_output in batch_outputs], dim=0
            ).view(*spatial_size, -1) for output_i in (0, 1)
        ]
        return rays_densities, rays_colors
