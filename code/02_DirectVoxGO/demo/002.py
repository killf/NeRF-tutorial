import torch
from torch import nn
import torch.nn.functional as F


class DenseGrid(nn.Module):
    def __init__(self, channels, world_size, xyz_min, xyz_max):
        super(DenseGrid, self).__init__()
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


def main():
    grid_size = 100
    grid = DenseGrid(1, (grid_size + 1, grid_size + 2, grid_size + 3), (0, 0, 0), (grid_size, grid_size + 1, grid_size + 2))
    for x in range(grid_size + 1):
        grid.grid.data[:, :, :, x] = x

    print(grid(torch.tensor([(1.2, 1.2, 1.1), (1.3, 1.2, 1.1)], dtype=torch.float32)))


if __name__ == '__main__':
    main()
