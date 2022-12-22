"""
线性插值
参考：https://blog.csdn.net/weixin_42795611/article/details/111566400
"""

import math
import torch
import torch.nn.functional as F


def linear_1d(c0, c1, x):
    """
    1-D的线性插值
    :param c0: x=0处的值
    :param c1: x=1处的值
    :param x: 位置x
    :return: 插值
    """
    c = c0 * (1 - x) + c1 * x
    return c


def linear_2d(c00, c01, c10, c11, x, y):
    """
    2-D的线性插值
    :param c00: (0,0)处的值
    :param c01: (0,1)处的值
    :param c10: (1,0)处的值
    :param c11: (1,1)处的值
    :param x: x
    :param y: y
    :return: 插值
    """
    c0 = c00 * (1 - x) + c10 * x
    c1 = c01 * (1 - x) + c11 * x

    return linear_1d(c0, c1, y)


def linear_3d(c000, c001, c010, c011, c100, c101, c110, c111, x, y, z):
    """
    3-D的线性插值
    :param c000: (0,0,0)处的值
    :param c001: (0,0,1)处的值
    :param c010: (0,1,0)处的值
    :param c011: (0,1,1)处的值
    :param c100: (1,0,0)处的值
    :param c101: (1,0,1)处的值
    :param c110: (1,1,0)处的值
    :param c111: (1,1,1)处的值
    :param x: x
    :param y: y
    :param z: z
    :return: 插值
    """
    c00 = c000 * (1 - x) + c100 * x
    c01 = c001 * (1 - x) + c101 * x
    c10 = c010 * (1 - x) + c110 * x
    c11 = c011 * (1 - x) + c111 * x

    return linear_2d(c00, c01, c10, c11, y, z)


def test_1(grid: torch.Tensor, x, y, z):
    x0, y0, z0 = math.floor(x), math.floor(y), math.floor(z)
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

    c000 = grid[z0, y0, x0]
    c001 = grid[z1, y0, x0]
    c010 = grid[z0, y1, x0]
    c011 = grid[z1, y1, x0]
    c100 = grid[z0, y0, x1]
    c101 = grid[z1, y0, x1]
    c110 = grid[z0, y1, x1]
    c111 = grid[z1, y1, x1]

    return linear_3d(c000, c001, c010, c011, c100, c101, c110, c111, x - x0, y - y0, z - z0)


def test_2(grid: torch.Tensor, x, y, z):
    D, H, W = grid.shape
    indices = torch.tensor([x * 2 / (W - 1) - 1, y * 2 / (H - 1) - 1, z * 2 / (D - 1) - 1], dtype=torch.float32)

    grid = grid.unsqueeze(0).unsqueeze(0)
    indices.unsqueeze_(0).unsqueeze_(0).unsqueeze_(0).unsqueeze_(0)

    result = F.grid_sample(grid, indices, align_corners=True)
    return result[0, 0, 0, 0, 0]


def test_3(grid: torch.Tensor, points: torch.Tensor):
    D, H, W = grid.shape
    N, _ = points.shape

    indices = torch.empty_like(points)
    indices[:, 0] = points[:, 0] * 2 / (W - 1) - 1
    indices[:, 1] = points[:, 1] * 2 / (H - 1) - 1
    indices[:, 2] = points[:, 2] * 2 / (D - 1) - 1

    grid = grid.unsqueeze(0).unsqueeze(0)
    indices = indices.unsqueeze(1).unsqueeze(1).unsqueeze(1)

    grid = grid.repeat(N, 1, 1, 1, 1)

    result = F.grid_sample(grid, indices, align_corners=True)
    return result[:, 0, 0, 0, 0]


def test_4(grid: torch.Tensor, *points):
    return test_3(grid, torch.tensor(points, dtype=torch.float32))


def test_5(grid: torch.Tensor, points: torch.Tensor):
    """

    :param grid: CxDxHxW
    :param points: NxRxPx3
    :return: NxRxPxC
    """
    C, D, H, W = grid.shape
    N, R, P, _ = points.shape

    grid = grid.unsqueeze(0).repeat(N, 1, 1, 1, 1)

    points = points.unsqueeze(1)
    indices = torch.empty_like(points)
    indices[..., 0] = points[..., 0] * 2 / (W - 1) - 1
    indices[..., 1] = points[..., 1] * 2 / (H - 1) - 1
    indices[..., 2] = points[..., 2] * 2 / (D - 1) - 1

    result = F.grid_sample(grid, indices, align_corners=True)
    return result[:, 0, ...]


def main():
    grid_size = 100
    grid = torch.zeros((grid_size, grid_size, grid_size))
    for x in range(grid_size):
        grid[:, :, x] = x

    print(test_1(grid, 1.2, 1.2, 1.1))
    print(test_2(grid, 1.2, 1.2, 1.1))
    print(test_3(grid, torch.tensor([(1.2, 1.2, 1.1), (1.3, 1.2, 1.1)], dtype=torch.float32)))
    print(test_4(grid, (1.2, 1.2, 1.1), (1.3, 1.2, 1.1)))
    print(test_5(grid.unsqueeze(0), torch.tensor([[[(1.2, 1.2, 1.1), (1.3, 1.2, 1.1)]]], dtype=torch.float32)))


if __name__ == '__main__':
    main()
