import torch
import numpy as np
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import look_at_view_transform, MeshRenderer, MeshRasterizer, SoftPhongShader, PerspectiveCameras, RasterizationSettings, PointLights, BlendParams
import matplotlib.pylab as plt
import torchvision

from utils import generate_video

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mesh = load_objs_as_meshes(["data/cow_mesh/cow.obj"], device=device)


def render_batch(dist, elev, azim, up, count):
    R, T = look_at_view_transform(dist, elev, azim, up=up)
    cameras = PerspectiveCameras(R=R, T=T, focal_length=2., device=device)

    raster_settings = RasterizationSettings(image_size=512)
    lights = PointLights(location=[(0., 0., -3.)], device=device)

    render = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            blend_params=BlendParams(background_color=(0, 0, 0))
        )
    )

    output = render(mesh.extend(count))
    images = (output * 255).type(torch.uint8)[..., :3]
    images = images.cpu().numpy()

    silhouette = (output[..., 3] > 1e-4).float().cpu().numpy()
    return {"R": R, "T": T, "images": images, "silhouette": silhouette}


def render_1():
    up = []
    for i in range(360):
        if 90 <= i < 270:
            up.append((0, -1, 0))
        else:
            up.append((0, 1, 0))

    return render_batch(2.7, list(range(360)), 0, up, len(up))


def render_2():
    return render_batch(2.7, 0, list(range(360)), [(0, 1, 0)], 360)


def render_3():
    return render_batch(2.7, 45, list(range(360)), [(0, 1, 0)], 360)


def render_4():
    return render_batch(2.7, -45, list(range(360)), [(0, 1, 0)], 360)


def concatenate(*args):
    result = {k: [] for k in args[0]}
    for item in args:
        for k in item:
            result[k].append(item[k])

    for k in list(result.keys()):
        result[k] = np.concatenate(result[k], axis=0)
    return result


def main():
    r1 = render_1()
    r2 = render_2()
    r3 = render_3()
    r4 = render_4()

    r = concatenate(r1, r2, r3, r4)
    np.savez_compressed("data/cow.npz", **r)
    generate_video(r["images"], "data/cow.mp4")


if __name__ == '__main__':
    main()
