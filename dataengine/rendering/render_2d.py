# render to 2d and keep track of face/pixel correspondence
import torch
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    PointLights,
    RasterizationSettings,
    MeshRasterizer,
    SoftPhongShader
)
import os
import json
import gzip
from PIL import Image
import numpy as np
import pandas as pd
from pytorch3d.structures import Meshes
from tqdm import tqdm
from dataengine.utils.meshutils import glb_to_py3d
from common.utils import rotate_pts
from dataengine.configs import DATA_ROOT


def get_cameras(num_views, dist, device = None):
    # up and down alternating
    elev = torch.tile(torch.tensor([30,-20]), (num_views //2,))
    azim = torch.tile(torch.tensor(np.linspace(-180, 180, num=num_views//1, endpoint=False)).float(), (1,))
    R, T = look_at_view_transform(dist, elev, azim)
    cameras = FoVOrthographicCameras(device=device, R=R, T=T)
    return cameras

def get_rasterizer(image_size, blur_radius, faces_per_pixel, cameras, device = None):
    if device is None:
        device = torch.device("cpu")
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=blur_radius,
        faces_per_pixel= faces_per_pixel,
        bin_size = 0,
        perspective_correct=False, # this is important, otherwise gradients will explode!!
    )
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )
    return rasterizer

def get_phong_shader(cameras, lights, device = None):
    if device is None:
        device = torch.device("cpu")
    shader=SoftPhongShader(
        device=device,
        cameras=cameras,
        lights=lights
    )
    return shader

def get_face_pixel_correspondence(fragments, faces):
    pix2frontface = fragments.pix_to_face[:,:,:,0]
    # note the index of the faces increments with view
    n_faces = faces.shape[0]
    pix2frontface = pix2frontface*(pix2frontface>=0) % n_faces + -1.0*(pix2frontface<0)
    # the -1 masks remain and others become in the range of [0,n_faces-1]
    return pix2frontface # this is (n_views, h,w)


# assume all uids are in this specific partition
def render_k_views(uid_list, num_views, data_root):
    root_dir = f"{data_root}/labeled/glbs"
    fp_correspondence_path = f"{root_dir}/object-paths.json.gz"
    out_dir = f"{data_root}/labeled/rendered"

    with gzip.open(fp_correspondence_path, "rb") as f:
        corr_dict = json.loads(f.read())

    class_corr = pd.read_csv(f"{data_root}/obj1lvis/metadata.csv")
    uid_classes = class_corr[class_corr["uid"].isin(uid_list)]
    del uid_list # to avoid accidentally using this later
    
    uids_neworder = uid_classes["uid"].tolist()
    classes = uid_classes["class"].tolist()
    fps = [corr_dict[uid] for uid in uids_neworder]
    
    cameras = get_cameras(num_views, 3, device = 'cuda')
    lights = PointLights(device='cuda', location=[[0.0, 0.0, -3.0]])
    rasterizer = get_rasterizer(500, 0.00001, 5, cameras, device='cuda')
    shader = get_phong_shader(cameras, lights, device="cuda")
    
    file = open("render_exceptions.txt", "a")  # append mode
    for (uid, fp, classname) in tqdm(zip(uids_neworder, fps, classes), total=len(fps)):
        try:
            mesh = glb_to_py3d(root_dir + "/" + fp).cuda()
            # most meshes need to be rotated 180 degrees by z axis
            # after this rotation, most objects are front-facing, some top are facing front
            # they need to be rotated around x axis by 90 degrees
            # since we don't know ahead of time, we render out both
            verts_rotated_v1 = rotate_pts(mesh.verts_packed(), torch.tensor([0,3.14,0]).cuda(), device="cuda")
            verts_rotated_v2 = rotate_pts(verts_rotated_v1, torch.tensor([1.57,0,0]).cuda(), device="cuda")
            verts_rotated_v3 = rotate_pts(verts_rotated_v1, torch.tensor([3.14,0,0]).cuda(), device="cuda")
            mesh_v1 = Meshes(verts=[verts_rotated_v1], faces = [mesh.faces_packed()], textures = mesh.textures)
            mesh_v2 = Meshes(verts=[verts_rotated_v2], faces = [mesh.faces_packed()], textures = mesh.textures)
            mesh_v3 = Meshes(verts=[verts_rotated_v3], faces = [mesh.faces_packed()], textures = mesh.textures)
            del mesh
            mesh_dict = {"norotate": mesh_v1, "front2top": mesh_v2, "flip": mesh_v3}
            for rotate in mesh_dict:
                mesh_new = mesh_dict[rotate]
                fragments = rasterizer(mesh_new.extend(num_views), cameras = cameras)
                pix2face = get_face_pixel_correspondence(fragments, mesh_new.faces_list()[0])
                images = shader(fragments, mesh_new.extend(num_views), cameras=cameras, lights=lights)
                cur_out_dir = f"{out_dir}/{classname}_{uid}/{rotate}"
                os.makedirs(cur_out_dir, exist_ok=True)
                os.makedirs(f"{cur_out_dir}/imgs", exist_ok=True)
                torch.save(pix2face, f"{cur_out_dir}/pix2face.pt")
                for i in range(num_views):
                    rgb = images[i,:,:,:3].cpu().numpy()*255
                    im = Image.fromarray(rgb.astype(np.uint8))
                    im.save(f"{cur_out_dir}/imgs/{i:02d}.jpeg")
                print(f"saved {cur_out_dir}")
                del mesh_new
                del fragments
                del pix2face
                del images
        except Exception as e:
            file.write(f"{classname}_{uid}, {e}\n")
    file.close()
    return


if __name__ == "__main__":
    chunk_idx = 0
    uids = pd.read_csv(f"{DATA_ROOT}/labeled/chunk_ids/chunk{chunk_idx}.csv")["uid"].tolist()
    render_k_views(uids, 10)