import torch
import numpy as np
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer.mesh.textures import TexturesUV
from pytorch3d.io import IO
from pytorch3d.ops.sample_points_from_meshes import sample_points_from_meshes
from rendering.utils import get_phong_renderer
import matplotlib.pyplot as plt
import open3d as o3d

def normalize_mesh(mesh):
    verts = mesh.verts_packed()
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-center)
    mesh.scale_verts_((0.75 / float(scale))) # put in 0.75-size box
    return


def glb_to_py3d(path):
    import trimesh
    trimesh_scene = trimesh.load(path)
    meshes_list = []
    for node in list(trimesh_scene.geometry.keys()):
        geometry = trimesh_scene.geometry[node]
        # skip if it doesn't have faces (e.g. Path3D is curves in 3D)
        if not hasattr(geometry, 'faces'):
            continue
        verts = torch.tensor(geometry.vertices).float()
        try:
            transform_tuple = trimesh_scene.graph.get(node)
            assert transform_tuple[1] == node
            transform_mat = torch.tensor(transform_tuple[0]).float()
            verts_transformed = (transform_mat @ torch.cat([verts, torch.ones(verts.shape[0],1)], axis=1).T)[:3,:].T   
        except Exception:
            verts_transformed = verts
        faces = torch.tensor(geometry.faces)
        
        if geometry.visual.material.baseColorTexture and len(np.asarray(geometry.visual.material.baseColorTexture).shape)>=3:
            # convert to rgb if greyscale
            if geometry.visual.material.baseColorTexture.mode in ["LA", "L"]:
                geometry.visual.material.baseColorTexture = geometry.visual.material.baseColorTexture.convert("RGB")
            texture_map = (torch.tensor(np.asarray(geometry.visual.material.baseColorTexture))/255)[:,:,:3]
            verts_uv = torch.tensor(geometry.visual.uv).float()
            # some uv needs to be wrapped around
            if verts_uv.min() < 0:
                verts_uv = (verts_uv - verts_uv.min(0, keepdim=True)[0]) / (verts_uv.max(0, keepdim=True)[0] - verts_uv.min(0, keepdim=True)[0])
            cur_mesh = Meshes(verts=[verts_transformed], faces=[faces], textures=TexturesUV(maps=[texture_map.float()], faces_uvs=[faces], verts_uvs=[verts_uv]))
            meshes_list.append(cur_mesh)
        else:
            # no texture map, take main color, but this still needs to be of type TexturesUV because py3d asks that
            color = torch.tensor(geometry.visual.material.main_color[:3])/255 # this is rgba
            texture_map = torch.tile(color, (100, 100, 1)).float()
            verts_uv = torch.ones(verts_transformed.shape[0],2)*0.5
            try:
                verts_uv = torch.tensor(geometry.visual.uv).float()
            except Exception:
                pass
            cur_mesh = Meshes(verts=[verts_transformed], faces=[faces], textures=TexturesUV(maps=[texture_map], faces_uvs=[faces], verts_uvs=[verts_uv]))
            meshes_list.append(cur_mesh)
    if len(meshes_list) == 0:
        raise Exception("all meshes fail to meet criteria")
    overall_mesh = join_meshes_as_scene(meshes_list)
    normalize_mesh(overall_mesh)
    return overall_mesh

def glb_to_pcd(in_path, out_path, n_pts):
    mesh = glb_to_py3d(in_path).cuda()
    # sample pts
    pts, normals, textures = sample_points_from_meshes(mesh, n_pts, return_normals= True, return_textures = True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.squeeze().cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(textures.squeeze().cpu().numpy())
    pcd.normals = o3d.utility.Vector3dVector(normals.squeeze().cpu().numpy())
    o3d.io.write_point_cloud(out_path, pcd)   
    return

def obj_to_pcd(in_path, out_path, n_pts):
    print(in_path)
    mesh = IO().load_mesh(in_path, device='cuda')
    print(mesh.verts_packed())
    pts, normals, textures = sample_points_from_meshes(mesh, n_pts, return_normals= True, return_textures = True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.squeeze().cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(textures.squeeze().cpu().numpy())
    pcd.normals = o3d.utility.Vector3dVector(normals.squeeze().cpu().numpy())
    o3d.io.write_point_cloud(out_path, pcd)
    return

def visualize_save_mesh(mesh, caption, path):
    renderer = get_phong_renderer(500, 0, 1, 0, 0, dist=10, device="cuda")
    rgb = renderer(mesh)[0,:,:,:3].detach().cpu()
    
    plt.clf()
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    plt.axis("off")
    #plt.title(caption, fontsize=18, wrap=False)
    plt.text(-0.5, 0, caption, fontsize=18, wrap=True)
    plt.savefig(path)
    plt.close()