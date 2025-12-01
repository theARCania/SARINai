import numpy as np
import trimesh
from mesh_processor import load_and_clean_mesh

def refract_vectors(incident, normals, n1, n2):
    r = n1/n2
    incident = incident / (np.linalg.norm(incident, axis=1, keepdims=True) + 1e-8)
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
    c = -np.sum(incident * normals, axis=1, keepdims=True)
    discriminant = 1.0 - r**2 * (1.0 - c**2)
    refracted = np.zeros_like(incident)
    discriminant = np.maximum(discriminant, 0)
    valid_mask = (discriminant >= 0).flatten()

    if np.any(valid_mask):
        r_valid = r
        c_valid = c[valid_mask]
        n_valid = normals[valid_mask]
        i_valid = incident[valid_mask]
        d_valid = discriminant[valid_mask]
        refracted[valid_mask] = r_valid * i_valid + (r_valid * c_valid - np.sqrt(d_valid)) * n_valid
    return refracted, valid_mask

def reflect_vectors(incident, normals):
    incident = incident / (np.linalg.norm(incident, axis=1, keepdims=True) + 1e-8)
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
    dot_prod = np.sum(incident * normals, axis=1, keepdims=True)
    reflected = incident - 2 * dot_prod * normals
    return reflected


def create_ray_grid(mesh, resolution=50):
    bounds = mesh.bounding_box.bounds
    min_x, min_y = bounds[0, 0], bounds[1, 0]
    max_x, max_y = bounds[1, 0], bounds[1, 1]

    padding = 0.5
    x_range = np.linspace(min_x - padding, max_x + padding, resolution)
    y_range = np.linspace(min_y - padding, max_y + padding, resolution)

    grid_x, grid_y = np.meshgrid(x_range, y_range)

    ray_origins = np.column_stack((grid_x.flatten(), grid_y.flatten(), np.full_like(grid_x.flatten(), 5)))
    ray_directions = np.tile([0, 0, -1], (len(ray_origins), 1))
    return ray_origins, ray_directions


def fire_rays(mesh, ray_origins, ray_directions):
    locations_1, index_ray_1, index_tri_1 = mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions
    )
    normals_1 = mesh.face_normals[index_tri_1]
    valid_directions = ray_directions[index_ray_1]
    directions_inside, _ = refract_vectors(valid_directions, normals_1, 1.0, 2.42)
    origins_2 = locations_1 + directions_inside * 0.001
    locations_2, index_ray_2, index_tri_2 = mesh.ray.intersects_location(
        ray_origins=origins_2,
        ray_directions=directions_inside
    )


    return locations_1, locations_2


def calculate_brilliance(mesh):
    origins, directions = create_ray_grid(mesh, resolution=40)
    locs_1, idx_1, tri_1 = mesh.ray.intersects_location(origins, directions)
    if lens(locs_1) == 0: return 0.0
    normals_1 = mesh.face_normals[tri_1]
    dirs_1 = directions[idx_1]
    dirs_inside, _ = refract_vectors(dirs_1, normals_1, 1.0, 2.42)
    origins_2 = locs_1 + (dirs_inside * 0.001)
    locs_2, idx_2, tri_2 = mesh.ray.intersects_location(origins_2, dirs_inside)
    normals_2 = mesh.face_normals[tri_2]


    dirs_2 = dirs_inside[idx_2]
    dirs_exit, valid_refraction = refract_vectors(dirs_2, normals_2, 2.42, 1.0)


    hit_z = locs_2[:, 2] 
    bottom_hits = hit_z < 0
    tir_mask = ~valid_refraction
    sparkle_rays = np.logical_and(bottom_hits, tir_mask)
    
    score = np.sum(sparkle_rays) / len(locs_1)

    return score


if __name__ == "__main__":
    stone = load_and_clean_mesh()
    origins, directions = create_ray_grid(stone, resolution=40)

    surface_hits, internal_hits = fire_rays(stone, origins, directions)
    viz_entry = trimesh.points.PointCloud(surface_hits, colors=[255, 0, 0, 255])
    viz_internal = trimesh.points.PointCloud(internal_hits, colors=[0, 255, 0, 255])

    stone.visual.face_colors = [200, 200, 255, 50]
    scene = trimesh.Scene([stone, viz_entry, viz_internal])

    scene.show()
