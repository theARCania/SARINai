import numpy as np
import trimesh
from mesh_processor import load_and_clean_mesh

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
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions
    )
    return locations, index_ray

if __name__ == "__main__":
    stone = load_and_clean_mesh()
    origins, directions = create_ray_grid(stone, resolution=40)

    hit_points, hit_indices = fire_rays(stone, origins, directions)

    print(f"Number of rays fired: {len(origins)}")
    print(f"Number of hits: {len(hit_points)}")

    points_viz = trimesh.points.PointCloud(hit_points, colors=[255, 0, 0, 255])

    hit_origins = origins[hit_indices]
    scene = trimesh.Scene([stone, points_viz])
    scene.show()