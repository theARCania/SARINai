import numpy as np
import trimesh

def load_and_clean_mesh(file_path = None):
    if file_path is None:
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    else:
        mesh = trimesh.load(file_path)

    if not mesh.is_watertight:
        mesh.fill_holes()
    else:
        print("Mesh is already watertight.")
    
    mesh.vertices -= mesh.center_mass
    return mesh

if __name__ == "__main__":
    my_stone = load_and_clean_mesh()
    my_stone.show()