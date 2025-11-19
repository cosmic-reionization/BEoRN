import numpy as np
import logging
logger = logging.getLogger(__name__)


# this batch size can easily take up to 50 GB, so be careful
PARTICLE_BATCH_SIZE = 1_000_000_000

def map_particles_to_mesh(mesh, box_size, particle_positions):
    # map them to a mesh that is LBox x LBox x LBox
    # Create a density mesh from the particle positions
    mesh_size = mesh.shape[0]
    scaling = float(mesh_size / box_size)
    particle_count = particle_positions.shape[0]

    # Convert to physical coordinates and map to grid indices - but do it in batches to avoid memory issues
    for start in range(0, particle_count, PARTICLE_BATCH_SIZE):
        end = min(start + PARTICLE_BATCH_SIZE, particle_count)
        batch_positions = particle_positions[start:end, :] * scaling

        # Clip to ensure indices are within bounds
        x = np.clip(np.round(batch_positions[:, 0]).astype(int), 0, mesh_size - 1)
        y = np.clip(np.round(batch_positions[:, 1]).astype(int), 0, mesh_size - 1)
        z = np.clip(np.round(batch_positions[:, 2]).astype(int), 0, mesh_size - 1)

        # Increment the mesh
        np.add.at(mesh, (x, y, z), 1)
