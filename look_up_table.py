from dataclasses import dataclass

spec = GridSpec(x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0, z_min=0.0, z_max=2.0, step=0.03)
voxels = create_voxel_grid(spec)
dims = grid_dims(spec)

lookup_tables = build_all_lookup_tables(voxels)  # your existing function
frame_files = sorted([...])

results = optimized_voxel_reconstruction_sequence(
    lookup_tables=lookup_tables,
    dims=dims,
    frame_files=frame_files,
    refresh_every=15,
    dilation_radius=2,
    min_voxels_keep=500
)

# For each frame:
for frame_name, active_idx in results.items():
    active_voxels = voxels[active_idx]
    engine_voxels = world_to_engine(active_voxels)  # your existing mapping