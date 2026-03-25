from __future__ import annotations

import argparse
from pathlib import Path
import re

import h5py
import numpy as np


DEFAULT_TREE_ROOT = Path("/xdisk/timeifler/yhhuang/Thesan/postprocessing/trees/LHaloTree")
DEFAULT_TREE_GLOB = "trees_sf1_*.hdf5"
DEFAULT_OUTPUT_NAME = "tree_cache.hdf5"
MASS_FIELD_CANDIDATES = (
    "SubhaloMass",
    "GroupMass",
    "Group_M_Crit200",
    "Group_M_TopHat200",
    "Group_M_Mean200",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the flattened THESAN merger-tree cache expected by "
            "beorn.load_input_data.ThesanLoader."
        )
    )
    parser.add_argument(
        "--tree-root",
        type=Path,
        default=DEFAULT_TREE_ROOT,
        help="Directory containing raw LHaloTree HDF5 chunks.",
    )
    parser.add_argument(
        "--tree-glob",
        default=DEFAULT_TREE_GLOB,
        help="Glob pattern used to find LHaloTree chunk files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HDF5 path. Defaults to <tree-root>/tree_cache.hdf5.",
    )
    parser.add_argument(
        "--mass-field",
        default=None,
        help=(
            "Mass dataset to cache. If omitted, the script picks the first "
            f"available field from: {', '.join(MASS_FIELD_CANDIDATES)}."
        ),
    )
    return parser.parse_args()


def tree_group_sort_key(name: str) -> tuple[int, str]:
    match = re.match(r"Tree(\d+)$", name)
    if match:
        return int(match.group(1)), name
    return 10**18, name


def discover_tree_files(tree_root: Path, tree_glob: str) -> list[Path]:
    files = sorted(tree_root.glob(tree_glob))
    if not files:
        raise FileNotFoundError(
            f"No tree files found in {tree_root} matching pattern {tree_glob!r}."
        )
    return files


def iter_tree_groups(file_handle: h5py.File):
    for group_name in sorted(file_handle.keys(), key=tree_group_sort_key):
        if group_name.startswith("Tree"):
            yield group_name, file_handle[group_name]


def pick_mass_field(
    tree_files: list[Path],
    explicit_mass_field: str | None,
) -> str:
    for tree_file in tree_files:
        with h5py.File(tree_file, "r") as handle:
            for _, group in iter_tree_groups(handle):
                if explicit_mass_field is not None:
                    if explicit_mass_field not in group:
                        raise KeyError(
                            f"Requested mass field {explicit_mass_field!r} not found "
                            f"in {tree_file.name}/{group.name}."
                        )
                    return explicit_mass_field

                for candidate in MASS_FIELD_CANDIDATES:
                    if candidate in group:
                        return candidate

    if explicit_mass_field is not None:
        raise KeyError(f"Requested mass field {explicit_mass_field!r} was not found.")
    raise KeyError(
        "Could not infer a usable mass field from the raw tree files. "
        f"Tried: {', '.join(MASS_FIELD_CANDIDATES)}."
    )


def count_total_nodes(tree_files: list[Path]) -> int:
    total = 0
    for tree_file in tree_files:
        with h5py.File(tree_file, "r") as handle:
            for _, group in iter_tree_groups(handle):
                total += int(group["SnapNum"].shape[0])
    return total


def main() -> None:
    args = parse_args()
    tree_root = args.tree_root
    output_path = args.output or (tree_root / DEFAULT_OUTPUT_NAME)

    tree_files = discover_tree_files(tree_root, args.tree_glob)
    mass_field = pick_mass_field(tree_files, args.mass_field)
    total_nodes = count_total_nodes(tree_files)
    sentinel_index = total_nodes

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(tree_files)} LHaloTree files under {tree_root}")
    print(f"Using mass field: {mass_field}")
    print(f"Total merger-tree nodes: {total_nodes}")
    print(f"Writing cache to: {output_path}")

    with h5py.File(output_path, "w") as out:
        tree_halo_ids = out.create_dataset(
            "tree_halo_ids",
            shape=(total_nodes + 1,),
            dtype=np.int64,
        )
        tree_snap_num = out.create_dataset(
            "tree_snap_num",
            shape=(total_nodes + 1,),
            dtype=np.int32,
        )
        tree_mass = out.create_dataset(
            "tree_mass",
            shape=(total_nodes + 1,),
            dtype=np.float32,
        )
        tree_main_progenitor = out.create_dataset(
            "tree_main_progenitor",
            shape=(total_nodes + 1,),
            dtype=np.int64,
        )

        write_offset = 0
        for tree_file in tree_files:
            print(f"Processing {tree_file.name}")
            with h5py.File(tree_file, "r") as handle:
                for group_name, group in iter_tree_groups(handle):
                    snap_num = group["SnapNum"][:]
                    node_count = int(snap_num.shape[0])
                    if node_count == 0:
                        continue

                    start = write_offset
                    stop = start + node_count

                    subhalo_number = group["SubhaloNumber"][:]
                    mass = np.asarray(group[mass_field][:], dtype=np.float32)
                    first_progenitor = np.asarray(group["FirstProgenitor"][:], dtype=np.int64)
                    mapped_progenitor = np.where(
                        first_progenitor >= 0,
                        start + first_progenitor,
                        sentinel_index,
                    )

                    tree_halo_ids[start:stop] = np.asarray(subhalo_number, dtype=np.int64)
                    tree_snap_num[start:stop] = np.asarray(snap_num, dtype=np.int32)
                    tree_mass[start:stop] = mass
                    tree_main_progenitor[start:stop] = mapped_progenitor

                    write_offset = stop

        if write_offset != total_nodes:
            raise RuntimeError(
                f"Expected to write {total_nodes} nodes, but wrote {write_offset}."
            )

        # Sentinel node used by the loader for missing progenitors.
        tree_halo_ids[sentinel_index] = -1
        tree_snap_num[sentinel_index] = -1
        tree_mass[sentinel_index] = 0.0
        tree_main_progenitor[sentinel_index] = sentinel_index

        out.attrs["mass_field"] = mass_field
        out.attrs["total_nodes"] = total_nodes
        out.attrs["sentinel_index"] = sentinel_index
        out.attrs["tree_root"] = str(tree_root)
        out.attrs["tree_glob"] = args.tree_glob

    print("Done.")


if __name__ == "__main__":
    main()
