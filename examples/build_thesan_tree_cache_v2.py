from __future__ import annotations

import argparse
from pathlib import Path
import re

import h5py
import numpy as np


DEFAULT_TREE_ROOT = Path("/xdisk/timeifler/yhhuang/Thesan/postprocessing/trees/LHaloTree")
DEFAULT_TREE_GLOB = "trees_sf1_*.hdf5"
DEFAULT_OUTPUT_NAME = "tree_cache_v2.hdf5"
FOF_FIELD_CANDIDATES = (
    "FirstHaloInFOFgroup",
    "FirstHaloInFOFGroup",
    "FirstHaloInFoFGroup",
    "FOFHalo",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the format-v2 THESAN merger-tree cache used by the current "
            "BEoRN ThesanLoader. This matches the newer notebook workflow and "
            "stores snapshot_redshifts plus tree_is_central."
        )
    )
    parser.add_argument(
        "--tree-root",
        type=Path,
        default=DEFAULT_TREE_ROOT,
        help="Directory containing raw LHaloTree HDF5 chunk files.",
    )
    parser.add_argument(
        "--tree-glob",
        default=DEFAULT_TREE_GLOB,
        help="Glob pattern used to discover LHaloTree files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HDF5 path. Defaults to <tree-root>/tree_cache_v2.hdf5.",
    )
    parser.add_argument(
        "--mass-field",
        choices=("LenType", "SubhaloMassType"),
        default=None,
        help="Mass field to use. Defaults to LenType when available, else SubhaloMassType.",
    )
    parser.add_argument(
        "--high-res",
        action="store_true",
        help="Label the cache as THESAN-DARK-1 instead of THESAN-DARK-2.",
    )
    return parser.parse_args()


def tree_group_sort_key(name: str) -> tuple[int, str]:
    match = re.match(r"Tree(\d+)$", name)
    if match:
        return int(match.group(1)), name
    return 10**18, name


def discover_tree_files(tree_root: Path, tree_glob: str, output_name: str) -> list[Path]:
    files = sorted(tree_root.glob(tree_glob))
    files = [path for path in files if path.name != output_name]
    if not files:
        raise FileNotFoundError(
            f"No LHaloTree files found in {tree_root} matching pattern {tree_glob!r}."
        )
    return files


def pick_mass_and_fof_fields(tree_file: Path, explicit_mass_field: str | None) -> tuple[str, str]:
    with h5py.File(tree_file, "r") as handle:
        tree0 = handle["Tree0"]
        fields = set(tree0.keys())

    mass_field = explicit_mass_field
    if mass_field is None:
        mass_field = "LenType" if "LenType" in fields else "SubhaloMassType"
    if mass_field not in fields:
        raise KeyError(
            f"Requested mass field {mass_field!r} not found in {tree_file.name}/Tree0."
        )

    fof_field = next((name for name in FOF_FIELD_CANDIDATES if name in fields), None)
    if fof_field is None:
        raise KeyError(
            f"Cannot find a FoF-first-halo pointer in {sorted(fields)}."
        )

    return mass_field, fof_field


def main() -> None:
    args = parse_args()
    tree_root = args.tree_root
    output_path = args.output or (tree_root / DEFAULT_OUTPUT_NAME)

    tree_files = discover_tree_files(tree_root, args.tree_glob, output_path.name)
    mass_field, fof_field = pick_mass_and_fof_fields(tree_files[0], args.mass_field)

    print(f"Found {len(tree_files)} LHaloTree files under {tree_root}")
    print(f"Using mass field: {mass_field}")
    print(f"Using FoF-central field: {fof_field}")
    print(f"Writing cache to: {output_path}")

    with h5py.File(tree_files[0], "r") as handle:
        snapshot_redshifts = handle["Header"]["Redshifts"][:].astype(np.float32)
        particle_mass_1e10 = float(handle["Header"].attrs["ParticleMass"])
        hubble_param = float(handle["Header"].attrs.get("HubbleParam", 0.6774))

    print(f"Header reports {snapshot_redshifts.size} snapshots")

    all_halo_ids: list[np.ndarray] = []
    all_snap_nums: list[np.ndarray] = []
    all_masses: list[np.ndarray] = []
    all_main_prog: list[np.ndarray] = []
    all_is_central: list[np.ndarray] = []
    global_offset = 0

    for tree_file in tree_files:
        file_offset = 0
        with h5py.File(tree_file, "r") as handle:
            tree_keys = sorted(
                (key for key in handle.keys() if key.startswith("Tree")),
                key=tree_group_sort_key,
            )

            for tree_key in tree_keys:
                tree = handle[tree_key]
                n_local = int(tree["SnapNum"].shape[0])
                if n_local == 0:
                    continue

                snap_num = tree["SnapNum"][:].astype(np.int32)
                subhalo_index = tree["SubhaloNumber"][:].astype(np.int64)
                first_prog = tree["FirstProgenitor"][:].astype(np.int64)

                if mass_field == "LenType":
                    mass = tree["LenType"][:, 1].astype(np.float32) * particle_mass_1e10
                else:
                    mass = tree["SubhaloMassType"][:, 1].astype(np.float32)

                fof_first = tree[fof_field][:].astype(np.int64)
                local_idx = np.arange(n_local, dtype=np.int64)
                is_central = fof_first == local_idx

                abs_offset = global_offset + file_offset
                main_prog = np.where(first_prog >= 0, first_prog + abs_offset, np.int64(-1))

                all_halo_ids.append(subhalo_index)
                all_snap_nums.append(snap_num)
                all_masses.append(mass)
                all_main_prog.append(main_prog)
                all_is_central.append(is_central)
                file_offset += n_local

        global_offset += file_offset
        print(
            f"Processed {tree_file.name}: {file_offset:,} entries "
            f"(cumulative {global_offset:,})"
        )

    tree_halo_ids = np.concatenate(all_halo_ids).astype(np.int64)
    tree_snap_num = np.concatenate(all_snap_nums).astype(np.int32)
    tree_mass = np.concatenate(all_masses).astype(np.float32)
    tree_main_progenitor = np.concatenate(all_main_prog).astype(np.int64)
    tree_is_central = np.concatenate(all_is_central)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as out:
        out.attrs["format_version"] = 2
        out.attrs["generated_by"] = "beorn"
        out.attrs["simulation_code"] = "THESAN-DARK-1" if args.high_res else "THESAN-DARK-2"
        out.attrs["is_high_res"] = args.high_res
        out.attrs["hubble_param"] = hubble_param
        out.attrs["mass_field_used"] = mass_field
        out.attrs["fof_field_used"] = fof_field
        out.attrs["n_snapshots"] = int(snapshot_redshifts.size)
        out.attrs["n_tree_files"] = len(tree_files)
        out.attrs["n_total_entries"] = int(tree_halo_ids.size)
        out.attrs["particle_mass_1e10_Msol_per_h"] = particle_mass_1e10

        out.create_dataset("snapshot_redshifts", data=snapshot_redshifts, compression="gzip")
        out.create_dataset("tree_halo_ids", data=tree_halo_ids, compression="gzip")
        out.create_dataset("tree_snap_num", data=tree_snap_num, compression="gzip")
        out.create_dataset("tree_mass", data=tree_mass, compression="gzip")
        out.create_dataset("tree_main_progenitor", data=tree_main_progenitor, compression="gzip")
        out.create_dataset("tree_is_central", data=tree_is_central, compression="gzip")

    print(
        f"Tree cache saved to {output_path} "
        f"({tree_halo_ids.size:,} total entries, {tree_is_central.sum():,} centrals)"
    )


if __name__ == "__main__":
    main()
