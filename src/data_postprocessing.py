"""
https://github.com/metrics-lab/surface-vision-transformers/blob/main/tools/preprocessing.py

triangle_indices_ico_6_sub_ico_1 -> ico6_80_561
    num_patches: 80 
    num_vertices: 561 

triangle_indices_ico_6_sub_ico_2 -> ico6_320_153 
    num_patches: 320
    num_vertices: 153 
"""

# %% import
import argparse

import joblib
import pandas as pd
import pyrootutils
from joblib import Parallel, delayed
from tqdm import tqdm

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils.feature_extract import RunningStats, get_patch_data
from src.utils.spharmnet.lib.io import read_mesh

# %% args
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Extract sphere from freesurfer")

# paths
parser.add_argument(
    "--freesurfer_dir",
    type=str,
    default="data/freesurfer/",
    help="Path to FreeSurfer output directory",
)
parser.add_argument(
    "--subject_list",
    type=str,
    default="data/train_subjects.txt",
    help="List of subjects to process",
)
parser.add_argument(
    "--ico6_sphere_path",
    type=str,
    default="src/utils/ico6.vtk",
    help="Path to ico6 sphere",
)

parser.add_argument(
    "--output_dir",
    type=str,
    default="data/sphere/train/",
    help="FreeSurfer sphere output directory",
)

# features
parser.add_argument(
    "--in_ch",
    type=str,
    default=["thickness", "volume", "curv", "sulc"],
    nargs="+",
    help="List of geometry to process",
)
parser.add_argument(
    "--annot_file",
    type=str,
    default="aparc",
    choices=["aparc", "aparc.a2009s"],
    help="Manual labels (e.g. aparc for ?h.aparc.annot)",
)
parser.add_argument(
    "--hemi",
    type=str,
    default="lh",
    choices=["lh", "rh"],
    help="Hemisphere for data generation",
)
parser.add_argument(
    "--n_jobs",
    type=int,
    default=-1,
    help="# of CPU n_jobs for parallel data generation",
)
args, unknown = parser.parse_known_args()


# %% main
# ------------------------------------------------------------------------------
def main(args):
    # init
    proj_root_dir = pyrootutils.find_root()
    out_dir = proj_root_dir / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # load subject list
    with open(proj_root_dir / args.subject_list, "r") as f:
        subjects = f.read().splitlines()
    subjects = [proj_root_dir / sub for sub in subjects]

    # load ico mesh & triangle indices
    ico_v, _ = read_mesh(
        str(proj_root_dir / args.ico6_sphere_path)
    )  # ico_v: ico vertices (40962, 3)
    patch_ids_path = proj_root_dir / "src/utils/ico6_320_153.csv"
    triangle_mesh_indices = pd.read_csv(patch_ids_path)

    # extract feature
    # ------------------------------------------------------------------------------
    print(f"Extractiing {args.subject_list}: {args.in_ch}")
    sphere_data = Parallel(n_jobs=args.n_jobs)(
        delayed(get_patch_data)(
            ico_v=ico_v,
            triangle_mesh_indices=triangle_mesh_indices,
            in_ch=args.in_ch,
            annot_file=args.annot_file,
            sub=sub,
            hemi=args.hemi,
        )
        for sub in tqdm(subjects, desc=f"{args.hemi}")
    )

    # store sphere data & phenotypic data in pkl file
    # ------------------------------------------------------------------------------
    running_stats = {channel: RunningStats() for channel in args.in_ch}
    for sub_folder, feat_patches, roi_anno, structure_map in sphere_data:
        # save to pkl file
        sub = sub_folder.name
        pkl_file = f"{out_dir}/{sub}.pkl"
        joblib.dump(
            {
                "feat_patches": feat_patches,
                "roi_anno": roi_anno,
                "structure_map": structure_map,
            },
            pkl_file,
        )

        # udpate running stats (mean, std) for each channel
        for channel in args.in_ch:
            running_stats[channel].update(feat_patches[channel])

    running_stats = {
        channel: {
            "mean": running_stats[channel].get_mean(),
            "std": running_stats[channel].get_std(),
        }
        for channel in args.in_ch
    }
    print(f"Running stats: {running_stats}")
    joblib.dump(running_stats, f"{out_dir}/stats.pkl")


# %% main
if __name__ == "__main__":
    main(args)
