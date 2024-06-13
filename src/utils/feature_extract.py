#%% import
import pyrootutils
import numpy as np
import traceback
import pandas as pd
import os

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils.spharmnet.lib.io import read_annot, read_feat, read_mesh
from src.utils.spharmnet.lib.sphere import TriangleSearch


# %% extract patch data from surface
def get_patch_data(
    ico_v: np.array,
    triangle_mesh_indices: pd.DataFrame,
    in_ch: list[str] = ["area", "sphere", "thickness", "volume", "curv", "sulc", "inflated.H"],
    annot_file: str = "aparc",
    sub: str = "data/freesurfer/sub-IXI031",
    hemi: str = "lh",
) -> tuple[str, dict[np.array], list]:

    # paths
    surf_dir = sub / "surf"
    label_dir = sub / "label"

    # load native sphere
    # ------------------------------------------------------------------------------
    try:
        sphere_path = os.path.join(surf_dir, hemi + "." + "sphere")
        native_v, native_f = read_mesh(sphere_path)
    except FileNotFoundError as e:
        print(f"\tsub: {sub} | Error: File {sphere_path} not found.")
        # raise e  # Re-raise the exception to see the full traceback
        return None, None, None, None
    except Exception as e:
        print(f"\tsub: {sub} | An error occurred while reading the mesh:\n{e}")
        traceback.print_exc()
        return None, None, None, None
    try:
        tree = TriangleSearch(native_v, native_f)
        triangle_idx, bary_coeff = tree.query(ico_v)
    except Exception as e:
        print(f"\tsub: {sub} | An error occurred during triangle search and query:\n{e}")
        traceback.print_exc()
        return None, None, None, None

    # extract sphere features
    # ------------------------------------------------------------------------------
    try:
        feat_patches = {feat_name: None for feat_name in in_ch}
        for feat_name in in_ch:
            # load surface feature
            feat_path = os.path.join(surf_dir, hemi + "." + feat_name)
            try:
                feat = read_feat(feat_path)  # feat: features (115231, 1)
            except Exception as feat_read_error:
                print(f"\tsub: {sub} | Error reading feature '{feat_name}' from '{feat_path}': {feat_read_error}")
                traceback.print_exc()
                return None, None, None, None

            # remesh surf feature: 115231 -> 40962
            try:
                feat_remesh = np.multiply(feat[native_f[triangle_idx]], bary_coeff).sum(
                    -1
                )  # feat_remesh: features (40962, 1)
                assert feat_remesh.shape[0] == ico_v.shape[0], f"feat_remesh.shape[0] != ico_v.shape[0]"
            except Exception as feat_processing_error:
                print(f"\tsub: {sub} | Error processing feature '{feat_name}': {feat_processing_error}")
                traceback.print_exc()
                return None, None, None, None

            # extract triangle patches
            try:
                data = feat_remesh[triangle_mesh_indices.values].T  # num_patches x num_vertices
                feat_patches[feat_name] = data
            except Exception as feat_extract_error:
                print(f"\tsub: {sub} | Error extracting feature '{feat_name}': {feat_extract_error}")
                traceback.print_exc()
                return None, None, None, None

    except Exception as e:
        print(f"\tsub: {sub} | An error occurred during feature extraction:\n{e}")
        traceback.print_exc()
        return None, None, None, None

    # extract labels
    # ------------------------------------------------------------------------------
    try:
        # laod annotation
        num_vert = native_v.shape[0]
        label_arr = np.zeros(num_vert, dtype=np.int16)
        annot = os.path.join(label_dir, hemi + "." + annot_file + ".annot")
        try:
            vertices, label, sturcture_ls, structureID_ls = read_annot(
                annot
            )  # vertices: vertex indices (115231,), label: labels (115231,), sturcture_ls: structure names (36,), structureID_ls: structure IDs (36,)
        except Exception as annot_read_error:
            print(f"\tsub: {sub} | Error reading annotation from '{annot}': {annot_read_error}")
            traceback.print_exc()
            return None, None, None, None

        # remesh roi label: 115231 -> 40962
        try:
            label = [structureID_ls.index(l) if l in structureID_ls else 0 for l in label]
            label_arr[vertices] = label
            label_remesh = label_arr[
                native_f[triangle_idx, np.argmax(bary_coeff, axis=1)]
            ]  # label_remesh: labels (40962,)
            assert label_remesh.shape[0] == ico_v.shape[0], "label_remesh.shape[0] != ico_v.shape[0]"
        except Exception as label_processing_error:
            print(f"\tsub: {sub} | Error processing label: {label_processing_error}")
            traceback.print_exc()
            return None, None, None, None

        # extract triangle patches
        try:
            label_remesh = label_remesh[triangle_mesh_indices.values].T  # num_patches x num_vertices
        except Exception as label_extract_error:
            print(f"\tsub: {sub} | Error extracting label: {label_extract_error}")
            traceback.print_exc()
            return None, None, None, None

    except Exception as e:
        print(f"\tsub: {sub} | An error occurred during label extraction:\n{e}")
        traceback.print_exc()
        return None, None, None, None

    # extract structure map
    structure_map = list(enumerate(sturcture_ls))
    return sub, feat_patches, label_remesh, structure_map


#%% calcualte running stats
class RunningStats:
    def __init__(self):
        self.N = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, data):
        self.N += 1
        self.mean += np.mean(data)
        self.M2 += np.mean(data**2)

    def get_mean(self):
        return self.mean / self.N

    def get_std(self):
        mean = self.mean / self.N
        m2 = self.M2 / self.N
        return np.sqrt(m2 - mean**2)
# %%
