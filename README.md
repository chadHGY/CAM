![](https://github.com/chadHGY/CAM/blob/gh-pages/static/images/flowchart.png)

# CAM: Learning Cortical Anomaly through Masked Encoding for Unsupervised Heterogeneity Mapping

![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/chadHGY/CAM?include_prereleases)
![GitHub last commit](https://img.shields.io/github/last-commit/chadHGY/CAM)
![GitHub pull requests](https://img.shields.io/github/issues-pr/chadHGY/CAM)
![GitHub](https://img.shields.io/github/license/chadHGY/CAM)
![contributors](https://img.shields.io/github/contributors/chadHGY/CAM) 
![codesize](https://img.shields.io/github/languages/code-size/chadHGY/CAM) 

[Update 2024/05/29] Training / Inference code will be available soon.


Official PyTorch Implementation for the [Learning Cortical Anomaly through Masked Encoding for Unsupervised Heterogeneity Mapping](https://arxiv.org/abs/2312.02762).

# CAM
A new **Self-Supervised** framework designed for **Unsupervised Anomaly Detection** of brain disorders using **3D cortical surface features**.

## Getting Started
1. Clone the repo:
```bash
git clone git@github.com:chadHGY/CAM.git
cd CAM
```

2. Install the required packages:
```bash
conda create -n cam python=3.10
conda activate cam
pip install -r requirements.txt
```

## Data
To easily demonstrate the usage of CAM, we provide a toy dataset in the `data` directory. The toy dataset contains 10 subjects from [IXI dataset](https://brain-development.org/ixi-dataset/). For each subject we will extract 4 cortical surface features using FreeSurfer (Curvature, Sulci, Thickness, Volume).

1. Data Preprocessing:
Please make sure you have gone through [FreeSurfer's](https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all) `recon-all pipeline` to extract the cortical surface features. The surface features should be found under each subject's `surf` directory. You can find the already processed toy data in the `data/freesurfer` directory.

2. Data Postprocessing:
Here we provide a simple script to convert the surface features to a numpy array. 
```bash
# training set
python src/data_postprocessing.py --freesurfer_dir data/freesurfer/ --subject_list data/train_subjects.txt  --output_dir data/sphere/train/ --in_ch thickness volume curv sulc --annot_file aparc --hemi lh

# validation set
python src/data_postprocessing.py --freesurfer_dir data/freesurfer/ --subject_list data/val_subjects.txt  --output_dir data/sphere/val/ --in_ch thickness volume curv sulc --annot_file aparc --hemi lh

# testing set
python src/data_postprocessing.py --freesurfer_dir data/freesurfer/ --subject_list data/test_subjects.txt  --output_dir data/sphere/test/ --in_ch thickness volume curv sulc --annot_file aparc --hemi lh
```


## Training
```bash
python src/train.py --data_dir /path/to/your/postprocessed/data --output_dir /path/to/your/output
```

## Inference
```bash
python src/inference.py --data_dir /path/to/your/postprocessed/data --output_dir /path/to/your/output
```


# Citation
If you find this repository useful for your research, please use the following.
```
@article{yang2023learning,
  title={Learning Cortical Anomaly through Masked Encoding for Unsupervised Heterogeneity Mapping},
  author={Yang, Hao-Chun and Andreassen, Ole and Westlye, Lars Tjelta and Marquand, Andre F and Beckmann, Christian F and Wolfers, Thomas},
  journal={arXiv preprint arXiv:2312.02762},
  year={2023}
}
```


# Acknowledgments/References
1. IXI data: https://brain-development.org/ixi-dataset/
2. Sphere postprocessing code borrowed from:
    - [surface-vision-transformers](https://github.com/metrics-lab/surface-vision-transformers)
    - [SPHARM-Net](https://github.com/Shape-Lab/SPHARM-Net)
3. We would like to thank all participants in this study, making the work possible. This work was supported the German Research Foundation (DFG) Emmy Noether with reference 513851350 (TW), the Cluster of Excellence with reference 390727645 (TW) and the BMBF-funded de.NBI Cloud within the German Network for Bioinformatics Infrastructure (de.NBI) (031A532B, 031A533A, 031A533B, 031A534A, 031A535A, 031A537A, 031A537B, 031A537C, 031A537D, 031A538A).
