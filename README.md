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

# Getting Started

## Setup
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

3. Data Preprocessing:
Please make sure you have gone through [FreeSurfer's](https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all) `recon-all pipeline` to extract the cortical surface features. The surface features should be found under each subject's `surf` directory.


4. Data Postprocessing:
Here we provide a simple script to convert the surface features to a numpy array. 
```bash
python data_postprocessing.py --data_dir /path/to/your/freesurfer/output --output_dir /path/to/your/postprocessed/data
```


# Data
To easily demonstrate the usage of CAM, we provide a toy dataset in the `data` directory. The toy dataset contains 10 subjects, each with 3 cortical surface features extracted by FreeSurfer (Thickness, Sulc, Curvature).


# Training
```bash
python src/train.py --data_dir /path/to/your/postprocessed/data --output_dir /path/to/your/output
```

# Inference
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
3. We would like to thank all participants in this study, making the work possible. This work was supported the German Research Foundation (DFG) Emmy Noether with reference 513851350 (TW), the Cluster of Excellence with reference 390727645 (TW) and the BMBF-funded de.NBI Cloud within the German Network for Bioinformatics Infrastructure (de.NBI) (031A532B, 031A533A, 031A533B, 031A534A, 031A535A, 031A537A, 031A537B, 031A537C, 031A537D, 031A538A).
