
# AniXplore

This is the code for **AniXplore**, the model we proposed in*AnimeDL-2M: Million-Scale AI-Generated Anime Image Detection and Localization in Diffusion Era*

## Notice

- AniXplore is developed based on the existing framework of [IMDLBenCo](https://github.com/scu-zjz/IMDLBenCo), and utilizes part of the code from existing IMDL models cited in the AnimeDL-2M paper.
- Since we made significant modifications to the original IMDLBenCo framework, the modified framework code is included in this repository.
- If you wish to use the original IMDLBenCo framework provided by its authors, we list the modified files below for your reference:

### Modified Files

-**Data loading**:

  -`IMDLBenco/IMDLBenCo/datasets/iml_datasets.py`

  -`IMDLBenco/IMDLBenCo/datasets/abstract_dataset.py`

  -`IMDLBenco/IMDLBenCo/datasets/__init__.py`

-**Data augmentation**:

  -`IMDLBenco/IMDLBenCo/transforms/iml_transforms.py`

-**Model code**:

  -`IMDLBenco/IMDLBenCo/model_zoo/AniXplore/`

  -`IMDLBenco/IMDLBenCo/model_zoo/__init__.py`

-**Training and testing code**:

  -`IMDLBenCo/run/test_datasets_anime_test.json`

  -`IMDLBenCo/run/test-AniXplore.py`

  -`IMDLBenCo/run/train-AniXplore.py`

-**Training and testing scripts**:

  -`IMDLBenCo/run/runs/test_AniXplore.sh`

  -`IMDLBenCo/run/runs/train_AniXplore.sh`

  -`IMDLBenCo/run/runs/test_AniXploreHR.sh`

  -`IMDLBenCo/run/runs/train_AniXploreHR.sh`

## Environment Setup

- You may start by using the original `requirements.txt` from IMDLBenCo and running:

  ```bash

  pip install -e .

  ```
- If there are any issues, you can recreate the full conda environment using:

  ```bash

  conda env create -f animedl_conda_env.yml

  pip install -e .

  ```

## Dataset Setup

- Download the **AnimeDL-2M** dataset as mentioned in the paper.
- Update the dataset path in:

  -`IMDLBenCo/run/test_datasets_anime_test.json`

  - and other related scripts.

## Checkpoint Setup

- Download pretrained checkpoints from: [Google Drive Link](https://drive.google.com/drive/folders/1HQWMh0SSOL1rWTNgbTdS8Jokm-Imq0CQ?usp=sharing)
- Update the checkpoint paths in the scripts accordingly.

## Output Setup

- Modify the `base_dir` (output path) variable in the scripts to your desired output directory.

## How to Run

-**Train**:

```bash

  sh train_AniXplore.sh

```

-**Test**:

```bash

  sh test_AniXplore.sh

```
