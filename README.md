# landmark_detection


Implementation Landmark Detection Module 


# Reference 

Fake It Till You Make It Face analysis in the wild using synthetic data alone (ICCV2021)

3D Face Reconstruction with Dense Landmarks (ECCV 2022)


# MEMO

pip install pandas

pip install natsort

pip install retina-face

pip install "opencv-python-headless<4.3"


# ==================instruction 

# 3D Landmark detection module (Pytorch)

This project is inspired by these two papers:

[Fake It Till You Make It Face analysis in the wild using synthetic data alone (ICCV2021)](https://microsoft.github.io/FaceSynthetics/)

[3D Face Reconstruction with Dense Landmarks (ECCV 2022)](https://microsoft.github.io/DenseLandmarks/)

-----------------





----------
# Usage:

## Run on sample data:
First, the sample data(Degraded Set5) already are placed in ```<ZSRGAN_path>/datasets/MySet5```

The results will save in ```<ZSRGAN_path>/experiments/```

```
python train.py --name <save_result_path>
```
## Run on your data:
You can find  dataset 
from [Here]([https://drive.google.com/file/d/16L961dGynkraoawKE2XyiCh4pdRS-e4Y/view](https://microsoft.github.io/FaceSynthetics/)) 

First, put your data files in ```<ZSRGAN_path>/datasets/```

The results will save in ```<ZSRGAN_path>/experiments/```

```
python train.py --name <save_result_path> --dataset <name_of_your_dataset> --GT_path <HR_folder_in_your_dataset> --LR_path <LR_folder_in_your_dataset>
```

