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

# 3D Landmark Detection Module (Pytorch)

This project is inspired by these two papers:

[Fake It Till You Make It Face analysis in the wild using synthetic data alone (ICCV2021)](https://microsoft.github.io/FaceSynthetics/)

[3D Face Reconstruction with Dense Landmarks (ECCV 2022)](https://microsoft.github.io/DenseLandmarks/)

-----------------

# Simple usage(Inference):

## Run on sample data:

Put your test image at ```<module_path>/test_image/``` (The size of test images should be bigger than 512x512)

Take the command written below:

```
python test.py
```

The results will save in ```<module_path>/test_result/```

-----------------


