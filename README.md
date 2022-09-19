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

### 1. Download the pretrained weights from [Here]()

### 2. Put downloaded pretrained weights at ```<module_path>/pretrained/``` 

### 3. Put your test image at ```<module_path>/test_image/``` (The size of test images should be bigger than 512x512)

Take these example commands written below:

```.bash
#General command
python test.py --datasetPath <test dataset directory> --pretrained <pretrained weight paht>\
    --saveDir <directory for saving test results> --gpu <gpu number>\
    --IsGNLL <Whether to use models trained with GNLL loss(boolean, default=False)>\
    --modelType <modelType(ResNet34 or MoblieNetv2)>

# Using ResNet34 model trained with MSE loss
python test.py --pretrained resnet_MSE.pt

# Using ResNet34 model trained with GNLL loss
python test.py --pretrained resnet_GNLL.pt --IsGNLL True

# Using MoblieNetv2 model trained with MSE loss
python test.py --pretrained moblilenet_MSE.pt --modelType MoblieNetv2

# Using MoblieNetv2 model trained with GNLL loss
python test.py --pretrained moblilenet_GNLL.pt --IsGNLL True --modelType MoblieNetv2
```

### 4. The results will save in ```<module_path>/test_result/```

-----------------

# Training the model:

## Dataset download:

## make bbox:

## training 

-----------------

# ??



