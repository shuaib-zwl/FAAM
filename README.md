
# FAAM

faam.jpg
# Algorithm Introduction
This is an implementation for FAAM    
FAAM utilizes a lightweight 3D convolutional network combined with multi-scale fusion to model motion cues, integrating two core mechanisms to enhance the supervision quality for tiny targets. Firstly, considering that background interference presents a greater challenge than feature enhancement, the Foreground Region Selector (FRS) is designed. This mechanism utilizes a learnable mask to explicitly highlight motion-consistent foreground regions and suppress large background areas during training, thereby improving the signal-to-noise ratio of positive supervision at the source. Secondly, the Dynamic Sample Label Assignment (DSLA) strategy is introduced for robust sample selection. This strategy first achieves geometric consistency pre-selection by modeling the Effective Receptive Field (ERF) and ground-truth boxes as 2D Gaussians and ranking candidates in a scale-invariant manner via Kullback-Leibler

# Usage
## On Ubuntu
### 1. Training on a single GPU
```bash
python train.py configs/yolc.py
### 2. Training on multiple GPUs
```bash
>> ./dist_train.sh configs/yolc.py <your_gpu_num>
### eval
```bash
>> python hieum_eval_.py
