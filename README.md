# Robust Brain Tumor Segmentation with Incomplete MRI Modalities Using Hölder Divergence and Mutual Information-Enhanced Knowledge Transfer
Official implementatation for paper: Robust Brain Tumor Segmentation with Incomplete MRI Modalities Using Hölder Divergence and Mutual Information-Enhanced Knowledge Transfer

## Environment
The required libraries are listed in `environment.yml`
```
cond create -n your_name -f environment.yml
```
## Data preparation
download [BraTS18](https://www.med.upenn.edu/sbia/brats2018/registration.html) and modify paths in `mypath.py`

## training & eval
run `sh cli/train.sh`

## no ET dataset
train  15 例   263 264 268 269 272 275 279 281 289 297 304 312 319 324 330
val     4 例   265 286 305 321
test    8 例   262 266 278 294 306 310 329 335
