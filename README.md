## Introduction

This is a repository of our model for weakly-supervised video dense anticipation. 

Please refer to our paper **Weakly-Supervised Dense Action Anticipation**, published in *The British Machine Vision Conference (BMVC), 2021*. 
Paper link: http://arxiv.org/abs/2111.07593


## How to use the code

`python main.py --dataset YOURDATASET --feature_type YOURFEATURETYPE --n_classes NUMBEROFCLASSES --observation OBSERVEPERCENTAGE --prediction PREDICTPERCENTAGE --fps VIDEOFPS --batch BATCH --model PATHTOSAVEMODEL`

Please refer to **main.py** for the meaning of each argument.

The code is written on the basis of the ECCV 2020 paper *Temporal Aggregate Representations for Long-Range Video Understanding*, which is one of the backbones we used in our paper. Please refer to this repository https://github.com/dipika-singhania/multi-scale-action-banks for the original code. The default arguments in **main.py** follow this paper.

Please contact the authors of the above ECCV paper if you need the original data.
If you want to use your own data, please format it as the original data, or edit **data_preprocessing.py** and **data_loader.py**.
