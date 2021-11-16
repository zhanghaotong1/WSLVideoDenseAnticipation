## Introduction

This is a repository of our model for weakly-supervised video dense anticipation. 

More results on GTEA, Epic-Kitchens etc. will come soon and publish here.

Please refer to our paper **Weakly-Supervised Dense ActionAnticipation**, published in *The British Machine Vision Conference (BMVC), 2021*. 
Paper link: http://arxiv.org/abs/2111.07593


## How to use the code

`python main.py --dataset YOURDATASET --feature_type YOURFEATURETYPE --n_classes NUMBEROFCLASSES --observation OBSERVEPERCENTAGE --prediction PREDICTPERCENTAGE --fps VIDEOFPS --batch BATCH --model PATHTOSAVEMODEL`

Please refer to **main.py** for the meaning of each argument.

The code is written on the basis of the ECCV 2020 paper *Temporal Aggregate Representations for Long-Range Video Understanding*, which is one of the backbones we used in our paper. Please refer to this repository https://github.com/dipika-singhania/multi-scale-action-banks for the original code.

Please also contact the authors of the above ECCV paper if you need the original data.
