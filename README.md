# Attacking "Attacks Meet Interpretability"

This repository contains an attack on the the NeurIPS 2018 spotlight paper [Attacks Meet Interpretability](https://arxiv.org/abs/1902.02322).

## Prerequisite

* [opencv-python](https://pypi.org/project/opencv-python/)
* [dlib](https://pypi.org/project/dlib/)
* [caffe](http://caffe.berkeleyvision.org/)

## Setup

* Please download VGG-Face caffe model from [here](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/).
* Unzip the model under `data/` folder.

## Usage

Running `bash run_attack.sh` will take some time but should successfully generate adversarial examples.

To confirm they are indeed adversarial, run adversary_detection.ipynb. You should see at the model accuracy is 0% with a 0% detection rate.


## Citation

The citation for the attack paper is here

    @article{carlini2019ami,
        title={Is AmI (Attacks Meet Interpretability) Robust to Adversarial Examples?},
        author={Nicholas Carlini},
        year={2019},
        journal={arXiv preprint arXiv:1902.02322}
    }

The original citation for the AmI paper is here

    @inproceedings{tao2018attacks,
        title={Attacks Meet Interpretability: Attribute-steered Detection of Adversarial Samples},
        author={Tao, Guanhong and Ma, Shiqing and Liu, Yingqi and Zhang, Xiangyu},
        booktitle={Proceedings of Thirty-second Conference on Neural Information Processing Systems},
        year={2018}
    }

