# STPGCN
This is an implementation of [Spatial-Temporal Position-Aware Graph Convolution Networks for Traffic Flow Forecasting (TITS, 2022)](https://ieeexplore.ieee.org/abstract/document/9945663).

## Requirements
- mxnet>=1.5.0
- easydict

Use ```nvcc -V``` to check the cuda version and install mxnet with the corresponding version. For example, use ```pip install mxnet-cu101``` to install mxnet for cuda version 10.1.

## Data
- PEMS: Refer to https://github.com/Davidham3/STSGCN
- Metro：Refer to https://github.com/yijizhao/MR-STN

## Usage
- python main.py --rid=1 --seed=1 --L=3 --a=4 --b=2 --d=8 --data=PEMS08 --batch=32 --C=64 --workname=STPGCN-PEMS08

## Citing
If our paper benefits to your research, please cite our paper using the bitex below:

    @article{zhao2022spatial,
        title={Spatial-Temporal Position-Aware Graph Convolution Networks for Traffic Flow Forecasting},
        author={Zhao, Yiji and Lin, Youfang and Wen, Haomin and Wei, Tonglong and Jin, Xiyuan and Wan, Huaiyu},
        journal={IEEE Transactions on Intelligent Transportation Systems},
        volume={24},
        number={8},
        pages={8650-8666},
        year={2023},
        publisher={IEEE}
    }
