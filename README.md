# MIA-GCL
This is the implementation for our paper "GCL-LLeak: Link Membership Inference Attacks against Graph Contrastive Learning", which has been accepted by PoPETs 2024.

## Datasets

- The datasets of Cora, Citeseer, Amazon-computer, Amazon-photo can be download with the package of pytroch Geometric, 

- The datasets of ENZYMES, COX2, Google+ are provided in the file "data"
 
- The dataset of Facebook can be download here: https://snap.stanford.edu/data/ego-Facebook.html
 
- The datasets of Cora, Citeseer with different density are provided in the file "data/density"
 
- The datasets of Facebook - Ego and Google+ with different homophily are provided in the file "data/homophily"

## GCL models (target model)

The original implemenations of GCL models we used in the paper can be found here:

- GRACE: https://github.com/CRIPAC-DIG/GRACE

- MVGRL: https://github.com/kavehhassani/mvgrl

- GCA: https://github.com/CRIPAC-DIG/GCA

- CCA-SSG: https://github.com/hengruizhang98/CCA-SSG

- MERIT: https://github.com/merit-gem/merit

Thanks for the authors providing the implementations. 

## Requirements

We tested the implementations with the following reqirements:

 - Python 3.8
 
 - dgl-cuda11.3 0.9.0 
 
 - torch 1.10.0+cu113
 
 - torch-geometric 2.0.3 
 
 - torch-scatter 2.0.9 
 
 - torch-sparse 0.6.15   

## Attacks against GRACE

    python GRACE-mia-white.py
    
## Attacks against MAGRL

    python MVGRL-mia-white.py
    
## Attacks against GCA

    python python train-mia-white.py --device cuda:0 --dataset Cora --param local:cora.json --drop_scheme degree
    
## Attacks against CCA-SSG

    python main-cora-mia-white.py
    
## Attacks against MERIT

    python train-cora-mia.py
    
## Evaluate the defense mechanisms

For DP-SGD

    python GRACE-mia-white-dpsgd-defense.py
    
    python MVGRL-mia-white-dpsgd-defense.py
    
    python main-cora-mia-white-dpsgd-defense.py

For Noisy embedding

    python GRACE-mia-white-lap-defense.py
    
    python MVGRL-mia-white-lap-defense.py
    
    python main-cora-mia-white-lap-defense.py



