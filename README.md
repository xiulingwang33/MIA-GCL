# MIA-GCL
This is the implementation for our paper "White-box Membership Inference Attacks against Graph Contrastive Learning", which has been submitted to ACM CCS 2023.

## Datasets

The datasets of Cora, Citeseer, Amazon-computer, Amazon-photo can be download with the package of pytroch Geometric, the datasets of Facebook, ENZYMES are provided in the file "data", the datasets of Cora, Citeseer with different density are provided in the file "data/density".

## GCL models (target model)

The original implemenations of GCL models we used in the paper can be found here:

- GRACE: https://github.com/CRIPAC-DIG/GRACE

- MVGRL: https://github.com/kavehhassani/mvgrl

- GCA: https://github.com/CRIPAC-DIG/GCA

- CCA-SSG: https://github.com/hengruizhang98/CCA-SSG

- MERIT: https://github.com/merit-gem/merit

- Thanks for the authors providing the implementations. 

## Requirements

We tested the implementations with the following reqirements:

 - PyTorch 3.8
 
 - dgl-cuda11.3 0.9.0 
 
 - torch 1.10.0+cu113
 
 - torch-geometric 2.0.3 
 
 - torch-scatter 2.0.9 
 
 - torch-sparse 0.6.15   

## Attacks against GRACE

    python 
    
## Attacks against MAGRL

    python 
    
## Attacks against GCA

    python 
    
## Attacks against CCA-SSG

    python 
    
## Attacks against MERIT

    python 
    
## Evaluate the defense mechanisms

For Noisy embedding

    python defense-laplace.py

For DP-SGD

    python defense-embedding-truncation.py



