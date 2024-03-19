# MoME
A Foundation Model for Brain Lesion Segmentation with Mixture of Modality Experts

![image](https://github.com/ZhangxinruBIT/MoME/blob/main/fig/comb12.png)

Fig.1: Different paradigms for brain lesion segmentation and an overview of the proposed framework:
a) the traditional paradigm;
b) the foundation model paradigm;
c) the proposed \textit{mixture of modality experts} ($\text{MoME}$) framework for constructing the foundation model.

# Installation

```
conda create --name MoME python=3.9.18
conda activate MoME
git clone https://github.com/ZhangxinruBIT/MoME.git
cd MoME/MoME
pip install -e .
pip install git+https://github.com/ZhangxinruBIT/MoME/tree/main/dynamic_network_architectures
```
# Usage
**CarveMix augmentation method is agnostic to the network structure**
