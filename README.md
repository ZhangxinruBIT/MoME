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
```
Additionally, since we have made significant changes to the third-party library named  **dynamic_network_architectures**, after executing the installation command, please manually replace the  **dynamic_network_architectures** under your MoME virtual environment. The path will be something like **/vol/biomedic3/xz2223/anaconda3/envs/MoME/lib/python3.9/site-packages/dynamic_network_architectures**.

# Usage

**Data Preprocessing**
```
cd MoME/Codes_prepro
python one-step.py -dataset_path -ss #if the brain images in your dataset are with skull, please set the -ss, otherwise omit.
```
**Datasplit**

Please set your own datasplit, for me we save the information in the **datasplit.json**.

For traning and inference, since we implement within the nnU-NetV2, the well-introduced usage can be follwed at [nnU-Net](https://github.com/MIC-DKFZ/nnUNet.git).


**Training**
```
nnUNet_train XXX 3d_fullres MoME
```
**Inference**
```

```
