# MICCAI Conference Verision, MoME: 

# A Foundation Model for Brain Lesion Segmentation with Mixture of Modality Experts
## 1. Overview
![image](https://github.com/ZhangxinruBIT/MoME/blob/main/fig/comb12.png)

Fig.1: Different paradigms for brain lesion segmentation and an overview of the proposed framework:
a) the traditional paradigm;
b) the foundation model paradigm;
c) the proposed mixture of modality experts (**MoME**) framework for constructing the foundation model.
![image](https://github.com/ZhangxinruBIT/MoME/blob/main/fig/comb34.png)
Fig.2:Detailed analysis of the MoME result on seen datasets. a) A radar chart that compares the average Dice score of foundation models from the perspectives of different modalities and lesion types. b) t-SNE plots of latent spaces for nnU-Net and  **MoME**, where each dot represents a brain image.

Please also cite this paper [[link]](https://arxiv.org/pdf/2405.10246) if you are using MoME for your research!

      @InProceedings{
      author="Zhang, Xinru and Ou, Ni and Basaran, Berke Doga and Visentin, Marco and Gu, Renyang and Ouyang, Cheng and Liu, Yaou and Matthew, Paul M.
      and Ye, Chuyang and Bai, Wenjia",
      title="A Foundation Model for Brain Lesion Segmentation with Mixture of Modality Experts",
      booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2024",
      year="2024",
      publisher="Springer International Publishing",
      }

## 2. Installation

```
conda create --name MoME python=3.9.18
conda activate MoME
git clone https://github.com/ZhangxinruBIT/MoME.git
cd MoME/MoME_foundation
pip install -e .
```
Additionally, since we have made significant changes to the third-party library named  **dynamic_network_architectures**, we did not take the [original one](https://github.com/MIC-DKFZ/dynamic-network-architectures). 

## 3. Usage Preparation

**Data Preprocessing**

We perform affine registration, skull stripping, and brain imaging cropping, along with their respective annotations. 
But make sure you have downloaded the Advanced Normalization Tools ([ANTs](https://github.com/ANTsX/ANTs)) before.
The following command can be executed after you convert the dataset to the format required by [nnU-NetV2](https://github.com/MIC-DKFZ/nnUNet.git).
```
cd MoME/Codes_prepro
python one-step.py -dataset_path -ss #If all the brain images in your dataset include the skull, please add the `-ss` flag. If none of the images include the skull, you can omit the flag. Do not mix brain images with and without skulls in the same directory, as their preprocessing steps are different.
```
**Datasplit**

Please set your own datasplit, for me we save the information in the **[datasplit.json](MoME_foundation/datasplit.json)**.
And then modify the **do_split** in [nnUNetTrainer](MoME_foundation/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py) with the code like:
```
if self.fold == 'MoME':

    tr_keys = DATA['train']['BraTS'] + DATA['train']['ATLAS'] +DATA['train']['OASIS']\
        +DATA['train']['ISLES']+DATA['train']['WMH2017']+DATA['train']['MSSEG']
    
    val_keys = DATA['val']['BraTS'] + DATA['val']['ATLAS'] +DATA['val']['OASIS']\
        +DATA['val']['ISLES']+DATA['val']['WMH2017']+DATA['val']['MSSEG'] 
```
to better manage the data split.

## 4. Usage with nnU-NetV2
For traning and inference, since we implement within the nnU-NetV2, the well-introduced usage can be follwed at [nnU-Net](https://github.com/MIC-DKFZ/nnUNet.git). Please ensure that the dataset format meets the requirements expected by nnU-Net with our preprocessed data (Rename the directories to resemble **imagesTr** and **labelsTr** under **Dataset_XXXX**.). 

**Experiment planning and preprocessing**
```
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

**Training**

Before utilizing the MoME training, make sure you have acquired pretrained modality experts trained with nnUNet using corresponding modality images. Our pretrained modality experts are available at Hugging Face [Pretrained Experts](https://huggingface.co/ZhangxinruBIT/MoME/tree/main/Pretrained_Experts).

You can adjust the path to align with the your own specialized modality nnUNets in [nnUNetTrainer.initialize(from line 214 to 245)](MoME_foundation/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py). 
Alternatively, you can utilize our well-trained experts and configure the path accordingly.
```
nnUNet_train XXX 3d_fullres MoME
```
**Inference**

When performing inference, ensure you have the checkpoint list similar to [MoME_CHECKPOINT](https://huggingface.co/ZhangxinruBIT/MoME/tree/main/MoME_CHECKPOINT).
```
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -f MoME -c 3d_fullres -chk checkpoint_best.pth
```
Additionally, the final MoME model has been released on Hugging Face at [MoME_CHECKPOINT](https://huggingface.co/ZhangxinruBIT/MoME/tree/main/MoME_CHECKPOINT).

# TMI Journal Extension Verision, MoME+: 
## 1. Overview
The MoME model has been extended to MoME+ to handle combined multiple modalities as input. To this end, we develop a novel dispatch network with a soft assignment strategy and prior constraints, and employ random dropout of modalities during training to mimic the heterogeneous input.
![image](https://github.com/ZhangxinruBIT/MoME/blob/main/fig/Extension.png)

## 2. Installation
MoME+ and MoME can share the same Conda environment. However, each time you switch between MoME and MoME+, you need to navigate to the corresponding folder and run **pip install -e .** to properly link the package.

Assuming you have already set up the MoME virtual environment, follow these steps to activate it and configure MoME+:
```
conda activate MoME
cd MoME/MoME_plus
pip install -e .
```
This ensures that the correct package is linked within the environment. Always remember to repeat this process whenever switching between MoME and MoME+.

## 3. Usage Preparation
.......

## 4. Usage with nnU-NetV2
**Training**

```
nnUNet_train XXX 3d_fullres .......
```
**Inference**

```
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -f MoME -c 3d_fullres -chk checkpoint_best.pth ......
```
