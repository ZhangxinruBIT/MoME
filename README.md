# MoME
A Foundation Model for Brain Lesion Segmentation with Mixture of Modality Experts

![image](https://github.com/ZhangxinruBIT/MoME/blob/main/fig/comb12.png)

Fig.1: Different paradigms for brain lesion segmentation and an overview of the proposed framework:
a) the traditional paradigm;
b) the foundation model paradigm;
c) the proposed \textit{mixture of modality experts} (**MoME**) framework for constructing the foundation model.
![image](https://github.com/ZhangxinruBIT/MoME/blob/main/fig/comb34.png)
Fig.2:Detailed analysis of the MoME result on seen datasets. a) A radar chart that compares the average Dice score of foundation models from the perspectives of different modalities and lesion types. b) t-SNE plots of latent spaces for nnU-Net and  **MoME**, where each dot represents a brain image.
# Installation

```
conda create --name MoME python=3.9.18
conda activate MoME
git clone https://github.com/ZhangxinruBIT/MoME.git
cd MoME/MoME
pip install -e .
```
Additionally, since we have made significant changes to the third-party library named  **dynamic_network_architectures**, after executing the installation command, please manually replace the  **dynamic_network_architectures** under your MoME virtual environment. The path will be something like **/vol/biomedic3/xz2223/anaconda3/envs/MoME/lib/python3.9/site-packages/dynamic_network_architectures**.

# Usage Preparation

**Data Preprocessing**

We perform affine registration, skull stripping, and brain imaging cropping, along with their respective annotations. 
The following command can be executed after you convert the dataset to the format required by [nnU-NetV2](https://github.com/MIC-DKFZ/nnUNet.git).
```
cd MoME/Codes_prepro
python one-step.py -dataset_path -ss #If the brain images in your dataset include the skull, please include the `-ss` flag; otherwise, omit it.
```
**Datasplit**

Please set your own datasplit, for me we save the information in the **[datasplit.json](https://github.com/ZhangxinruBIT/MoME/blob/main/MoME/datasplit.json)**.
And then modify the **do_split** in [nnUNetTrainer](https://github.com/ZhangxinruBIT/MoME/blob/main/MoME/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py) with the code like:
```
if self.fold == 'MoME':

    tr_keys = DATA['train']['BraTS'] + DATA['train']['ATLAS'] +DATA['train']['OASIS']\
        +DATA['train']['ISLES']+DATA['train']['WMH2017']+DATA['train']['MSSEG']
    
    val_keys = DATA['val']['BraTS'] + DATA['val']['ATLAS'] +DATA['val']['OASIS']\
        +DATA['val']['ISLES']+DATA['val']['WMH2017']+DATA['val']['MSSEG'] 
```
to better manage the data split.

# Usage with nnU-NetV2
For traning and inference, since we implement within the nnU-NetV2, the well-introduced usage can be follwed at [nnU-Net](https://github.com/MIC-DKFZ/nnUNet.git). Please ensure that the dataset format meets the requirements expected by nnU-Net with our preprocessed data (Rename the directories to resemble **imagesTr** and **labelsTr** under **Dataset_XXXX**.). 


**Experiment planning and preprocessing**
```
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

**Training**
```
nnUNet_train XXX 3d_fullres MoME
```
**Inference**
```
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION
```
