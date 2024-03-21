import os
import shutil
import argparse
import subprocess

def print_text_structure(text):
    lines = text.split('\n')
    for line in lines:
        print(line)
text = """
nnUNet/nnUNet_raw/Dataset100_XXXX/
├── imagesTr
│   ├── ATLAS_001_0000.nii.gz
│   ├── ATLAS_002_0000.nii.gz
│   ├── ATLAS_003_0000.nii.gz
└── labelsTr
    ├── ATLAS_001.nii.gz
    ├── ATLAS_002.nii.gz
    ├── ATLAS_003.nii.gz
"""
parser = argparse.ArgumentParser()
parser.add_argument('-dataset_path', type=str,
                    help="Dataset path to preprocess with")
parser.add_argument('-ss', action='store_true',
					help='Skull stripping apply to the original images with skull on')

opt = parser.parse_args()
dataset_path = opt.dataset_path
skull_stripping = opt.ss

print('please make sure you dataset is following nnU-Net required structure is expected as:')
print_text_structure(text)
# print(' \
# ├── imagesTr \
# │   ├── ATLAS_001_0000.nii.gz \
# │   ├── ATLAS_002_0000.nii.gz \
# │   ├── ATLAS_003_0000.nii.gz \
# │   ├── ...                   \
# └── labelsTr                  \
# │   ├── ATLAS_001.nii.gz      \
# │   ├── ATLAS_002.nii.gz      \
# │   ├── ATLAS_003.nii.gz      \
# │   ├── ...')
makesure = input('Using Enter key before the preprocessing: ')

current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)

if skull_stripping:
    os.system('python ./AffineRegPrepro.py -dataset_path %s -ss'%(dataset_path))
    os.system('python ./skull_stripping.py -dataset_path %s'%(dataset_path+'_RegMNI152'))
    os.system('python ./crop.py -dataset_path %s -ss'%(dataset_path+'_RegMNI152_skull_stripping'))
    os.rename(dataset_path+'_RegMNI152_skull_stripping',dataset_path+'_RegMNI152+skull_stripping+crop')
    if os.path.exists(dataset_path.replace('images','labels')):
        os.rename(dataset_path.replace('images','labels')+'_RegMNI152',dataset_path.replace('images','labels')+'_RegMNI152+skull_stripping+crop')
    shutil.rmtree(dataset_path+'_RegMNI152')
else:
    os.system('python ./AffineRegPrepro.py -dataset_path %s'%(dataset_path))
    os.system('python ./crop.py -dataset_path %s -ss'%(dataset_path+'_RegMNI152'))
    os.rename(dataset_path+'_RegMNI152',dataset_path+'_RegMNI152+crop')
    if os.path.exists(dataset_path.replace('images','labels')):
        os.rename(dataset_path.replace('images','labels')+'_RegMNI152',dataset_path.replace('images','labels')+'_RegMNI152+crop')
print('Done')
